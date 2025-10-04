import os
import json
import re
import time
import uuid
import logging
import asyncio
from typing import List, Any, Dict, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_together import ChatTogether

from backend.pdf_utils.graph_db import GraphDB
from backend.pdf_utils.indexer import Indexer

logger = logging.getLogger(__name__)

#definizione dello stato del grafo langgraph
class AgentState(TypedDict):
    chat_history: List[BaseMessage]
    question: str
    intermediate_steps: List[Any]

# --- simple checkpoints (memory) ---
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "agent_checkpoints.json")
MAX_CHECKPOINTS = 50

# default number of chunks to retrieve from vector search (tunable via env AGENT_DEFAULT_K)
try:
    DEFAULT_K = int(os.getenv("AGENT_DEFAULT_K", "8"))
except Exception:
    DEFAULT_K = 8
logger.info(f"agent: DEFAULT_K for vector search set to {DEFAULT_K}")

def _load_checkpoints() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.exception("could not load checkpoints")
    return []

def _save_checkpoint(entry: Dict[str, Any]):
    try:
        cps = _load_checkpoints()
        cps.append(entry)
        cps = cps[-MAX_CHECKPOINTS:]
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(cps, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("could not save checkpoint")


# --- globals set at setup time ---
_global_graph_db: Optional[GraphDB] = None
_global_indexer: Optional[Indexer] = None
_global_active_filename: Optional[str] = None

def initialize_agent_tools_resources(graph_db: GraphDB, indexer: Indexer, active_filename: Optional[str] = None):
    global _global_graph_db, _global_indexer, _global_active_filename
    _global_graph_db = graph_db
    _global_indexer = indexer
    _global_active_filename = active_filename
    logger.debug(f"agent tools resources initialized (active_filename={active_filename})")


# --- tools ---
@tool(description="firewall: analyze user query and approve or reject")
def firewall_check(query: str) -> str:
    # simple placeholder firewall: always approve for now
    # later we will add prompt-injection checks here
    logger.info("firewall_check: validating user query")
    if not query or not str(query).strip():
        return "reject: empty query"
    # approved
    return "approve"


@tool(description="neo4j_vector_search: run vector search on chunk index and return excerpts; scoped to current document when available")
def neo4j_vector_search(query: str, k: int = DEFAULT_K) -> str:
    if _global_indexer is None or _global_graph_db is None:
        logger.error("neo4j_vector_search called before resources initialized")
        return "error: resources not initialized"
    logger.info(f"neo4j_vector_search: searching for: {query} (k={k}) filename_scope={_global_active_filename}")
    try:
        emb = _global_indexer.generate_embeddings(query)
        # If we have an active filename, scope the vector search to that document to avoid cross-doc leakage
        try:
            results = _global_graph_db.query_vector_index("chunk_embeddings_index", emb, k=k, filename=_global_active_filename)
        except TypeError:
            # fallback for older GraphDB signature without filename param
            results = _global_graph_db.query_vector_index("chunk_embeddings_index", emb, k=k)
        if not results:
            return "no_results"
        parts = []
        for i, r in enumerate(results, start=1):
            cid = r.get("chunk_id", "?")
            src = r.get("filename") or "?"
            txt = r.get("node_content", "").replace("\n", " ")[:800]
            parts.append(f"[{i}] (chunk:{cid} | file:{src}) {txt}")
        return "\n".join(parts)
    except Exception as e:
        logger.exception("neo4j_vector_search failed")
        return f"error: {e}"


# --- workflow builder (simple, doc-like) ---

def _build_agent_workflow(planner_llm: Any = None, synth_llm: Any = None, mcp_tools: List[Any] = None, mcp_loaded_tools: List[Any] = None, checkpointer: Any = None):
    # simple defaults
    planner = planner_llm
    synth = synth_llm or planner_llm

    # support callers that pass mcp tools with a different kwarg name (backwards compat)
    if mcp_loaded_tools and not mcp_tools:
        mcp_tools = mcp_loaded_tools

    # tools list: firewall + neo4j + any mcp tools
    tools = [firewall_check, neo4j_vector_search]
    if mcp_tools:
        tools.extend(mcp_tools)

    # If the planner supports tool metadata binding, give it the available tools so it
    # is less likely to hallucinate unknown tool names or output irrelevant tool calls.
    try:
        if planner is not None and hasattr(planner, "bind_tools"):
            try:
                planner.bind_tools(tools)
            except Exception:
                logger.debug("planner.bind_tools failed; continuing without binding")
    except Exception:
        pass

    # node: firewall - validate user query
    def node_firewall(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        logger.info(f"node_firewall: START - running firewall_check for question='{q}'")
        logger.debug(f"node_firewall: state keys={list(state.keys())}")
        try:
            # prefer invoke to avoid deprecated __call__ behavior; fall back to direct call
            if hasattr(firewall_check, "invoke"):
                logger.debug("node_firewall: using firewall_check.invoke()")
                res_obj = firewall_check.invoke({"query": q})
                logger.debug(f"node_firewall: firewall_check.invoke() returned {type(res_obj)}")
                # handle coroutine
                if hasattr(res_obj, "__await__"):
                    logger.debug("node_firewall: handling coroutine response")
                    loop = asyncio.new_event_loop()
                    try:
                        res_obj = loop.run_until_complete(res_obj)
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass
                res = getattr(res_obj, "content", None) or str(res_obj)
                logger.info(f"node_firewall: firewall_check result='{res}'")
            else:
                logger.debug("node_firewall: using direct firewall_check() call")
                res = firewall_check(q)
                logger.info(f"node_firewall: firewall_check result='{res}'")
        except Exception as e:
            logger.exception("firewall_check error")
            res = "reject"
        
        logger.debug(f"node_firewall: checking if result '{res}' starts with 'reject'")
        if isinstance(res, str) and res.startswith("reject"):
            # immediate final answer
            logger.info("node_firewall: REJECT - returning rejection message")
            final = AIMessage(content="your query was rejected by the firewall")
            return {"chat_history": state.get("chat_history", []) + [final]}
        
        # approved -> continue to presearch
        logger.info("node_firewall: APPROVE - continuing to presearch")
        result = {
            "chat_history": state.get("chat_history", []), 
            "question": q, 
            "intermediate_steps": state.get("intermediate_steps", []),
            "approved": True
        }
        logger.info(f"node_firewall: returning state with question='{result.get('question', '')}' and keys={list(result.keys())}")
        return result

    # node: presearch - mandatory neo4j_vector_search using the user query
    def node_presearch(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        logger.info(f"node_presearch: START - performing mandatory neo4j_vector_search for question='{q}'")
        logger.debug(f"node_presearch: state keys={list(state.keys())}")
        if not q or not q.strip():
            logger.error(f"node_presearch: question is empty or invalid! Full state={state}")
            return {"chat_history": state.get("chat_history", []), "question": q}
        try:
            # prefer invoke to avoid BaseTool.__call__ kwargs issues
            logger.debug(f"node_presearch: calling neo4j_vector_search with q={q[:120].replace('\n',' ')}")
            if hasattr(neo4j_vector_search, "invoke"):
                res_obj = neo4j_vector_search.invoke({"query": q, "k": DEFAULT_K})
                if hasattr(res_obj, "__await__"):
                    loop = asyncio.new_event_loop()
                    try:
                        res_obj = loop.run_until_complete(res_obj)
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass
                res = getattr(res_obj, "content", None) or str(res_obj)
            else:
                res = neo4j_vector_search(q, 3)
            logger.debug(f"node_presearch: neo4j_vector_search returned type={type(res)} repr={repr(res)[:400]}")
        except Exception:
            logger.exception("presearch failed")
            res = "error"
        # add tool output as ToolMessage; include the original query as metadata so
        # downstream nodes can detect whether the presearch corresponds to the
        # current question (this helps when the compiled graph resumes mid-flow).
        try:
            out_str = getattr(res, "content", None) or str(res)
        except Exception:
            out_str = str(res)
        try:
            meta = json.dumps({"q": q}, ensure_ascii=False)
            tm_content = f"__PRESEARCH__{meta}\n{out_str}"
        except Exception:
            tm_content = f"__PRESEARCH__{json.dumps({'q': str(q)})}\n{out_str}"
        tm = ToolMessage(content=tm_content, tool_call_id=f"presearch-{uuid.uuid4().hex[:8]}")
        logger.debug(f"node_presearch: appended ToolMessage with id={tm.tool_call_id} content_preview={out_str[:200].replace('\n',' ')}")
        result = {
            "chat_history": state.get("chat_history", []) + [tm], 
            "question": q,
            "intermediate_steps": state.get("intermediate_steps", [])
        }
        logger.info(f"node_presearch: returning state with question='{result.get('question', '')}' and keys={list(result.keys())}")
        return result

    # node: planner - ask ollama whether more tools are needed
    def node_planner(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        logger.debug(f"node_planner: processing question='{q}', state_keys={list(state.keys())}")
        if not q or not q.strip():
            logger.error(f"node_planner: question is empty! Full state={state}")
        # build a strict prompt: planner MUST output a single JSON object and nothing else
        # include a precise list of allowed tool names to reduce hallucination of tools
        allowed_tools_list = ", ".join([getattr(t, 'name', None) or getattr(t, '__name__', str(t)) for t in tools])
        prompt = (
            "you are a planner. OUTPUT A SINGLE VALID JSON OBJECT AND NOTHING ELSE.\n"
            "the JSON must have the key \"tool_calls\" whose value is a list of calls.\n"
            "each call is an object with fields: \"name\" (string) and \"args\" (object).\n"
            "if no tools are required, output exactly: {\"tool_calls\": []} and no additional text.\n"
            "do not output any explanatory text, commentary, or surrounding backticks — only the JSON object.\n"
            "ONLY use these tool names (case-sensitive): " + allowed_tools_list + "\n"
            "available tool example: {\"tool_calls\": [{\"name\": \"neo4j_vector_search\", \"args\": {\"query\": \"...\", \"k\": 2}}]}\n"
            f"user_question: {q}\n"
        )
        logger.debug("node_planner: invoking planner for decision")
        try:
            resp = planner.invoke([HumanMessage(content=prompt)])
            # debug: log raw planner response
            logger.info(f"node_planner: raw planner resp type={type(resp)} repr={repr(resp)[:300]}")
            text = getattr(resp, "content", None) or str(resp)
            tool_calls = getattr(resp, "tool_calls", None)
        except Exception:
            logger.exception("planner invoke failed")
            text = ""
            tool_calls = None

        # try to parse JSON fragments if planner emitted them in text. Models sometimes
        # output multiple JSON objects or trailing text, so search for all json objects
        # containing "tool_calls" and pick the last valid one (prefer last decision).
        parsed_calls = None
        if tool_calls:
            parsed_calls = tool_calls
        else:
            try:
                import re
                matches = re.findall(r"(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})", text)
                parsed_list = []
                for part in matches:
                    try:
                        parsed = json.loads(part)
                        if isinstance(parsed, dict) and "tool_calls" in parsed:
                            parsed_list.append(parsed.get("tool_calls"))
                    except Exception:
                        # skip invalid json part
                        continue
                if parsed_list:
                    # prefer the last parsed tool_calls entry
                    parsed_calls = parsed_list[-1]
                else:
                    parsed_calls = None
                had_json_fragment = bool(matches)
            except Exception:
                parsed_calls = None
                had_json_fragment = False

        logger.info(f"node_planner: text={text[:400].replace('\n',' ')} parsed_calls={repr(parsed_calls)[:800]}")

        # Defensive check: ensure that the latest presearch ToolMessage corresponds
        # to the current question. If the compiled app resumed mid-flow, the last
        # presearch may refer to a previous question and should not be trusted.
        try:
            chat = state.get("chat_history", []) or []
            # look for last ToolMessage whose content starts with our __PRESEARCH__ marker
            last_presearch_q = None
            for item in reversed(chat):
                if isinstance(item, ToolMessage) and isinstance(getattr(item, 'content', ''), str):
                    c = item.content
                    if c.startswith("__PRESEARCH__"):
                        # content format: __PRESEARCH__<json_metadata>\n<tool_output>
                        try:
                            meta_part = c.split('\n', 1)[0][len("__PRESEARCH__"):]
                            md = json.loads(meta_part)
                            last_presearch_q = md.get('q')
                        except Exception:
                            last_presearch_q = None
                        break
            if last_presearch_q is not None:
                # if mismatch, force a fresh neo4j_vector_search step
                current_question = q or state.get("question", "")
                if current_question and current_question.strip() and str(last_presearch_q).strip() != str(current_question).strip():
                    logger.info(f"node_planner: last presearch query '{last_presearch_q[:50]}' does not match current question '{current_question[:50]}'; forcing neo4j_vector_search intermediate step")
                    forced_tid = f"tc-forced-{uuid.uuid4().hex[:8]}"
                    forced_step = {"name": "neo4j_vector_search", "args": {"query": current_question, "k": DEFAULT_K}, "id": forced_tid}
                    return {"chat_history": state.get("chat_history", []), "intermediate_steps": [forced_step]}
                elif not current_question or not current_question.strip():
                    logger.error(f"node_planner: current question is empty, cannot force search. last_presearch_q='{last_presearch_q}', state={state}")
        except Exception:
            # be permissive: on any error fall back to planner normal behavior
            logger.exception("node_planner: error while validating last presearch metadata")

        # if no tool calls, decide how to proceed:
        if not parsed_calls:
            # if planner emitted pure JSON (e.g. {"tool_calls": []}) we don't want to show that raw JSON
            # to the user — instead proceed directly to synth. Otherwise, treat planner text as a
            # candidate instruction/rationale for the synth LLM.
            text_stripped = (text or "").strip()
            is_json_like = False
            try:
                if text_stripped.startswith("{") and text_stripped.endswith("}"):
                    # quick validation
                    json.loads(text_stripped)
                    is_json_like = True
            except Exception:
                is_json_like = False

            # also treat any text containing brace pairs as a JSON fragment to avoid
            # exposing planner-emitted JSON or hallucinated tool names to the user
            had_json_fragment = locals().get('had_json_fragment') or ('{' in (text or '') and '}' in (text or ''))

            if is_json_like or had_json_fragment:
                # planner returned only JSON or contained a JSON fragment — do not append raw
                # planner JSON/text to user chat. Proceed to synth using context.
                logger.info("node_planner: planner returned JSON-only or contained JSON fragment; proceeding to synth without exposing raw planner text")
                return {
                    "chat_history": state.get("chat_history", []), 
                    "question": q,
                    "intermediate_steps": state.get("intermediate_steps", []),
                    "synth": True
                }

            # otherwise store planner textual rationale as AIMessage and signal synth
            am = AIMessage(content=text)
            return {"chat_history": state.get("chat_history", []) + [am], "synth": True}

        # otherwise normalize tool calls and return them for execution
        normalized = []
        for tc in parsed_calls:
            if isinstance(tc, dict):
                name = tc.get("name")
                args = tc.get("args") or {}
            else:
                name = getattr(tc, "name", None)
                args = getattr(tc, "args", {}) or {}
            # ensure id and args dict
            tid = str(tc.get("id")) if isinstance(tc, dict) and tc.get("id") else f"tc-{uuid.uuid4().hex[:8]}"
            # sometimes planners mistakenly serialize a full JSON into args['query'] (recursive).
            # defensive: if args is not a dict, coerce it; if args['query'] looks like JSON containing tool_calls,
            # replace it with the original user question to avoid recursive self-calling.
            if not isinstance(args, dict):
                args = {"query": str(args)}
            try:
                q_in_arg = args.get("query") if isinstance(args, dict) else None
                if isinstance(q_in_arg, str):
                    s = q_in_arg.strip()
                    # detect obvious JSON-with-tool_calls embedded in the query arg
                    if s.startswith("{") and "\"tool_calls\"" in s:
                        # replace with the original user question (planner intended to call the tool with the user's question)
                        args["query"] = q
                    # also handle empty query
                    if s == "":
                        args["query"] = q
            except Exception:
                # be permissive on errors
                pass
            normalized.append({"name": name, "args": args, "id": tid})

        # Validate tool names against the agent's available tools to avoid executing arbitrary commands
        available_names = set()
        for t in tools:
            try:
                tn = (getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)).strip().lower()
                available_names.add(tn)
            except Exception:
                continue
        
        logger.debug(f"node_planner: available tool names: {sorted(available_names)}")
        logger.debug(f"node_planner: normalized tool calls before filtering: {[item.get('name') for item in normalized]}")

        filtered = []
        for item in normalized:
            nm = (item.get("name") or "").strip().lower()
            if nm in available_names:
                filtered.append(item)
            else:
                logger.warning(f"node_planner: planner requested unknown or disallowed tool '{item.get('name')}', ignoring it (available: {sorted(available_names)})")

        if not filtered:
            # no valid tool calls -> instead of proceeding directly to synth, force a
            # neo4j_vector_search call using the current user question. This ensures
            # every user question triggers a fresh vector search and a tool call
            # cycle (presearch already ran before planner, but planner may choose
            # to synth without executing further tools; forcing here guarantees
            # an additional tool execution pass when the planner didn't request any).
            current_question = q or state.get("question", "")
            logger.info(f"node_planner: no valid tool_calls after filtering; forcing neo4j_vector_search intermediate step for current question: '{current_question[:100]}'")
            try:
                forced_tid = f"tc-forced-{uuid.uuid4().hex[:8]}"
                forced_step = {"name": "neo4j_vector_search", "args": {"query": current_question, "k": DEFAULT_K}, "id": forced_tid}
                return {"chat_history": state.get("chat_history", []), "intermediate_steps": [forced_step]}
            except Exception:
                logger.exception("node_planner: failed to create forced intermediate step; falling back to synth")
                return {"chat_history": state.get("chat_history", []), "synth": True}

        return {
            "chat_history": state.get("chat_history", []), 
            "question": q,
            "intermediate_steps": filtered
        }

    # node: call_tool - execute one tool call (the last in intermediate_steps)
    def node_call_tool(state: Dict[str, Any]) -> Dict[str, Any]:
        steps = state.get("intermediate_steps", [])
        if not steps:
            return {"chat_history": state.get("chat_history", [])}
        tc = steps[-1]
        name = tc.get("name")
        args = tc.get("args", {})
        tid = tc.get("id")
        # sanitize args: ensure 'query' is a non-empty string and 'k' is an int >= 1
        try:
            if isinstance(args, dict):
                # ensure query fallback
                qval = args.get("query")
                if qval is None or (isinstance(qval, str) and not qval.strip()):
                    args["query"] = state.get("question", "")
                else:
                    # coerce to string
                    args["query"] = str(qval)

                # coerce k to int >=1
                kval = args.get("k")
                if kval is None:
                    args["k"] = DEFAULT_K
                else:
                    try:
                        args_k_float = float(kval)
                        args_k_int = max(1, int(round(args_k_float)))
                        args["k"] = args_k_int
                    except Exception:
                        args["k"] = DEFAULT_K
        except Exception:
            logger.exception("node_call_tool: error sanitizing args")

        # Defensive guarantee: if the tool is neo4j_vector_search, always search using the
        # current state's question. This prevents planner-provided args containing
        # previous queries or embedded JSON from causing repeated/incorrect searches.
        try:
            if isinstance(name, str) and "neo4j_vector_search" in name.lower():
                current_question = state.get("question", "")
                if current_question and current_question.strip():
                    args["query"] = current_question
                    logger.debug(f"node_call_tool: enforcing current question for {name}: '{current_question[:100]}'")
                elif not args.get("query", "").strip():
                    # fallback if no current question and no valid query in args
                    args["query"] = "dispositivo funzionalità"  # generic fallback
                    logger.warning(f"node_call_tool: no valid query for {name}, using generic fallback")
        except Exception:
            logger.exception("node_call_tool: failed to enforce current question for neo4j_vector_search")

        logger.info(f"node_call_tool: executing {name} with args={args}")

        # find tool by name among our tools + mcp tools
        def _match(t):
            try:
                return getattr(t, "name", None) == name or getattr(t, "__name__", None) == name or name in str(t)
            except Exception:
                return False

        all_tools = tools
        sel = next((t for t in all_tools if _match(t)), None)
        if sel is None:
            out = f"error: tool {name} not found"
        else:
            try:
                if hasattr(sel, "invoke"):
                    result = sel.invoke(args)
                    if hasattr(result, "__await__"):
                        # run coroutine
                        loop = asyncio.new_event_loop()
                        try:
                            result = loop.run_until_complete(result)
                        finally:
                            try:
                                loop.close()
                            except Exception:
                                pass
                else:
                    # call directly
                    if isinstance(args, dict) and "query" in args and "k" in args:
                        result = sel(args["query"], args.get("k"))
                    elif isinstance(args, dict) and "query" in args:
                        result = sel(args["query"])
                    else:
                        result = sel(args)
            except Exception:
                logger.exception("tool execution failed")
                result = "error"
            out = result
            # debug: log tool execution result type and repr
            try:
                logger.info(f"node_call_tool: tool {name} returned type={type(out)} repr={repr(out)[:400]}")
            except Exception:
                pass

        # ensure ToolMessage content is a string (avoid passing complex objects)
        out_str = getattr(out, "content", None) or str(out)
        # If this was a forced neo4j_vector_search intermediate step (planner forced
        # a search because the last presearch didn't match the current question),
        # tag the ToolMessage as a PRESEARCH with metadata so downstream planner
        # sees that a fresh presearch for the current question has been performed.
        try:
            tm_content = out_str
            if isinstance(name, str) and "neo4j_vector_search" in name.lower() and isinstance(tid, str) and tid.startswith("tc-forced-"):
                try:
                    meta = json.dumps({"q": state.get("question", "")}, ensure_ascii=False)
                    tm_content = f"__PRESEARCH__{meta}\n{out_str}"
                except Exception:
                    # fallback to plain output if metadata creation fails
                    tm_content = out_str
        except Exception:
            tm_content = out_str
        tm = ToolMessage(content=tm_content, tool_call_id=tid)
        # append tool output and remove the executed intermediate step
        new_steps = steps[:-1]
        return {
            "chat_history": state.get("chat_history", []) + [tm], 
            "question": state.get("question", ""),
            "intermediate_steps": new_steps
        }

    # node: synth - call together once with consolidated context
    def node_synth(state: Dict[str, Any]) -> Dict[str, Any]:
        # collect recent tool outputs and chat history, then call synth once
        chat = state.get("chat_history", [])
        # keep only last N messages to avoid token overload
        # increase history so synth sees more retrieved chunks (but still bounded)
        MAX_HISTORY = 40
        context = chat[-MAX_HISTORY:]
        # build prompt: system + context
        system = (
            "You are a helpful assistant specialized in answering questions about a document given a conversation history and retrieved document chunks. "
            "MOST IMPORTANT: Always prioritize and directly answer ONLY the CURRENT user's question with a fresh, concise, and contextually accurate answer. "
            "Avoid filler phrases like 'Sure!', 'Here’s your answer based on the document', or similar introductions — respond directly. "
            "Your goal is to provide clear, informative, and human-like answers grounded in the retrieved chunks. "
            "If the chunks contain conflicting or ambiguous information, acknowledge the uncertainty briefly and summarize the most plausible interpretation. "
            "If previous answers were generic, and new chunks provide partial clues, synthesize actionable insights grounded in those chunks without inventing unsupported details. "
            "Always maintain a neutral, professional, and cooperative tone. "
            "Use facts mentioned earlier in the conversation (for example, if the user said 'mi chiamo <name>', use that name naturally in your replies). "
            "Cite chunk numbers where appropriate using the format (chunk:<chunk_id>) AT THE END of each relevant fact or quote. "
            "When relevant, combine information across multiple chunks to build a coherent answer. "
            "If genuinely no relevant chunk content exists for the specific aspect, say so briefly and then offer what related or adjacent information IS documented. "
            "Never repeat the exact same sentence structure as a previous answer unless it is a factual citation. "
            "When explaining technical or complex concepts, favor clarity and simplicity without losing accuracy. "
            "Do not over-explain; assume the user has a basic understanding of the topic. "
            "Avoid hallucinating: only include information explicitly present in the chunks or previously stated by the user. "
            "If the user asks for summaries or step-by-step guidance, provide them in a structured but concise way. "
            "Keep responses self-contained, so they make sense without needing to reread previous messages, but remain consistent with prior context. "
        )
        # Ensure the user's current question is included explicitly as the last HumanMessage
        try:
            current_q = state.get("question") or ""
        except Exception:
            current_q = ""

        # build messages: system + context + explicit current question (helps model focus on follow-up)
        msgs = [HumanMessage(content=system)] + context
        try:
            # append explicit question as last message to guarantee visibility
            if current_q:
                msgs.append(HumanMessage(content=current_q))
        except Exception:
            pass

        # diagnostic logging: preview the messages sent to synth (truncated)
        try:
            preview = []
            for m in msgs[-12:]:
                t = type(m).__name__
                c = (getattr(m, 'content', '') or '')
                c_short = c.replace('\n', ' ')[:300]
                preview.append(f"{t}:{c_short}")
            logger.debug(f"node_synth: invoking synth llm for final answer; msgs_preview(last {len(preview)}): {preview}")
        except Exception:
            logger.exception("node_synth: error building synth preview")

        logger.debug("node_synth: invoking synth llm for final answer")
        try:
            resp = synth.invoke(msgs)
            # debug: log raw synth resp
            logger.info(f"node_synth: raw synth resp type={type(resp)} repr={repr(resp)[:400]}")
            final = getattr(resp, "content", None) or str(resp)
        except Exception:
            logger.exception("synth invoke failed")
            final = "error generating final answer"

        am = AIMessage(content=final)
        # checkpoint: save summary of this interaction
        try:
            ck = {"question": state.get("question"), "answer": final, "ts": int(time.time())}
            _save_checkpoint(ck)
        except Exception:
            logger.exception("checkpoint save failed")

        # go to end by returning chat_history with final ai message
        # include a final_content marker to ensure callers can detect final answer regardless of stream shape
        return {"chat_history": state.get("chat_history", []) + [am], "final_content": final, "final_ai": True}

    # build langgraph workflow
    wf = StateGraph(dict)
    wf.add_node("firewall", node_firewall)
    wf.add_node("presearch", node_presearch)
    wf.add_node("planner", node_planner)
    wf.add_node("call_tool", node_call_tool)
    wf.add_node("synth", node_synth)

    wf.set_entry_point("firewall")

    # transitions: firewall -> presearch -> planner -> either call_tool or synth
    wf.add_conditional_edges(
        "firewall",
        lambda s: "presearch" if s.get("approved") else END,
        {"presearch": "presearch", END: END},
    )
    wf.add_edge("presearch", "planner")
    wf.add_conditional_edges(
        "planner",
        lambda s: "call_tool" if s.get("intermediate_steps") else "synth",
        {"call_tool": "call_tool", "synth": "synth"},
    )
    wf.add_edge("call_tool", "planner")
    wf.add_edge("synth", END)

    # compile the workflow; if a checkpointer (e.g. InMemorySaver) is provided, pass it so
    # LangGraph will checkpoint state per-node during execution and support thread-scoped memory.
    if checkpointer is not None:
        app = wf.compile(checkpointer=checkpointer)
    else:
        app = wf.compile()
    logger.info("built langgraph agent (simple flow)")
    return app


async def invoke_agent(agent_app, user_message: str, chat_history: List[BaseMessage], config: Optional[Dict[str, Any]] = None) -> str:
    """Run the LangGraph agent astream and return the final AI response text.

    Behavior:
    - Accumulates any BaseMessage emitted by the astream (dicts/lists/AIMessages/etc.).
    - Ignores AI messages that were present before this run (pre-existing history).
    - Prefers an explicit 'final_content' marker returned by nodes.
    - Ensures the async generator is closed to avoid pending tasks.
    """
    initial_state = {"chat_history": chat_history or [], "question": user_message, "intermediate_steps": []}
    final_response = ""
    agen = None
    
    # Debug logging for follow-up question tracking
    logger.info(f"invoke_agent: processing question='{user_message}' with {len(chat_history or [])} messages in chat_history")
    if chat_history:
        last_msg = chat_history[-1] if chat_history else None
        if isinstance(last_msg, (HumanMessage, AIMessage)):
            logger.debug(f"invoke_agent: last message type={type(last_msg).__name__} content='{getattr(last_msg, 'content', '')[:100]}'")
    logger.debug(f"invoke_agent: initial_state keys={list(initial_state.keys())}")
    try:
        last_state = None
        accumulated: List[BaseMessage] = list(initial_state.get("chat_history", []))
        pre_existing_ai_count = sum(1 for m in accumulated if isinstance(m, AIMessage))

        def _extract_messages_from(obj) -> List[BaseMessage]:
            found: List[BaseMessage] = []
            if obj is None:
                return found
            if isinstance(obj, BaseMessage):
                found.append(obj)
                return found
            if isinstance(obj, dict):
                for v in obj.values():
                    found.extend(_extract_messages_from(v))
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    found.extend(_extract_messages_from(it))
            return found

        # --- configure run (memory by default) ---
        try:
            resume_run = True
            if config and config.get('fresh'):
                resume_run = False
        except Exception:
            resume_run = True

        if isinstance(config, dict):
            cfg_to_use = config if 'configurable' in config else {'configurable': config}
        else:
            cfg_to_use = None
        if not cfg_to_use:
            cfg_to_use = {'configurable': {'thread_id': f"session-{uuid.uuid4().hex[:8]}"}}
        if 'configurable' not in cfg_to_use or 'thread_id' not in cfg_to_use['configurable']:
            cfg_to_use['configurable'] = {'thread_id': f"session-{uuid.uuid4().hex[:8]}"}

        if resume_run:
            try:
                agen = agent_app.astream(initial_state, cfg_to_use)
                logger.info(f"invoke_agent: running with persistent thread_id={cfg_to_use['configurable']['thread_id']}")
            except TypeError:
                logger.debug("agent_app.astream rejected config; falling back to no-config run")
                agen = agent_app.astream(initial_state)
        else:
            fresh_cfg = {'configurable': {'thread_id': f"fresh-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"}}
            logger.info(f"invoke_agent: forcing fresh run with thread_id={fresh_cfg['configurable']['thread_id']}")
            agen = agent_app.astream(initial_state, fresh_cfg)

        merged_state: Dict[str, Any] = dict(initial_state)

        try:
            last_node_name = None
            async for chunk in agen:
                extracted: Dict[str, Any] = {}
                if isinstance(chunk, dict):
                    if 'values' in chunk and isinstance(chunk['values'], dict):
                        extracted = chunk['values']
                    else:
                        if len(chunk.keys()) == 1:
                            k, v = next(iter(chunk.items()))
                            if isinstance(v, dict):
                                extracted = v
                                last_node_name = k
                        if not extracted and any(k in chunk for k in ('chat_history','final_content','question','intermediate_steps')):
                            extracted = chunk
                            # try to infer a pseudo node name if one of the known keys missing
                            if last_node_name is None and 'final_content' in chunk:
                                last_node_name = 'synth'
                elif isinstance(chunk, BaseMessage):
                    extracted = {'chat_history': [chunk]}

                if extracted:
                    # question
                    q_new = extracted.get('question')
                    if isinstance(q_new, str) and q_new.strip():
                        merged_state['question'] = q_new
                    elif 'question' not in merged_state:
                        merged_state['question'] = initial_state.get('question','')
                    # chat history
                    if 'chat_history' in extracted and isinstance(extracted['chat_history'], list):
                        existing = merged_state.get('chat_history', [])
                        for m in extracted['chat_history']:
                            if not any((getattr(m,'content',None)==getattr(o,'content',None) and type(m)==type(o)) for o in existing):
                                existing.append(m)
                        merged_state['chat_history'] = existing
                    # intermediate steps
                    if 'intermediate_steps' in extracted:
                        merged_state['intermediate_steps'] = extracted.get('intermediate_steps') or []
                    # finals
                    if 'final_content' in extracted:
                        merged_state['final_content'] = extracted['final_content']
                    if extracted.get('final_ai'):
                        merged_state['final_ai'] = True

                last_state = dict(merged_state)
                logger.info(f"invoke_agent: merged_state keys={list(merged_state.keys())}")
                logger.info(f"invoke_agent: merged_state.question='{merged_state.get('question','')}' chat_len={len(merged_state.get('chat_history',[]))}")

                # accumulate messages for fallback detection
                try:
                    if extracted:
                        msgs = _extract_messages_from(extracted)
                        if msgs:
                            accumulated.extend(msgs)
                except Exception:
                    logger.exception("invoke_agent: error accumulating messages")

                if merged_state.get('final_content'):
                    final_response = merged_state.get('final_content')
                    logger.info("invoke_agent: final_content detected -> stopping loop")
                    break

                # We intentionally DO NOT early-stop on generic AIMessage appearances
                # to avoid cutting the flow before presearch/planner/synth complete.
                # Only final_content (set by synth or firewall rejection) terminates the loop.
        finally:
            try:
                if agen is not None and hasattr(agen, "aclose"):
                    await agen.aclose()
            except Exception:
                logger.debug("invoke_agent: failed to aclose agent astream")

        if final_response:
            return final_response

        # no final response found: log helpful summary and optionally try checkpoint recovery
        try:
            if accumulated:
                summary = []
                for m in accumulated[-10:]:
                    t = type(m).__name__
                    c = (getattr(m, "content", None) or str(m)).replace("\n", " ")
                    summary.append(f"{t}:{c[:200]}")
                logger.info(f"invoke_agent: no final AIMessage found. accumulated chat_history (up to 10): {summary}")
            else:
                logger.info("invoke_agent: no final AIMessage found and accumulated chat_history is empty.")
        except Exception:
            logger.exception("invoke_agent: error while logging last state summary")

        # checkpoint recovery (best-effort)
        try:
            cps = _load_checkpoints()
            if cps:
                # try to recover from last_state
                try:
                    if last_state is not None:
                        if isinstance(last_state, AIMessage):
                            final_response = getattr(last_state, "content", None) or str(last_state)
                            if final_response:
                                logger.info("invoke_agent: recovered final_response from last_state AIMessage")
                                return final_response
                        if isinstance(last_state, BaseMessage):
                            final_response = getattr(last_state, "content", None) or str(last_state)
                            if final_response:
                                logger.info("invoke_agent: recovered final_response from last_state BaseMessage")
                                return final_response
                        if isinstance(last_state, list):
                            for item in reversed(last_state):
                                if isinstance(item, AIMessage):
                                    final_response = getattr(item, "content", None) or str(item)
                                    break
                            if final_response:
                                logger.info("invoke_agent: recovered final_response from last_state list")
                                return final_response
                except Exception:
                    logger.exception("invoke_agent: error while extracting from last_state")

                # heuristics over checkpoints (conservative)
                now_ts = int(time.time())
                user_l = str(user_message).strip().lower()
                for entry in reversed(cps):
                    q = str(entry.get("question", "") or "").strip().lower()
                    ans = entry.get("answer")
                    ts = int(entry.get("ts", 0))
                    if not q or not ans:
                        continue
                    if q == user_l or q in user_l or user_l in q:
                        logger.info("invoke_agent: recovered final answer from checkpoints by containment")
                        return ans
                    try:
                        import re
                        q_words = set(re.findall(r"\w+", q))
                        u_words = set(re.findall(r"\w+", user_l))
                        if q_words and u_words:
                            overlap = q_words.intersection(u_words)
                            ratio = len(overlap) / max(1, min(len(q_words), len(u_words)))
                            if ratio >= 0.4 and (now_ts - ts <= 600):
                                logger.info(f"invoke_agent: recovered final answer from checkpoints by word-overlap (ratio={ratio:.2f})")
                                return ans
                    except Exception:
                        pass
        except Exception:
            logger.exception("invoke_agent: error while trying to recover from checkpoints")

        return "L'agente non ha completato la sua esecuzione."
    except Exception as e:
        logger.exception("invoke_agent failed")
        return f"errore durante esecuzione agente: {e}"