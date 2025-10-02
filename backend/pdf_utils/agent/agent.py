import os
import json
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

def initialize_agent_tools_resources(graph_db: GraphDB, indexer: Indexer):
    global _global_graph_db, _global_indexer
    _global_graph_db = graph_db
    _global_indexer = indexer
    logger.debug("agent tools resources initialized")


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


@tool(description="neo4j_vector_search: run vector search on chunk index and return excerpts")
def neo4j_vector_search(query: str, k: int = DEFAULT_K) -> str:
    if _global_indexer is None or _global_graph_db is None:
        logger.error("neo4j_vector_search called before resources initialized")
        return "error: resources not initialized"
    logger.info(f"neo4j_vector_search: searching for: {query} (k={k})")
    try:
        emb = _global_indexer.generate_embeddings(query)
        results = _global_graph_db.query_vector_index("chunk_embeddings_index", emb, k=k)
        if not results:
            return "no_results"
        parts = []
        for i, r in enumerate(results, start=1):
            cid = r.get("chunk_id", "?")
            txt = r.get("node_content", "").replace("\n", " ")[:800]
            parts.append(f"[{i}] (chunk:{cid}) {txt}")
        return "\n".join(parts)
    except Exception as e:
        logger.exception("neo4j_vector_search failed")
        return f"error: {e}"


# --- workflow builder (simple, doc-like) ---

def _build_agent_workflow(planner_llm: Any = None, synth_llm: Any = None, mcp_tools: List[Any] = None, mcp_loaded_tools: List[Any] = None):
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

    # node: firewall - validate user query
    def node_firewall(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        logger.debug("node_firewall: running firewall_check")
        try:
            # prefer invoke to avoid deprecated __call__ behavior; fall back to direct call
            if hasattr(firewall_check, "invoke"):
                res_obj = firewall_check.invoke({"query": q})
                # handle coroutine
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
                res = firewall_check(q)
        except Exception as e:
            logger.exception("firewall_check error")
            res = "reject"
        if isinstance(res, str) and res.startswith("reject"):
            # immediate final answer
            final = AIMessage(content="your query was rejected by the firewall")
            return {"chat_history": state.get("chat_history", []) + [final]}
        # approved -> continue to presearch
        return {"chat_history": state.get("chat_history", []), "question": q, "approved": True}

    # node: presearch - mandatory neo4j_vector_search using the user query
    def node_presearch(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        logger.debug("node_presearch: performing mandatory neo4j_vector_search")
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
        # add tool output as ToolMessage
        try:
            out_str = getattr(res, "content", None) or str(res)
        except Exception:
            out_str = str(res)
        tm = ToolMessage(content=out_str, tool_call_id=f"presearch-{uuid.uuid4().hex[:8]}")
        logger.debug(f"node_presearch: appended ToolMessage with id={tm.tool_call_id} content_preview={out_str[:200].replace('\n',' ')}")
        return {"chat_history": state.get("chat_history", []) + [tm], "question": q}

    # node: planner - ask ollama whether more tools are needed
    def node_planner(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        # build a strict prompt: planner MUST output a single JSON object and nothing else
        prompt = (
            "you are a planner. OUTPUT A SINGLE VALID JSON OBJECT AND NOTHING ELSE.\n"
            "the JSON must have the key \"tool_calls\" whose value is a list of calls.\n"
            "each call is an object with fields: \"name\" (string) and \"args\" (object).\n"
            "if no tools are required, output exactly: {\"tool_calls\": []} and no additional text.\n"
            "do not output any explanatory text, commentary, or surrounding backticks — only the JSON object.\n"
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

        # try to parse single-line json if planner emitted it in text
        parsed_calls = None
        if not tool_calls:
            # look for json containing tool_calls
            try:
                import re
                m = re.search(r"(\{[\s\S]*\"tool_calls\"[\s\S]*\})", text)
                if m:
                    parsed = json.loads(m.group(1))
                    parsed_calls = parsed.get("tool_calls")
            except Exception:
                parsed_calls = None
        else:
            parsed_calls = tool_calls

        logger.info(f"node_planner: text={text[:400].replace('\n',' ')} parsed_calls={repr(parsed_calls)[:800]}")

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

            if is_json_like:
                logger.info("node_planner: planner returned JSON-only with no tool_calls; proceeding to synth")
                return {"chat_history": state.get("chat_history", []), "synth": True}

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

        filtered = []
        for item in normalized:
            nm = (item.get("name") or "").strip().lower()
            if nm in available_names:
                filtered.append(item)
            else:
                logger.warning(f"node_planner: planner requested unknown or disallowed tool '{item.get('name')}', ignoring it")

        if not filtered:
            # no valid tool calls -> proceed to synth (do not expose planner JSON to user)
            logger.info("node_planner: no valid tool_calls after filtering; proceeding to synth")
            return {"chat_history": state.get("chat_history", []), "synth": True}

        return {"chat_history": state.get("chat_history", []), "intermediate_steps": filtered}

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
        tm = ToolMessage(content=out_str, tool_call_id=tid)
        # append tool output and remove the executed intermediate step
        new_steps = steps[:-1]
        return {"chat_history": state.get("chat_history", []) + [tm], "intermediate_steps": new_steps}

    # node: synth - call together once with consolidated context
    def node_synth(state: Dict[str, Any]) -> Dict[str, Any]:
        # collect recent tool outputs and chat history, then call synth once
        chat = state.get("chat_history", [])
        # keep only last N messages to avoid token overload
        # increase history so synth sees more retrieved chunks (but still bounded)
        MAX_HISTORY = 20
        context = chat[-MAX_HISTORY:]
        # build prompt: system + context
        system = (
            "you are a helpful assistant. answer the user's question strictly using the provided context. "
            "cite chunk numbers where appropriate. if not present, say you cannot answer."
        )
        msgs = [HumanMessage(content=system)] + context
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
    wf.add_edge("firewall", "presearch")
    wf.add_edge("presearch", "planner")
    wf.add_conditional_edges(
        "planner",
        lambda s: "call_tool" if s.get("intermediate_steps") else "synth",
        {"call_tool": "call_tool", "synth": "synth"},
    )
    wf.add_edge("call_tool", "planner")

    app = wf.compile()
    logger.info("built langgraph agent (simple flow)")
    return app


async def invoke_agent(agent_app, user_message: str, chat_history: List[BaseMessage]) -> str:
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
    try:
        last_state = None
        accumulated: List[BaseMessage] = list(initial_state.get("chat_history", []))
        pre_existing_ai_count = sum(1 for m in accumulated if isinstance(m, AIMessage))

        def _extract_messages_from(obj) -> List[BaseMessage]:
            found: List[BaseMessage] = []
            try:
                if obj is None:
                    return found
                if isinstance(obj, BaseMessage):
                    found.append(obj)
                    return found
                if isinstance(obj, dict):
                    ch = obj.get("chat_history")
                    if ch:
                        for item in ch:
                            found.extend(_extract_messages_from(item))
                        return found
                    for v in obj.values():
                        found.extend(_extract_messages_from(v))
                    return found
                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        found.extend(_extract_messages_from(item))
                    return found
            except Exception:
                logger.exception("_extract_messages_from error")
            return found

        agen = agent_app.astream(initial_state)
        try:
            async for state in agen:
                last_state = state
                logger.debug(f"invoke_agent: got state keys={list(state.keys()) if isinstance(state, dict) else type(state)}")

                # explicit final_content short-circuit
                try:
                    if isinstance(state, dict) and state.get("final_content"):
                        final_response = state.get("final_content")
                        logger.info("invoke_agent: detected final_content in state, returning it")
                        break
                except Exception:
                    pass

                # accumulate any BaseMessage instances found in the emitted state
                try:
                    msgs = _extract_messages_from(state)
                    if msgs:
                        accumulated.extend(msgs)
                    # debug preview
                    preview = []
                    for m in accumulated[-5:]:
                        t = type(m).__name__
                        c = getattr(m, "content", None) or str(m)
                        preview.append(f"{t}:{c[:120].replace('\n',' ')}")
                    logger.debug(f"invoke_agent: accumulated chat_history preview (last {len(preview)}): {preview}")
                except Exception:
                    logger.exception("invoke_agent: error while building accumulated preview")

                # check whether new AIMessage(s) appeared during this run
                try:
                    total_ai_now = sum(1 for m in accumulated if isinstance(m, AIMessage))
                    if total_ai_now > pre_existing_ai_count:
                        for m in reversed(accumulated):
                            if isinstance(m, AIMessage):
                                final_response = getattr(m, "content", None) or str(m)
                                break
                        if final_response:
                            break
                except Exception:
                    logger.exception("invoke_agent: error while detecting AIMessage")
                    # fall back to scanning for last AIMessage
                    for m in reversed(accumulated):
                        if isinstance(m, AIMessage):
                            final_response = getattr(m, "content", None) or str(m)
                            break
                    if final_response:
                        break
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