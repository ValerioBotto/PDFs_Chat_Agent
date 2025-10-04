import streamlit as st
import os
import logging
import asyncio
import uuid
from typing import List, Any

class OllamaWrapper:
    def __init__(self, model_name: str = "llama3.2:1b", timeout: int = 90):
        # Using a lightweight model by default
        self.model = model_name
        self.timeout = timeout
        self._bound_tools = None
        logger.info(f"OllamaWrapper initialized with model={model_name}, timeout={timeout}s")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_together import ChatTogether

from backend.pdf_utils.loader import get_layout_extractor
from backend.pdf_utils.preprocessor import split_sections_with_layout
from backend.pdf_utils.chunker import split_sections_into_chunks
from backend.pdf_utils.indexer import Indexer
from backend.pdf_utils.extractor import Extractor

from backend.pdf_utils.agent.agent import (
    _build_agent_workflow,
    initialize_agent_tools_resources,
    invoke_agent,
)
# langgraph in-memory checkpointer for tutorial-style memory
from langgraph.checkpoint.memory import InMemorySaver

from backend.pdf_utils.graph_db import GraphDB

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

#carica le variabili d'ambiente
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

#configurazione streamlit
st.set_page_config(page_title="chat con pdf", layout="wide")

#inizializzazione stato sessione
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_app" not in st.session_state:
    st.session_state.agent_app = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "layout_extractor" not in st.session_state:
    st.session_state.layout_extractor = get_layout_extractor()
if "indexer" not in st.session_state:
    st.session_state.indexer = Indexer()
if "extractor" not in st.session_state:
    st.session_state.extractor = Extractor()
if "llm_agent" not in st.session_state:
    st.session_state.llm_agent = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=1.0,
        together_api_key=TOGETHER_API_KEY,
        max_tokens=1024,
        max_retries=3,
        timeout=60,
    )

#variabili mcp nello stato
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = None
if "mcp_client_context" not in st.session_state:
    st.session_state.mcp_client_context = None
if "mcp_session" not in st.session_state:
    st.session_state.mcp_session = None
if "mcp_session_context" not in st.session_state:
    st.session_state.mcp_session_context = None


class OllamaWrapper:
    def __init__(self, model_name: str = "mistral", timeout: int = 90):
        self.model = model_name
        self.timeout = timeout
        self._bound_tools = None
        logger.info(f"OllamaWrapper initialized with model={model_name}, timeout={timeout}s")

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            role = getattr(m, "type", None) or m.__class__.__name__
            text = getattr(m, "content", str(m))
            parts.append(f"[{role}]\n{text}\n")
        return "\n".join(parts)

    def invoke(self, messages: List[BaseMessage]):
        prompt = self._messages_to_prompt(messages)
        # if tools are bound, prepend a tools description and an instruction to output JSON for tool calls
        tool_instructions = ""
        if self._bound_tools:
            descr_lines = [
                '''You have access to the following tools. If you decide a tool should be called, output a single-line JSON object with the key "tool_calls" containing a list of calls. Example:
{"tool_calls": [{"name": "neo4j_vector_search", "args": {"query": "...", "k": 2}}]}
Otherwise, output a normal textual answer without that JSON line.''',
            ]
            for t in self._bound_tools:
                tname = getattr(t, 'name', None) or getattr(t, '__name__', str(t))
                tdesc = getattr(t, 'description', None) or (getattr(t, '__doc__', '') or '').splitlines()[0]
                descr_lines.append(f"- {tname}: {tdesc}")
            tool_instructions = "\n".join(descr_lines) + "\n\n"

        full_prompt = tool_instructions + prompt

        # use ollama run with positional prompt argument to avoid unsupported flags
        import subprocess, json, re
        args = ["ollama", "run", self.model, full_prompt]
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
            text = proc.stdout.strip()
            # try to parse a JSON object with tool_calls
            tool_calls = None
            try:
                # Search for JSON content first
                json_markers = [
                    (r"```(?:json)?\s*(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})\s*```", re.IGNORECASE),  # code fence
                    (r"(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})", 0),  # raw JSON
                    (r"(?:Tool call required|Using tool|Call tool):[\s\n]*(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})", re.IGNORECASE),  # with prefix
                ]
                
                for pattern, flags in json_markers:
                    m = re.search(pattern, text, flags)
                    if m:
                        json_part = m.group(1)
                        try:
                            parsed = json.loads(json_part)
                            if 'tool_calls' in parsed:
                                tool_calls = parsed['tool_calls']
                                if tool_calls:  # if non-empty list
                                    break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                tool_calls = None

            # heuristic: if model mentions neo4j_vector_search in plain text, convert to a tool_call
            if not tool_calls:
                try:
                    lower = text.lower()
                    if "neo4j_vector_search" in lower or "vector_search" in lower:
                        # try to extract a quoted query
                        qmatch = re.search(r"neo4j_vector_search\s*\(\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
                        if not qmatch:
                            qmatch = re.search(r"query\s*[:=]\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
                        if not qmatch:
                            # fallback: use last line or whole prompt as query
                            last_line = text.strip().splitlines()[-1]
                            q = last_line.strip()[:200]
                        else:
                            q = qmatch.group(1)
                        if q:
                            tool_calls = [{"name": "neo4j_vector_search", "args": {"query": q, "k": 2}}]
                            logger.debug(f"OllamaWrapper heuristic created tool_call for neo4j_vector_search with query: {q}")
                except Exception:
                    tool_calls = None

            class Resp:
                pass

            r = Resp()
            r.content = text
            if tool_calls:
                r.tool_calls = tool_calls
            return r
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or e.output or ""
            raise RuntimeError(f"ollama invoke failed: {stderr}")
        except subprocess.TimeoutExpired as e:
            # primo timeout: proviamo un retry con timeout raddoppiato (fino a 120s)
            logger.warning(f"ollama invoke timed out after {self.timeout}s; retrying with longer timeout.")
            try:
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    check=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=min(self.timeout * 2, 120),
                )
                text = proc.stdout.strip()
                class Resp:
                    pass

                r = Resp()
                r.content = text
                return r
            except subprocess.CalledProcessError as e2:
                stderr = e2.stderr or e2.output or ""
                raise RuntimeError(f"ollama invoke failed on retry: {stderr}")
            except subprocess.TimeoutExpired:
                raise RuntimeError("ollama invoke timeout")

    def bind_tools(self, tools: List[Any]):
        # store tools metadata so invoke can include descriptions
        self._bound_tools = tools
        return self


#funzione: ottieni o crea un event loop
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise


#fallback: embedding della domanda, ricerca vettoriale su neo4j e chiamata diretta all'llm
def direct_vector_qa(user_question: str, indexer_instance: Indexer, llm_agent: ChatTogether, top_k: int = 5) -> str:
    try:
        #1) genera embedding della domanda
        query_embedding = indexer_instance.generate_embeddings(user_question)

        #2) ricerca su neo4j
        graph_db = GraphDB()
        # If an uploaded filename is present in session, scope the search to avoid cross-document leakage
        try:
            active_filename = getattr(st.session_state, "uploaded_filename", None)
        except Exception:
            active_filename = None
        try:
            results = graph_db.query_vector_index("chunk_embeddings_index", query_embedding, k=top_k, filename=active_filename)
        except TypeError:
            results = graph_db.query_vector_index("chunk_embeddings_index", query_embedding, k=top_k)

        if not results:
            graph_db.close()
            return "Nessun chunk rilevante trovato nel documento tramite ricerca vettoriale."

        #log diagnostico
        logger.info(f"direct_vector_qa: retrieved {len(results)} results from Neo4j.")
        for i, r in enumerate(results, start=1):
            logger.debug(f"result #{i}: keys={list(r.keys())}")
            if "score" in r:
                logger.debug(f" result #{i} score: {r.get('score')}")

        #3) prepara estratti
        excerpts = []
        for i, r in enumerate(results, start=1):
            content = r.get("node_content", "").strip()
            excerpt = content if len(content) <= 1000 else content[:1000] + " ...[troncato]"
            excerpts.append(f"[{i}] {excerpt}")

        joined_context = "\n\n".join(excerpts)

        #prompt strutturato
        system_msg = (
            "Sei un Assistente alla consultazione di PDF. Rispondi alla domanda dell'utente in modo chiaro e conciso utilizzando il contenuto degli estratti. "
            "Nella tua risposta cita gli estratti utilizzati con il relativo numero tra parentesi quadre. "
            "Se l'informazione non è presente nei chunk, rispondi che non è disponibile e non inventare."
        )

        prompt = f"{system_msg}\n\nContesto (estratti dal documento):\n{joined_context}\n\nDomanda: {user_question}"

        #4) chiama l'llm direttamente
        messages = [HumanMessage(content=prompt)]
        response = llm_agent.invoke(messages)
        resp_text = getattr(response, "content", None) or str(response)

        graph_db.close()
        return resp_text

    except Exception as e:
        logger.exception(f"direct_vector_qa fallback error: {e}")
        try:
            graph_db.close()
        except Exception:
            pass
        return f"Errore durante ricerca diretta: {str(e)}"


#funzione per inizializzare l'agente e mcp in un contesto asincrono
async def setup_agent_and_mcp(uploaded_file_data, llm_agent, indexer_instance, extractor_instance):
    if not TOGETHER_API_KEY:
        st.error("ERRORE: TOGETHER_API_KEY non impostata nel file .env. L'agente e l'estrattore non possono funzionare.")
        return False

    #assicurati che l'agente e le sessioni mcp siano null prima di ripartire
    if st.session_state.agent_app is not None:
        st.session_state.agent_app = None

    #chiudi eventuali context manager mcp aperti
    if st.session_state.mcp_session_context:
        try:
            await st.session_state.mcp_session_context.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Errore durante la chiusura di mcp_session_context: {e}")
        st.session_state.mcp_session_context = None
        st.session_state.mcp_session = None

    if st.session_state.mcp_client_context:
        try:
            await st.session_state.mcp_client_context.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Errore durante la chiusura di mcp_client_context: {e}")
        st.session_state.mcp_client_context = None
        st.session_state.mcp_client = None

    #reset stato per nuovo caricamento
    st.session_state.pdf_uploaded = False
    st.session_state.chat_history = []
    st.session_state.processed_chunks = []
    st.session_state.uploaded_filename = uploaded_file_data.name

    file_bytes = uploaded_file_data.getvalue()
    logger.debug(f"pdf file size: {len(file_bytes)} bytes")

    graph_db = None
    try:
        #1) processing del pdf
        doc = st.session_state.layout_extractor(file_bytes)
        if doc is None:
            raise ValueError("errore nell'estrazione del documento dal pdf.")

        sections = split_sections_with_layout(doc)
        st.session_state.processed_sections = sections

        chunks = split_sections_into_chunks(sections)
        st.session_state.processed_chunks = chunks
        logger.info(f"total of {len(chunks)} chunks generated from all sections.")

        #2) interazione con neo4j e indicizzazione
        graph_db = GraphDB()
        user_id = "default_user"
        graph_db.create_user(user_id)
        graph_db.create_document(st.session_state.uploaded_filename)
        graph_db.link_user_to_document(user_id, st.session_state.uploaded_filename)

        indexer_instance.index_chunks_to_neo4j(st.session_state.uploaded_filename, chunks)

        full_text = doc.text
        st.info("estrazione di topic ed entità dal documento...")
        topics = extractor_instance.extract_topics(full_text)
        for topic in topics:
            graph_db.add_topic_to_neo4j(topic, st.session_state.uploaded_filename)

        entities = extractor_instance.extract_entities(full_text)
        for entity_type, entity_names in entities.items():
            for entity_name in entity_names:
                graph_db.add_entity_to_neo4j(entity_type, entity_name, st.session_state.uploaded_filename)

        #dopo l'ingestion chiudiamo la connessione locale
        graph_db.close()
        graph_db = None

        #3) avvio mcp server e creazione agente
        neo4j_mcp_server_params = StdioServerParameters(
            command="uvx",
            args=["mcp-neo4j-cypher@0.3.0", "--transport", "stdio"],
            env={
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
                "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "neo4j"),
            },
        )

        st.session_state.mcp_client = stdio_client(neo4j_mcp_server_params)
        st.session_state.mcp_client_context = await st.session_state.mcp_client.__aenter__()

        read_stream, write_stream = st.session_state.mcp_client_context
        st.session_state.mcp_session = ClientSession(read_stream, write_stream)
        st.session_state.mcp_session_context = await st.session_state.mcp_session.__aenter__()

        await st.session_state.mcp_session.initialize()
        loaded_mcp_tools = await load_mcp_tools(st.session_state.mcp_session)
        logger.info(f"tool MCP caricati: {[t.name for t in loaded_mcp_tools]}")

        filtered_mcp_tools = [t for t in loaded_mcp_tools if t.name in ["read_neo4j_cypher", "get_neo4j_schema"]]

        # costruisci agente langgraph con planner locale (Ollama) per planning e Together per la synth finale
        planner = OllamaWrapper(model_name=os.getenv("OLLAMA_MODEL", "llama3.2:1b"), timeout=int(os.getenv("OLLAMA_TIMEOUT", "90")))
        synth = st.session_state.llm_agent

        # create an in-memory saver to enable LangGraph checkpointer-based memory
        memory = InMemorySaver()
        # store memory in session so it persists for this Streamlit session
        st.session_state.agent_memory = memory

        # generate or reuse a thread_id for this session (used by the checkpointer)
        if "thread_id" not in st.session_state:
            # use filename + timestamp to scope memory per uploaded document/session
            st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"

        st.session_state.agent_app = _build_agent_workflow(
            planner_llm=planner,
            synth_llm=synth,
            mcp_loaded_tools=filtered_mcp_tools,
            checkpointer=memory,
        )

        #inizializza risorse globali dei tool, scoping all agent queries to the current uploaded document
        agent_graph_db = GraphDB()
        initialize_agent_tools_resources(agent_graph_db, indexer_instance, active_filename=st.session_state.uploaded_filename)

        st.session_state.pdf_uploaded = True
        st.success("documento elaborato e agente inizializzato")
        return True

    except Exception as e:
        st.error(f"errore durante l'elaborazione o indicizzazione del pdf: {str(e)}")
        logger.exception("errore critico nel processo di upload/indicizzazione del pdf.", exc_info=True)

        #pulizia robusta in caso di errore
        st.session_state.pdf_uploaded = False
        st.session_state.agent_app = None

        #chiusura context manager asincroni
        if st.session_state.mcp_session_context:
            try:
                await st.session_state.mcp_session_context.__aexit__(None, None, None)
            except Exception as e_close:
                logger.warning(f"Errore durante la chiusura di mcp_session_context: {e_close}")
            st.session_state.mcp_session = None

        if st.session_state.mcp_client_context:
            try:
                await st.session_state.mcp_client_context.__aenter__(None, None, None)
            except Exception as e_close:
                logger.warning(f"Errore durante la chiusura di mcp_client_context: {e_close}")
            st.session_state.mcp_client = None

        initialize_agent_tools_resources(None, None)
        return False


#sidebar streamlit
with st.sidebar:
    st.header("📁 carica documento")
    uploaded_file = st.file_uploader("seleziona un pdf", type=["pdf"])

    # Debug: show InMemorySaver checkpoints and agent state for current session
    try:
        if st.session_state.get("agent_app") is not None and st.session_state.get("agent_memory") is not None:
            if st.button("🔎 mostra checkpoint memoria (debug)"):
                mem = st.session_state.get("agent_memory")
                thread = st.session_state.get("thread_id")
                st.write(f"thread_id: {thread}")
                try:
                    cps = list(mem.list({"configurable": {"thread_id": thread}}))
                    if not cps:
                        st.info("Nessun checkpoint trovato per questo thread nella InMemorySaver.")
                    else:
                        for ct in cps:
                            try:
                                ck = getattr(ct, 'checkpoint', ct)
                                st.write({
                                    "id": ck.get("id"),
                                    "channel_values": ck.get("channel_values"),
                                    "channel_versions": ck.get("channel_versions"),
                                })
                            except Exception:
                                st.write(repr(ct))
                except Exception as e:
                    st.error(f"errore leggendo checkpoint da InMemorySaver: {e}")

                # also attempt to fetch graph's state snapshot if supported
                try:
                    cfg = {"configurable": {"thread_id": thread}}
                    snapshot = None
                    try:
                        snapshot = st.session_state.agent_app.get_state(cfg)
                    except Exception:
                        # some compiled apps expose get_state without requiring cfg
                        try:
                            snapshot = st.session_state.agent_app.get_state()
                        except Exception:
                            snapshot = None
                    if snapshot is None:
                        st.info("agent_app.get_state non disponibile o non ha restituito snapshot.")
                    else:
                        # display a compact view
                        try:
                            st.write({
                                "values_keys": list(snapshot.values.keys()) if hasattr(snapshot, 'values') and isinstance(snapshot.values, dict) else str(type(snapshot))
                            })
                            st.write(snapshot)
                        except Exception:
                            st.write(repr(snapshot))
                except Exception as e:
                    st.error(f"errore ottenendo snapshot da agent_app: {e}")
    except Exception:
        # do not block normal flow if debug widget fails
        pass

    # gestione del caricamento del file e setup dell'agente
    if uploaded_file:
        if not st.session_state.pdf_uploaded or (
            st.session_state.uploaded_filename != uploaded_file.name
            if hasattr(st.session_state, "uploaded_filename")
            else True
        ) or st.session_state.mcp_client is None or st.session_state.mcp_session is None:

            with st.spinner("preparazione ambiente e agente..."):
                loop = get_or_create_event_loop()
                success = loop.run_until_complete(
                    setup_agent_and_mcp(
                        uploaded_file,
                        st.session_state.llm_agent,
                        st.session_state.indexer,
                        st.session_state.extractor,
                    )
                )
            if success:
                st.rerun()

    # bottone per pulire chat e resettare pdf
    if st.session_state.pdf_uploaded:
        if st.button("🗑️ pulisci chat e resetta pdf"):
            st.session_state.chat_history = []
            st.session_state.agent_app = None
            st.session_state.pdf_uploaded = False
            if hasattr(st.session_state, "uploaded_filename"):
                del st.session_state.uploaded_filename
            if hasattr(st.session_state, "processed_sections"):
                del st.session_state.processed_sections
            if hasattr(st.session_state, "processed_chunks"):
                del st.session_state.processed_chunks

            #chiudi le sessioni mcp
            if st.session_state.mcp_session_context:
                asyncio.run(st.session_state.mcp_session_context.__aexit__(None, None, None))
                st.session_state.mcp_session_context = None
                st.session_state.mcp_session = None

            if st.session_state.mcp_client_context:
                asyncio.run(st.session_state.mcp_client_context.__aexit__(None, None, None))
                st.session_state.mcp_client_context = None
                st.session_state.mcp_client = None

            initialize_agent_tools_resources(None, None)
            st.info("chat e stato pdf resettati.")
            st.rerun()


#area centrale: chat con l'agente
if st.session_state.pdf_uploaded and st.session_state.agent_app:
    st.title("💬 chatta con il tuo pdf")

    #render della cronologia: stocchiamo oggetti BaseMessage nella chat_history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content if isinstance(msg, BaseMessage) else str(msg))

    user_message = st.chat_input("fai una domanda sul documento...")

    if user_message:
        st.session_state.chat_history.append(HumanMessage(content=user_message))
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("l'agente sta elaborando la risposta..."):
                try:
                    loop = get_or_create_event_loop()
                    try:
                        # Use the persistent session thread id so the compiled graph's
                        # InMemorySaver can restore prior conversation state and memory.
                        # This enables multi-turn memory as in the LangGraph tutorial.
                        session_thread = st.session_state.get("thread_id")
                        if not session_thread:
                            # fallback: ensure a thread id exists
                            session_thread = f"session-{uuid.uuid4().hex[:8]}"
                            st.session_state.thread_id = session_thread

                        # Use persistent session thread id as the LangGraph thread key
                        cfg = {"configurable": {"thread_id": session_thread}}

                        # pass the current chat_history directly; let LangGraph + InMemorySaver
                        # restore and manage conversation state per the official tutorial
                        response_content = loop.run_until_complete(
                            asyncio.wait_for(
                                invoke_agent(
                                    st.session_state.agent_app, user_message, st.session_state.chat_history, cfg
                                ),
                                timeout=180,
                            )
                        )
                    except asyncio.TimeoutError:
                        logger.warning("invoke_agent timed out.")
                        response_content = "Errore: l'agente ha impiegato troppo tempo per rispondere. Riprova più tardi."
                    # original behavior: no checkpoint recovery, let invoke_agent results surface

                    logger.debug(f"invoke_agent returned: {response_content}")
                    if not response_content:
                        st.warning("l'agente non ha restituito una risposta. controlla i log.")
                        response_content = "mi dispiace, non sono riuscito a generare una risposta. prova a riformulare la domanda o riprova più tardi."
                    elif response_content.startswith("Errore") or "rate-limiting" in response_content.lower():
                        st.error(response_content)

                    st.markdown(response_content)
                    st.session_state.chat_history.append(AIMessage(content=response_content))
                except Exception as e:
                    st.error(f"errore durante l'interazione con l'agente: {e}")
                    logger.exception("errore nell'invoke dell'agente.")

else:
    st.info("👈 carica un pdf dalla sidebar per iniziare la conversazione")