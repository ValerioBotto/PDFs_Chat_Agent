import streamlit as st
import os
import logging
import asyncio
import uuid
from typing import List, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
from langgraph.checkpoint.memory import InMemorySaver

from backend.pdf_utils.graph_db import GraphDB

 

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
            #parse oggetti JSON con tool_calls
            tool_calls = None
            try:
                # Search for JSON content first
                json_markers = [
                    (r"```(?:json)?\s*(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})\s*```", re.IGNORECASE),  
                    (r"(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})", 0),  
                    (r"(?:Tool call required|Using tool|Call tool):[\s\n]*(\{[\s\S]*?\"tool_calls\"[\s\S]*?\})", re.IGNORECASE),  
                ]
                
                for pattern, flags in json_markers:
                    m = re.search(pattern, text, flags)
                    if m:
                        json_part = m.group(1)
                        try:
                            parsed = json.loads(json_part)
                            if 'tool_calls' in parsed:
                                tool_calls = parsed['tool_calls']
                                if tool_calls:  
                                    break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                tool_calls = None

            # euristica: se il modello menziona neo4j_vector_search nel testo normale, converti in un tool_call
            if not tool_calls:
                try:
                    lower = text.lower()
                    if "neo4j_vector_search" in lower or "vector_search" in lower:
                        # prova a estrarre una query tra virgolette
                        qmatch = re.search(r"neo4j_vector_search\s*\(\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
                        if not qmatch:
                            qmatch = re.search(r"query\s*[:=]\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
                        if not qmatch:
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
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise


#funzione per inizializzare l'agente in un contesto asincrono
async def setup_agent_and_mcp(uploaded_file_data, llm_agent, indexer_instance, extractor_instance):
    if not TOGETHER_API_KEY:
        st.error("ERRORE: TOGETHER_API_KEY non impostata nel file .env. L'agente e l'estrattore non possono funzionare.")
        return False

    #assicurati che l'agente sia null prima di ripartire
    if st.session_state.agent_app is not None:
        st.session_state.agent_app = None
    

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

        #collega l'Extractor al grafo e carica sinonimi dinamici
        try:
            extractor_instance.attach_graph(graph_db, st.session_state.uploaded_filename)
        except Exception:
            logger.exception("impossibile collegare l'extractor al grafo per i sinonimi dinamici")

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
            checkpointer=memory,
        )

        #inizializza risorse globali dei tool, scoping all agent queries to the current uploaded document
        agent_graph_db = GraphDB()
        # passa anche lo user_id (stesso usato per i nodi in Neo4j)
        initialize_agent_tools_resources(
            agent_graph_db,
            indexer_instance,
            active_filename=st.session_state.uploaded_filename,
            user_id=user_id,
        )

        st.session_state.pdf_uploaded = True
        st.success("documento elaborato e agente inizializzato")
        return True

    except Exception as e:
        st.error(f"errore durante l'elaborazione o indicizzazione del pdf: {str(e)}")
        logger.exception("errore critico nel processo di upload/indicizzazione del pdf.", exc_info=True)

        st.session_state.pdf_uploaded = False
        st.session_state.agent_app = None

        

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

                try:
                    cfg = {"configurable": {"thread_id": thread}}
                    snapshot = None
                    try:
                        snapshot = st.session_state.agent_app.get_state(cfg)
                    except Exception:
                        try:
                            snapshot = st.session_state.agent_app.get_state()
                        except Exception:
                            snapshot = None
                    if snapshot is None:
                        st.info("agent_app.get_state non disponibile o non ha restituito snapshot.")
                    else:
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
        pass

    #gestione del caricamento del file e setup dell'agente
    if uploaded_file:
        if not st.session_state.pdf_uploaded or (
            st.session_state.uploaded_filename != uploaded_file.name
            if hasattr(st.session_state, "uploaded_filename")
            else True
        ):

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

    #bottone per pulire chat e resettare pdf
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
                        session_thread = st.session_state.get("thread_id")
                        if not session_thread:
                            session_thread = f"session-{uuid.uuid4().hex[:8]}"
                            st.session_state.thread_id = session_thread

                        # Use persistent session thread id as the LangGraph thread key
                        cfg = {
                            "configurable": {
                                "thread_id": session_thread
                            },
                            "recursion_limit": 50  # Increase from default 25 to handle complex workflows
                        }

                        # pass the current chat_history directly; let LangGraph + InMemorySaver
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