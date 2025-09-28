# backend/pdf_utils/agent/agent.py
import os
import logging
import asyncio
import random
import threading
import time
import re
from typing import List, Dict, Any, Literal, TypedDict 

# Importazioni per LangChain e LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool # Importiamo il decoratore tool
from langchain_together import ChatTogether
from langgraph.graph import StateGraph, END

# Importazioni per MCP - ATTENZIONE: questi non vengono usati direttamente QUII nel file agent.py
# Vengono usati in main.py per caricare i tool MCP.
# dal momento che i tool MCP verranno passati a _build_agent_workflow come lista.
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_mcp_adapters.tools import load_mcp_tools 

# Importazioni dei nostri moduli
from backend.pdf_utils.graph_db import GraphDB
from backend.pdf_utils.indexer import Indexer # Serve per generare gli embeddings nella ricerca vettoriale

logger = logging.getLogger(__name__)

#definizione dello stato del grafo langgraph
class AgentState(TypedDict):
    chat_history: List[BaseMessage]
    question: str
    intermediate_steps: List[Any]

#prompt di sistema per guidare l'llm
AGENT_SYSTEM_PROMPT = (
        "sei un assistente esperto nell'analisi di documenti pdf. "
        "il tuo compito è rispondere alle domande basandoti ESCLUSIVAMENTE sul contenuto del documento fornito. "
        "se la risposta non è presente nel documento, dichiara che non puoi aiutarmi con l'informazione. "
        "hai accesso ai seguenti strumenti: "
        "- neo4j_vector_search: usa questo strumento per cercare nel contenuto del pdf. è il tuo strumento principale per recuperare informazioni rilevanti. "
        "- read_neo4j_cypher: usa questo strumento per eseguire query cypher in neo4j. è utile per informazioni strutturate come topic/entità/relazioni. "
        "prioritizza 'neo4j_vector_search' per recuperare informazioni dettagliate. "
        "analizza attentamente il risultato dei tool; se le informazioni sono sufficienti formula la risposta finale. "
        "non richiamare lo stesso tool con gli stessi argomenti se hai già ricevuto un risultato. non inventare risposte."
)

# ---- DEFINIZIONE TOOL GLOBALE ----

# Placeholder per le istanze di GraphDB e Indexer
# Queste verranno impostate quando l'agente viene creato.
_global_neo4j_adapter: GraphDB = None
_global_indexer_instance: Indexer = None

@tool(description="Esegue una ricerca vettoriale su Neo4j e restituisce risultati formattati come testo.")
def neo4j_vector_search(query: str, k: int = 2) -> str:
    """Esegue una ricerca di similarità vettoriale nel knowledge graph Neo4j e
    restituisce i risultati formattati come testo.

    Args:
        query (str): La query testuale da trasformare in embedding e cercare.
        k (int): Numero di risultati da ritornare (default=2).

    Returns:
        str: Testo formattato con i risultati oppure un messaggio di errore.
    """
    # esegue una ricerca di similarità vettoriale nel knowledge graph neo4j
    if _global_indexer_instance is None or _global_neo4j_adapter is None:
        logger.error("neo4j_vector_search: risorse neo4j non inizializzate.")
        return "Errore: le risorse per la ricerca Neo4j non sono disponibili."

    logger.info(f"tool: eseguendo ricerca vettoriale neo4j per: '{query}'")
    try:
        query_embedding = _global_indexer_instance.generate_embeddings(query)
        results = _global_neo4j_adapter.query_vector_index(
            index_name="chunk_embeddings_index",
            query_embedding=query_embedding,
            k=k
        )

        if results:
            formatted_results = []
            for i, record in enumerate(results, start=1):
                chunk_content = record.get("node_content", "")
                score = record.get("score", 0.0)
                chunk_id = record.get("chunk_id", "unknown")
                section = record.get("section", "unspecified")
                excerpt = chunk_content.replace("\n", " ")[:500]
                formatted_results.append(f"[{i}] (chunk:{chunk_id} section:{section}) score:{score:.2f} -- {excerpt}...")
            guidance = (
                "\n\nuso per l'llm: usa questi estratti numerati per formulare una risposta completa. "
                "cita gli estratti usati indicando il numero tra parentesi quadre."
            )
            return "\n".join(formatted_results) + guidance
        else:
            return "Nessun risultato rilevante trovato nel PDF tramite ricerca vettoriale."

    except Exception as e:
        logger.error(f"errore durante la ricerca vettoriale neo4j per '{query}': {e}", exc_info=True)
        return f"Errore interno durante la ricerca vettoriale Neo4j: {str(e)}"

#nota: neo4j_graph_search non è più un tool locale; sarà fornito dal server mcp come read_neo4j_cypher

#funzione per inizializzare le risorse globali per i tool
def initialize_agent_tools_resources(neo4j_adapter: GraphDB, indexer_instance: Indexer):
    global _global_neo4j_adapter, _global_indexer_instance
    _global_neo4j_adapter = neo4j_adapter
    _global_indexer_instance = indexer_instance
    logger.debug("risorse globali per i tool dell'agente inizializzate.")

#lista per i tool mcp caricati
_global_mcp_tools: List[Any] = []

#funzione per definire i tools disponibili per l'agente
def get_tools():
    #aggiungi i tool locali poi quelli caricati dal server mcp
    local_tools = [neo4j_vector_search]
    return local_tools + _global_mcp_tools

#---- fine definizione tool globale ----

#funzione core per costruire il workflow langgraph
def _build_agent_workflow(llm: ChatTogether = None, planner_llm: Any = None, synth_llm: Any = None, mcp_loaded_tools: List[Any] = None):
    #se non viene passato planner/synth, usa la vecchia firma llm
    if planner_llm is None and synth_llm is None:
        planner_llm = synth_llm = llm
    else:
        planner_llm = planner_llm or llm
        synth_llm = synth_llm or llm

    initialize_agent_tools_resources(None, None)
    global _global_mcp_tools
    _global_mcp_tools = mcp_loaded_tools or []

    tools = get_tools()

    #bind degli strumenti al planner (se supporta bind_tools)
    try:
        planner_with_tools = planner_llm.bind_tools(tools)
    except Exception:
        planner_with_tools = planner_llm
    synth_llm_instance = synth_llm
    
    def call_model(state: AgentState) -> AgentState:
        #prepara i messaggi da inviare al planner
        messages = [HumanMessage(content=AGENT_SYSTEM_PROMPT)] + state["chat_history"]

        logger.debug("call_model: invoking planner LLM with messages (len=%d)" % len(messages))

        #retry esponenziale per rate limit / 503
        max_retries = 4
        backoff_base = 1.0
        response = None

        for attempt in range(1, max_retries + 1):
            try:
                response = planner_with_tools.invoke(messages)  # invoke del planner per decidere tool_call
                break
            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = ("429" in msg) or ("rate limit" in msg) or ("model_rate_limit" in msg)
                is_service_unavailable = ("503" in msg) or ("service unavailable" in msg)
                if is_rate_limit or is_service_unavailable:
                    wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.info(f"call_model: LLM rate-limited/service unavailable (attempt {attempt}/{max_retries}), retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.exception(f"call_model: errore durante invoke dell'LLM: {e}")
                    return {"chat_history": state["chat_history"] + [AIMessage(content=f"Errore LLM: {str(e)}")]}

        if response is None:
            # dopo i retry non abbiamo ottenuto risposta: segnaliamo il problema in modo chiaro
            logger.error("call_model: impossibile ottenere risposta dall'LLM dopo retry per rate limit/service unavailable.")
            rate_msg = (
                "Impossibile contattare il servizio LLM a causa di rate-limiting o indisponibilità. "
                "Riprova tra qualche minuto oppure usa la modalità fallback (ricerca diretta + generazione)."
            )
            return {"chat_history": state["chat_history"] + [AIMessage(content=rate_msg)]}

        accepted_tool_calls_format = []
        try:
            if getattr(response, "tool_calls", None):
                logger.debug(f"planner ha deciso di chiamare il tool: {response.tool_calls}")
                for tc in response.tool_calls:
                    if isinstance(tc, dict):
                        processed_tc = tc
                    else:
                        processed_tc = {"name": getattr(tc, "name", None), "args": getattr(tc, "args", None), "id": getattr(tc, "id", None)}
                    accepted_tool_calls_format.append(processed_tc)

                # loop detection: controlla ultimi messaggi di tool per evitare richiami ripetuti
                last_tool_call_request_msg = None
                last_tool_output_msg = None

                for msg_idx in range(len(state["chat_history"]) - 1, -1, -1):
                    msg = state["chat_history"][msg_idx]
                    if isinstance(msg, ToolMessage) and last_tool_output_msg is None:
                        last_tool_output_msg = msg
                    elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None) and last_tool_call_request_msg is None:
                        last_tool_call_request_msg = msg.tool_calls[0]
                    if last_tool_call_request_msg and last_tool_output_msg:
                        break

                if last_tool_call_request_msg and last_tool_output_msg and accepted_tool_calls_format:
                    current_tool_call_name = accepted_tool_calls_format[0].get('name')
                    current_tool_call_args = accepted_tool_calls_format[0].get('args')

                    prev_tool_call_name = last_tool_call_request_msg.get('name')
                    prev_tool_call_args = last_tool_call_request_msg.get('args')

                    if (current_tool_call_name == prev_tool_call_name and
                        current_tool_call_args == prev_tool_call_args and
                        ("Nessun risultato rilevante trovato" in getattr(last_tool_output_msg, 'content', '') or
                         "Nessuna informazione strutturata" in getattr(last_tool_output_msg, 'content', '') or
                         "Errore" in getattr(last_tool_output_msg, 'content', ''))):

                        logger.warning(f"Rilevato potenziale loop: LLM tentando di richiamare '{current_tool_call_name}' con stesse argomentazioni dopo un output non conclusivo.")

                        final_msg_content = (
                            f"Ho tentato di trovare informazioni usando lo strumento '{current_tool_call_name}' "
                            f"con la query '{current_tool_call_args}', ma non ho trovato nuove informazioni o "
                            f"l'operazione ha incontrato un problema: {last_tool_output_msg.content}. "
                            "Al momento non riesco a procedere oltre con queste informazioni."
                        )
                        return {"chat_history": state["chat_history"] + [AIMessage(content=final_msg_content)]}

                return {"chat_history": state["chat_history"] + [response], "intermediate_steps": accepted_tool_calls_format}
            else:
                # Se il planner non richiede tool, usiamo lo synth_llm per creare la risposta finale
                logger.debug("planner non ha richiesto tool -> usare synth llm per la risposta finale.")
                try:
                    # raccogli i messaggi tool recenti per logging e per capire quali chunk vengono passati
                    tool_msgs = [m for m in state["chat_history"] if isinstance(m, ToolMessage)]
                    tool_texts = [str(m.content) for m in tool_msgs]
                    logger.info(f"call_model: passing {len(tool_texts)} tool outputs to synth (first 1000 chars each).")
                    for i, t in enumerate(tool_texts, start=1):
                        logger.debug(f"synth tool #{i} (first 1000 chars): {t[:1000]}")

                    # prova a estrarre riferimenti a chunk dai tool outputs per debug
                    chunk_refs = []
                    for t in tool_texts:
                        try:
                            # matches tipo: [1] (chunk:abc123 section:...)
                            matches = re.findall(r"\[ *\d+ *\]\s*\(chunk:([^ )]+)", t)
                            chunk_refs.extend(matches)
                        except Exception:
                            continue
                    if chunk_refs:
                        logger.debug(f"call_model: parsed chunk references for synth: {chunk_refs}")

                    synth_messages = [HumanMessage(content="Genera una risposta completa, utile e passo-passo basata sulla seguente storia:")] + state["chat_history"]
                    synth_resp = synth_llm_instance.invoke(synth_messages)
                    final_content = getattr(synth_resp, "content", None) or str(synth_resp)
                    return {"chat_history": state["chat_history"] + [AIMessage(content=final_content)]}
                except Exception as e:
                    logger.exception(f"Errore durante synth_llm invoke: {e}")
                    return {"chat_history": state["chat_history"] + [AIMessage(content=f"Errore nella generazione della risposta finale: {e}")]}
        except Exception as e:
            logger.exception(f"call_model: errore processing response: {e}")
            return {"chat_history": state["chat_history"] + [AIMessage(content=f"Errore interno parsing response: {str(e)}")]}
        
        
    #definizione del nodo call_tool_node_function
    def call_tool_node_function(state: AgentState) -> AgentState:
        #nodo che esegue il tool selezionato dall'llm
        tool_call = state["intermediate_steps"][-1]
        
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args')
        tool_id = tool_call.get('id')
        
        if not tool_name:
            raise ValueError(f"nome del tool non trovato nel tool_call: {tool_call}")
            
        logger.info(f"esecuzione tool: {tool_name} con input: {tool_args}")

        try:
            selected_tool = next((t for t in get_tools() if getattr(t, 'name', None) == tool_name), None)
            if selected_tool is None:
                raise ValueError(f"tool '{tool_name}' non trovato tra i tool disponibili")

            #invoke may be sync or async; handle both
            tool_result = selected_tool.invoke(tool_args)
            #se è una coroutine, eseguila e attendi il risultato
            if hasattr(tool_result, '__await__'):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                try:
                    tool_output = loop.run_until_complete(tool_result)
                except Exception as e:
                    logger.exception(f"errore eseguendo tool async '{tool_name}': {e}")
                    tool_output = f"Errore durante l'esecuzione del tool '{tool_name}': {e}"
            else:
                tool_output = tool_result

            tool_message = ToolMessage(content=tool_output, tool_call_id=tool_id)
            return {"chat_history": state["chat_history"] + [tool_message]}

        except Exception as e:
            logger.exception(f"call_tool_node_function: errore eseguendo tool '{tool_name}': {e}")
            err_msg = f"Errore eseguendo tool '{tool_name}': {str(e)}"
            return {"chat_history": state["chat_history"] + [ToolMessage(content=err_msg, tool_call_id=tool_id)]}


    workflow = StateGraph(AgentState)

    workflow.add_node("call_model_node", call_model)
    workflow.add_node("call_tool_node", call_tool_node_function)
    workflow.set_entry_point("call_model_node")

    workflow.add_conditional_edges(
        "call_model_node",
        lambda state: "call_tool_node" if state.get("intermediate_steps") else END,
        {"call_tool_node": "call_tool_node", END: END}
    )
    workflow.add_edge("call_tool_node", "call_model_node")

    app = workflow.compile()

    #nota: rimozione della generazione dell'immagine del grafo per semplificare
    logger.info("langgraph agent workflow built and compiled.")
    return app


async def invoke_agent(agent_app, user_message: str, chat_history: List[BaseMessage]) -> str:
    #questa funzione invoca l'agente e raccoglie la risposta finale
    initial_state = {"chat_history": chat_history,
                     "question": user_message, 
                     "intermediate_steps": []}
    
    final_response = ""
    try:
        async for state in agent_app.astream(initial_state):
            if isinstance(state, dict) and "__end__" in state:
                final_ai_message = None
                for message in state["__end__"]["chat_history"][::-1]:
                    if isinstance(message, AIMessage) and not message.tool_calls: 
                        final_ai_message = message
                        break
            
                if final_ai_message:
                    final_response = final_ai_message.content
                else: 
                    final_response = "L'agente ha eseguito dei passi o chiamato un tool, ma non ha generato una risposta finale. Potrebbe aver bisogno di più contesto o di un'altra iterazione."
                break
                
        else:
            final_response = "L'agente non ha completato la sua esecuzione. Potrebbe esserci un errore interno non catturato."

    except Exception as e:
        logger.error(f"errore durante l'invocazione di agent_app.astream: {str(e)}", exc_info=True)
        if "429 Too Many Requests" in str(e):
            return "Si è verificato un errore di rate-limiting con Together.ai. Per favore, attendi qualche minuto e riprova."
        else:
            return f"Si è verificato un errore inaspettato durante l'esecuzione dell'agente: {str(e)}"
            
    return final_response