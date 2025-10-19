import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_together import ChatTogether

from backend.pdf_utils.loader import get_layout_extractor
from backend.pdf_utils.preprocessor import split_sections_with_layout
from backend.pdf_utils.chunker import split_sections_into_chunks
from backend.pdf_utils.indexer import Indexer
from backend.pdf_utils.extractor import Extractor
from backend.pdf_utils.graph_db import GraphDB

from backend.pdf_utils.agent.agent import (
	_build_agent_workflow,
	initialize_agent_tools_resources,
	invoke_agent,
)


logger = logging.getLogger(__name__)


app = FastAPI(
	title="PDF Chat Agent API",
	description="API per interagire con l'agente di chat",
	version="2.0.0",
)


#memoria semplice per associare un documento caricato a un agente e alla sua chat history
class AgentState(BaseModel):
	filename: str
	thread_id: str


#runtime stores 
AGENTS: Dict[str, Any] = {}  # key: normalized filename -> agent app (LangGraph)
AGENT_GRAPH: Dict[str, GraphDB] = {}
AGENT_INDEXER: Dict[str, Indexer] = {}
CHAT_HISTORIES: Dict[str, List[BaseMessage]] = {}
THREAD_IDS: Dict[str, str] = {}
MEMORIES: Dict[str, InMemorySaver] = {}


def normalize_filename(filename: str) -> str:
	return (filename or "").replace("temp_", "").strip().lower()


class ChatRequest(BaseModel):
	user_message: str
	pdf_filename: str
	reset: Optional[bool] = False


class ChatResponse(BaseModel):
	agent_response: str


@app.get("/health")
async def health() -> Dict[str, str]:
	return {"status": "ok"}

#Carica un pdf, lo indicizza su neo4j e prepara un agente conversazionale per quel documento
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
	try:
		TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
		if not TOGETHER_API_KEY:
			raise HTTPException(status_code=500, detail="TOGETHER_API_KEY non è impostata nell'ambiente")

		#1) Lettura file in memoria
		file_bytes = await file.read()
		if not file_bytes:
			raise HTTPException(status_code=400, detail="File vuoto o non leggibile")

		norm_filename = normalize_filename(file.filename)

		#se ricarica lo stesso documento, chiudi e pulisci le risorse precedenti
		if norm_filename in AGENTS:
			try:
				prev_graph = AGENT_GRAPH.get(norm_filename)
				if prev_graph:
					prev_graph.close()
			except Exception:
				pass
			for m in (AGENTS, AGENT_GRAPH, AGENT_INDEXER, CHAT_HISTORIES, THREAD_IDS, MEMORIES):
				m.pop(norm_filename, None)

		#2) Estrazione layout e sezioni
		layout_extractor = get_layout_extractor()
		doc = layout_extractor(file_bytes)
		if doc is None:
			raise HTTPException(status_code=500, detail="Impossibile processare il PDF")
		sections = split_sections_with_layout(doc)

		#3) Chunking e indicizzazione Neo4j
		chunks = split_sections_into_chunks(sections)
		indexer = Indexer()
		graph_db_ingest = GraphDB()

		user_id = "default_user"
		graph_db_ingest.create_user(user_id)
		graph_db_ingest.create_document(norm_filename)
		graph_db_ingest.link_user_to_document(user_id, norm_filename)

		indexer.index_chunks_to_neo4j(norm_filename, chunks)

		#4) Topic ed entità via LLM + persistenza
		try:
			extractor = Extractor()
			extractor.attach_graph(graph_db_ingest, active_filename=norm_filename)
			full_text = getattr(doc, "text", "")
			if full_text:
				topics = extractor.extract_topics(full_text)
				for topic in topics:
					try:
						graph_db_ingest.add_topic_to_neo4j(topic, filename=norm_filename)
					except Exception:
						logger.exception("add_topic_to_neo4j failed")

				entities = extractor.extract_entities(full_text)
				for entity_type, names in entities.items():
					for name in names:
						try:
							graph_db_ingest.add_entity_to_neo4j(entity_type, name, filename=norm_filename)
						except Exception:
							logger.exception("add_entity_to_neo4j failed")
		except Exception:
			logger.exception("estrazione di topic/entità fallita; continuo senza")
		finally:
			try:
				graph_db_ingest.close()
			except Exception:
				pass

		#5) Costruzione agente LangGraph per il documento
		# Planner e synth con lo stesso modello Together (TEST, DEVO CAPIRE SE FUNZIONA)
		model_name = os.getenv("AGENT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
		planner = ChatTogether(
			model=model_name,
			temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
			together_api_key=TOGETHER_API_KEY,
			max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "1024")),
			timeout=int(os.getenv("AGENT_TIMEOUT", "60")),
		)
		synth = planner  # usa lo stesso modello per semplicità (DA TESTARE)

		memory = InMemorySaver()
		thread_id = uuid.uuid4().hex[:12]

		agent_app = _build_agent_workflow(
			planner_llm=planner,
			synth_llm=synth,
			checkpointer=memory,
		)

		# 6)Inizializza le risorse globali per i tools (vector search su Neo4j)
		agent_graph = GraphDB()
		initialize_agent_tools_resources(
			graph_db=agent_graph,
			indexer=indexer,
			active_filename=norm_filename,
			user_id=user_id,
		)

		# 7)Persistiamo lo stato in memoria per questo documento
		AGENTS[norm_filename] = agent_app
		AGENT_GRAPH[norm_filename] = agent_graph
		AGENT_INDEXER[norm_filename] = indexer
		CHAT_HISTORIES[norm_filename] = []
		THREAD_IDS[norm_filename] = thread_id
		MEMORIES[norm_filename] = memory

		return {"filename": norm_filename, "message": "PDF caricato e indicizzato con successo."}
	except HTTPException:
		raise
	except Exception as e:
		logger.exception("errore durante l'upload e l'indicizzazione del PDF")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdf_agent(request: ChatRequest) -> ChatResponse:
	norm_filename = normalize_filename(request.pdf_filename)
	agent_app = AGENTS.get(norm_filename)
	if not agent_app:
		raise HTTPException(status_code=404, detail="PDF non trovato o non indicizzato")

	#reset conversazione se richiesto
	if request.reset:
		CHAT_HISTORIES[norm_filename] = []

	chat_history = CHAT_HISTORIES.setdefault(norm_filename, [])

	#append del messaggio utente 
	chat_history.append(HumanMessage(content=request.user_message))

	#invoca agente
	try:
		cfg = {"configurable": {"thread_id": THREAD_IDS.get(norm_filename) or uuid.uuid4().hex[:12]}}
		response_text = await invoke_agent(agent_app, request.user_message, chat_history, config=cfg)
	except Exception as e:
		logger.exception("invoke_agent failure")
		raise HTTPException(status_code=500, detail=f"Errore agente: {e}")

	#add assistant message to history
	chat_history.append(AIMessage(content=response_text))

	# rimosso: logging Q/A diretto via GraphDB; ora demandato ai tool MCP nell'agente

	return ChatResponse(agent_response=response_text)


origins = [
	"http://localhost",
	"http://localhost:3001",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.on_event("shutdown")
async def on_shutdown():
	try:
		for name, g in list(AGENT_GRAPH.items()):
			try:
				g.close()
			except Exception:
				pass
			finally:
				AGENT_GRAPH.pop(name, None)
	except Exception:
		pass

