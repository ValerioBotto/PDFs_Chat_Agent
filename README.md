# PDFs_Chat_Agent

## Project Overview

This project introduces an **Interactive AI agent for Technical/medical/legislative PDF(s) analysis**. The goal is to simplify how users interact with complex, text-heavy documents by allowing them to upload PDFs, automatically extract key topics, and engage in contextual, personalized conversations with an AI agent. The system leverages advanced techniques including **LangChain/LangGraph**, a **Neo4j knowledge graph**, and a **vector database** for efficient semantic retrieval, offering a dynamic and significantly enhanced document consultation experience.

**Special Emphasis on Complex PDF Challenges:**
This project was developed with a strong focus on simplifying the tedious and often frustrating process of reading difficult technical PDFs. These documents frequently present challenges such as unusual layouts, multi-column text, embedded tabular information, or images with detailed textual descriptions of components. Our system is specifically designed to overcome these hurdles, transforming these data-rich, yet hard-to-parse, documents into an easily navigable and interactive knowledge source. Further sections will detail how we precisely extract this information.

**Academic Context:**
This project was developed as part of a curricular internship at **Logogramma s.r.l.**, undertaken for the LM-43 Master's Degree program at the **University of Catania**.

## Detailed Project Description

The PDFs_Chat_Agent is a comprehensive system designed to provide an intelligent interface for PDF documents. Its architecture is composed of several integrated modules:

*   **PDF Upload and Processing**: Users can upload PDF documents through a user-friendly Streamlit frontend. The system then processes these documents to extract text and structural information.
*   **Topic Extraction and Knowledge Graph Enrichment**: Key topics and entities are automatically identified from the PDF content using a Large Language Model (LLM). This information, along with document structure and user preferences, is used to dynamically enrich a **Neo4j knowledge graph**, forming a structured and semantic representation of the document's knowledge.
*   **Conversational AI Agent**: An intelligent agent, built using the LangChain and LangGraph frameworks, enables interactive querying of the PDF content. The agent maintains conversational memory, ensuring contextual and personalized responses.
*   **Semantic Retrieval**: For accurate and relevant responses, the agent uses a combination of semantic search techniques, including vector search over document chunks and direct queries to the Neo4j knowledge graph.
*   **Interactive Frontend (Streamlit)**: A Streamlit-based interface provides the chat functionality, allowing users to converse with the AI agent and receive information drawn directly from their uploaded PDFs.

## PDF Preprocessing

Extracting usable text from complex PDFs is a critical first step. We employ
`spaCyLayout` (or similar layout-aware tools) within `loader.py` to process PDFs. This allows us to not only extract the full textual content but also to understand the document's layout, including multi-column sections, headers, and footers. This intelligent extraction is crucial for accurately segmenting the document into meaningful sections and chunks, overcoming the challenges posed by diverse and often non-standard technical document formats. Once the text is extracted, `preprocessor.py` and `chunker.py` ensure it is properly segmented for further indexing and analysis.

## The Agent System

The core intelligence of this project resides in its agent system, orchestrated using **LangChain** and **LangGraph**.

*   **Framework**: The agents are deployed using **LangChain** for handling interactions with Large Language Models and **LangGraph** for defining complex, stateful agentic workflows. LangGraph enables the creation of cyclical processes where the agent can perform actions, observe results, and decide on the next step.
*   **Agent Roles**: The primary conversational agent is responsible for understanding user queries, retrieving relevant information, and generating concise, accurate responses. Its main role is to act as an intelligent document consultant.
*   **Conversational Memory**: The agent maintains a persistent conversational memory, allowing it to remember past turns in the dialogue. This is critical for providing coherent, context-aware, and personalized answers. This memory is managed by retaining previous messages (`chat_history`) and using a checkpointer system [4, 1].
*   **Tools and Information Retrieval**: To enhance its capabilities, the agent is equipped with several tools:
    *   `neo4j_vector_search`: This tool performs a vector similarity search on the indexed chunks of the uploaded PDF, allowing the agent to retrieve highly relevant passages based on the user's query [4].
    *   `neo4j_graph_search`: This tool allows the agent to directly query the structured knowledge in the Neo4j graph, enabling it to answer questions relying on extracted entities and relationships.
    *   `web_search`: In some advanced configurations (not explicit in the current context but common), agents can also use web search tools to fetch external information.
    The agent dynamically decides which tool to use based on the user's question, ensuring it always has the most pertinent information at hand.
*   **Information Flow**: When a user asks a question, the agent processes it, potentially using the retriever tools to fetch relevant document chunks or graph data. This information is then passed back to the LLM within the agent, which synthesizes a coherent answer, grounding its response in the retrieved facts to mitigate "hallucinations."

## Neo4j Knowledge Graph

The **Neo4j knowledge graph** serves as the project's semantic memory and plays a pivotal role in enabling deep exploration and intelligent retrieval.

*   **Structured Knowledge**: It stores nodes representing users, documents, sections, and extracted topics. Relationships are established between these nodes, such as "HAS_PREFERENCE" (user-topic), "TALKS_ABOUT" (document-topic), "HAS_SECTION" (document-section), and "RELATED_TO" (topic-topic) [1].
*   **Dynamic Enrichment**: The graph is dynamically enriched during PDF parsing and user interaction. New topics, entities, and relationships are added as documents are processed and as the agent learns from user feedback.
*   **Vector Indexing and Search**: For enhanced retrieval capabilities, a vector index is created within Neo4j. This allows for efficient vector similarity searches directly on the graph data, complementing the standard graph traversal queries. Embeddings generated from document chunks and topics enable semantic searches, drastically improving the relevance of retrieved information by capturing conceptual meanings rather than just keywords.

## Setup Instructions

To get the PDFs_Chat_Agent running locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/PDFs_Chat_Agent.git
    cd PDFs_Chat_Agent
    ```

2.  **Install Dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Neo4j Setup**:
    *   Download and install Neo4j Desktop or run Neo4j in a Docker container.
    *   Create a new Neo4j database instance.
    *   Note down your Neo4j URI (`bolt://localhost:7687`), username, and password.

4.  **Ollama Setup (for Local Planner)**:
    *   **Install Ollama**: Download and install Ollama from [ollama.ai](https://ollama.ai/). Ensure it's running in the background.
    *   **Download the local model**: Pull the `llama3.2:1b` model (or `llama3:8b` for better performance if your system allows) to be used by the planner:
        ```bash
        ollama pull llama3.2:1b
        # or for a potentially better planner
        # ollama pull llama3:8b
        ```

5.  **API Keys Configuration**:
    You will need an API key for the external Large Language Model.
    *   **Together.ai**: Sign up on Together.ai to get your API key.
    *   Create a `.env` file in the root directory of the project and add your API key, along with the Neo4j and Ollama configurations:
        ```
        TOGETHER_API_KEY="YOUR_TOGETHER_API_KEY_HERE"
        # Neo4j configuration
        NEO4J_URI="bolt://localhost:7687"
        NEO4J_USERNAME="neo4j"
        NEO4J_PASSWORD="your_neo4j_password"
        # Ollama configuration for the local planner
        OLLAMA_MODEL="llama3.2:1b" # Ensure this model is pulled in Ollama
        OLLAMA_BASE_URL="http://localhost:11434" # Default Ollama API endpoint
        OLLAMA_TIMEOUT="90" # Timeout in seconds for Ollama calls
        # External LLM for synthesis
        LLM_SYNTHESIS_MODEL="ServiceNow-AI/Apriel-1.5-15b-Thinker"
        ```
    *   **Note**: The agent's final response generation (synthesis) is configured to use `ServiceNow-AI/Apriel-1.5-15b-Thinker` from Together.ai.

5.  **Run the Streamlit Application**:
    ```bash
    streamlit run main.py
    ```
    This will open the application in your web browser. You can then upload PDFs and start interacting with the AI agent.

## Pictures


<img width="1429" height="1379" alt="Screenshot 2025-10-29 125934" src="https://github.com/user-attachments/assets/7921adea-f63e-4b5b-aaba-7f96d9aa4e7a" />

<img width="1426" height="1705" alt="Screenshot 2025-10-20 162014" src="https://github.com/user-attachments/assets/0d72c2cd-173a-4cb1-a4f2-c76f17bd1b3c" />



## Credits

Developed by Valerio Botto as part of a master's degree internship at Logogramma s.r.l.

Connect with me on LinkedIn: [https://www.linkedin.com/in/valerio-botto-4844b2190/](https://www.linkedin.com/in/valerio-botto-4844b2190/)
=======
Interactive AI agent for Technical/medical/legislative PDF(s) analysis. Upload documents, extract key topics, and get contextual answers via an AI conversational agent. Leverages LangChain/LangGraph, Neo4j knowledge graph, and vector DB for semantic retrieval, offering dynamic, personalized document consultation.

