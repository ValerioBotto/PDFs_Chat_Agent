#!/usr/bin/env python3

import sys
import os
import logging
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our agent components
from backend.pdf_utils.agent.agent import _build_agent_workflow, invoke_agent, initialize_agent_tools_resources
from backend.pdf_utils.graph_db import GraphDB
from backend.pdf_utils.indexer import Indexer

# Mock LLM classes for testing
class MockLLM:
    def invoke(self, messages):
        # Mock planner response - always say no tools needed
        class MockResp:
            def __init__(self):
                self.content = '{"tool_calls": []}'
        return MockResp()

class MockSynthLLM:
    def invoke(self, messages):
        # Mock synthesis response
        class MockResp:
            def __init__(self):
                self.content = "This is a mock response for testing workflow"
        return MockResp()

async def test_workflow():
    """Test the workflow with a simple question to see if it gets past firewall."""
    try:
        # Initialize mock resources (we don't need real Neo4j for this test)
        mock_graph_db = None  # We'll handle this in the agent
        mock_indexer = None   # We'll handle this in the agent
        
        # Initialize the agent tools (pass None for now - we're testing workflow flow)
        # initialize_agent_tools_resources(mock_graph_db, mock_indexer)
        
        # Create mock LLMs
        planner_llm = MockLLM()
        synth_llm = MockSynthLLM()
        
        # Create checkpointer
        checkpointer = InMemorySaver()
        
        # Build the workflow
        logger.info("Building agent workflow...")
        agent_app = _build_agent_workflow(
            planner_llm=planner_llm, 
            synth_llm=synth_llm, 
            checkpointer=checkpointer
        )
        
        # Test with the problematic question
        test_question = "si possono impostare periodi vacanza?"
        chat_history = [
            HumanMessage(content="previous question"),
            AIMessage(content="previous answer") 
        ]
        
        logger.info(f"Testing workflow with question: '{test_question}'")
        
        # Use the invoke_agent function
        result = await invoke_agent(
            agent_app=agent_app,
            user_message=test_question,
            chat_history=chat_history,
            config={'configurable': {'thread_id': 'test-123'}}
        )
        
        logger.info(f"Workflow completed. Result: '{result}'")
        
    except Exception as e:
        logger.exception(f"Workflow test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_workflow())