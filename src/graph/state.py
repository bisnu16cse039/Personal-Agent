"""
AgentState TypedDict - The shared memory contract for the LangGraph agent.

This module defines the schema that every node reads from and writes to.
It follows Option B from the design guide: rich state defined upfront with all
Phase 2–6 fields pre-allocated, enabling phase-by-phase filling without
refactoring the state contract.

Field naming convention:
- Immutable inputs: thread_id, user_id
- Messages & reasoning: messages, thinking, final_response
- Tools & retrieval: tool_calls, retrieved_docs, needs_search
- Memory & context: memory_context
- Aggregators: _all_docs (for reducers)
"""

from typing import Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def add_list(x: list, y: list) -> list:
    """Reducer: append y to x for list fields."""
    return x + y


class AgentState(dict):
    """
    Agent state schema - the single source of truth for graph execution.
    
    This is implemented as a TypedDict-compatible class for runtime compatibility
    with LangGraph's state management and reducer system.
    
    Fields are organized by lifecycle:
    
    INPUT LAYER (set once, read-only after):
    - thread_id: Conversation session identifier for checkpointer
    - user_id: User identifier for personalization and audit
    
    MESSAGE LAYER (updated throughout):
    - messages: Full conversation history, managed by add_messages reducer
    
    THINKING LAYER (Phase 3):
    - thinking: Hidden chain-of-thought reasoning before response
    
    ROUTING LAYER (Phase 3-4):
    - needs_search: Boolean flag indicating web search is required
    
    TOOL LAYER (Phase 4):
    - tool_calls: List of tool invocations and results
    - retrieved_docs: Documents/web results from search
    
    RESPONSE LAYER (Phase 2+):
    - final_response: Polished final response to user
    
    MEMORY LAYER (Phase 5):
    - memory_context: Retrieved long-term memory context
    
    INTERNAL (helpers):
    - _config: Runtime configuration dict passed to nodes
    """
    
    pass  # TypedDict implementation via __annotations__ below


# Define the type annotation schema that LangGraph will use
AgentState.__annotations__ = {
    # Core I/O
    "thread_id": str,                                                    # Checkpoint key
    "user_id": Optional[str],                                            # User context (Phase 5+)
    
    # Messages
    "messages": Annotated[list[BaseMessage], add_messages],              # Conversation history
    
    # Phase 2: Core loop
    # (no new fields)
    
    # Phase 3: Thinking layer
    "thinking": Optional[str],                                           # Hidden chain-of-thought
    "needs_search": bool,                                                # Routing signal
    
    # Phase 4: Tool augmentation
    "tool_calls": Annotated[list[dict], add_list],                       # Tool invocations
    "retrieved_docs": Annotated[list[str], add_list],                    # Web/search results
    
    # Phase 5: Memory & persistence
    "memory_context": Optional[str],                                     # Retrieved long-term facts
    
    # Phase 6: Evaluation
    "final_response": Optional[str],                                     # User-facing response
    
    # Runtime config (internal)
    "_config": dict,                                                     # Node-level config
}


# Provide initialization defaults for clarity
DEFAULT_AGENT_STATE = {
    "thread_id": "",
    "user_id": None,
    "messages": [],
    "thinking": None,
    "needs_search": False,
    "tool_calls": [],
    "retrieved_docs": [],
    "memory_context": None,
    "final_response": None,
    "_config": {},
}


def create_initial_state(thread_id: str, user_id: Optional[str] = None) -> dict:
    """
    Factory function to create a new agent state instance.
    
    Args:
        thread_id: Unique identifier for this conversation session
        user_id: Optional user identifier for personalization
        
    Returns:
        Initialized state dict ready for graph execution
    """
    return {
        "thread_id": thread_id,
        "user_id": user_id,
        "messages": [],
        "thinking": None,
        "needs_search": False,
        "tool_calls": [],
        "retrieved_docs": [],
        "memory_context": None,
        "final_response": None,
        "_config": {},
    }
