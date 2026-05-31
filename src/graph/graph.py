"""
Graph definition for the Personal Assistant Agent.

This module defines the LangGraph StateGraph topology that orchestrates
the three core nodes: input_node, llm_node, and response_node.
"""

from langgraph.graph import StateGraph
from src.graph.state import AgentState
from src.nodes.input_node import input_node
from src.nodes.thinking_node import thinking_node
from src.nodes.llm_node import llm_node
from src.nodes.response_node import response_node
from src.config.settings import get_settings


def build_graph():
    """
    Build and compile the agent graph (Phase 2–3).

    Constructs a graph topology with conditional thinking layer support.
    - If ENABLE_THINKING_LAYER=true: Full Phase 3 with reasoning (4 nodes)
    - If ENABLE_THINKING_LAYER=false: Fast Phase 2 mode (2 nodes)

    Graph topology with thinking enabled (Phase 3):
        START
          ↓
        input_node (query → HumanMessage)
          ↓
        thinking_node (reasoning, set needs_search flag)
          ↓
        [conditional branch on needs_search]
          ├→ [true] llm_node (with search results)
          └→ [false] llm_node (direct response)
          ↓
        response_node (format output with metadata)
          ↓
         END

    Graph topology with thinking disabled (Phase 2):
        START
          ↓
        input_node (query → HumanMessage)
          ↓
        llm_node (direct response)
          ↓
        response_node (format output with metadata)
          ↓
         END

    Returns:
        Compiled LangGraph graph that can be invoked with state dict.
        The returned graph is immutable and ready for invocation.

    Example:
        >>> from src.graph.state import create_initial_state
        >>> graph = build_graph()
        >>> state = create_initial_state("test_thread")
        >>> state["_config"] = {"user_query": "What is AI?"}
        >>> result = graph.invoke(state)
        >>> print(result["final_response"]["response"])
        AI is intelligence...

    Raises:
        Exception: If graph compilation fails (usually indicates missing nodes/edges)
    """
    settings = get_settings()
    
    # Create the state graph with AgentState schema
    builder = StateGraph(AgentState)

    # Always add core nodes
    builder.add_node("input", input_node)
    builder.add_node("llm", llm_node)
    builder.add_node("response", response_node)

    # Define entry point (first node to execute)
    builder.set_entry_point("input")

    # Conditionally add thinking layer (Phase 3)
    if settings.ENABLE_THINKING_LAYER:
        builder.add_node("reason", thinking_node)
        builder.add_edge("input", "reason")  # input → thinking
        
        # Conditional edges from thinking_node based on needs_search flag
        builder.add_conditional_edges(
            "reason",
            lambda x: "route_to_search" if x.get("needs_search", False) else "route_to_response",
            {
                "route_to_search": "llm",  # TODO: Phase 4 - change to "search"
                "route_to_response": "llm",
            },
        )
    else:
        # Fast mode: skip thinking, go directly from input to llm (Phase 2)
        builder.add_edge("input", "llm")  # input → llm (no thinking)

    # Final edge to response formatting (same in both modes)
    builder.add_edge("llm", "response")

    # Define finish point (final node, graph stops here)
    builder.set_finish_point("response")

    # Compile the graph into an executable form
    # After compilation, the graph is immutable and thread-safe
    graph = builder.compile()

    return graph
