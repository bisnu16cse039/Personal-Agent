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


def build_graph():
    """
    Build and compile the Phase 3 agent graph.

    Constructs a branching graph topology with four nodes:
    1. input_node: Converts user query into HumanMessage
    2. thinking_node: Hidden chain-of-thought reasoning, sets needs_search flag (Phase 3)
    3. llm_node: Calls Ollama with thinking context for final response
    4. response_node: Formats output with metadata

    Graph topology (ASCII):
        START
          ↓
        input_node (query → HumanMessage)
          ↓
        thinking_node (reasoning, set needs_search flag)
          ↓
        [conditional branch on needs_search]
          ├→ [true] search_node (Phase 4: web search)
          └→ [false] llm_node (final response)
          ↓
        response_node (format output with metadata)
          ↓
         END

    State flow:
        input_node appends HumanMessage to messages list
            ↓
        thinking_node stores reasoning in state["thinking"], sets needs_search flag
            ↓
        [thinking_node's conditional edges evaluate needs_search]
            ↓
        llm_node appends AIMessage to messages list (using thinking as context),
            adds metadata to _config
            ↓
        response_node extracts final AIMessage and metadata, formats for display

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

    Topology details:
        - Entry point: input_node (mandatory starting point)
        - Finish point: response_node (mandatory exit point)
        - Edges: Sequential to thinking, then conditional branching (Phase 3+)
        - Conditional edges: Directly from thinking_node based on needs_search flag
        - State: Shared mutable dict, managed by LangGraph
        - Reducers: add_messages on messages field, add_list on lists
        - Phase 4 prep: route_to_search will eventually connect to search_node
    """
    # Create the state graph with AgentState schema
    builder = StateGraph(AgentState)

    # Add the four nodes to the graph
    builder.add_node("input", input_node)
    builder.add_node("reason", thinking_node)  # Node key is "reason", stores in state["thinking"]
    builder.add_node("llm", llm_node)
    builder.add_node("response", response_node)

    # Define entry point (first node to execute)
    builder.set_entry_point("input")

    # Add edges (control flow between nodes)
    builder.add_edge("input", "reason")  # After input_node, run thinking_node

    # Conditional edges from thinking_node
    # The thinking_node sets needs_search flag which determines routing
    # Phase 3: route_to_response (needs_search=False) goes to llm for final answer
    # Phase 4: route_to_search (needs_search=True) will go to search_node first
    builder.add_conditional_edges(
        "reason",
        lambda x: "route_to_search" if x.get("needs_search", False) else "route_to_response",
        {
            "route_to_search": "llm",  # TODO: Phase 4 - change to "search"
            "route_to_response": "llm",
        },
    )

    # Final edge to response formatting
    builder.add_edge("llm", "response")  # After llm_node, go to response_node

    # Define finish point (final node, graph stops here)
    builder.set_finish_point("response")

    # Compile the graph into an executable form
    # After compilation, the graph is immutable and thread-safe
    graph = builder.compile()

    return graph
