"""
Graph definition for the Personal Assistant Agent.

This module defines the LangGraph StateGraph topology that orchestrates
the three core nodes: input_node, llm_node, and response_node.
"""

from langgraph.graph import StateGraph
from src.graph.state import AgentState
from src.nodes.input_node import input_node
from src.nodes.llm_node import llm_node
from src.nodes.response_node import response_node


def build_graph():
    """
    Build and compile the Phase 2 agent graph.

    Constructs a linear graph topology with three nodes:
    1. input_node: Converts user query into HumanMessage
    2. llm_node: Calls Ollama and tracks latency/tokens
    3. response_node: Formats output with metadata

    Graph topology (ASCII):
        START
          ↓
        input_node (query → HumanMessage)
          ↓
        llm_node (HumanMessage → AIMessage + metadata)
          ↓
        response_node (format output with metadata)
          ↓
         END

    State flow:
        input_node appends HumanMessage to messages list
            ↓
        llm_node appends AIMessage to messages list, adds metadata to _config
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
        - Edges: All sequential (no branching in Phase 2)
        - State: Shared mutable dict, managed by LangGraph
        - Reducers: add_messages on messages field, add_list on lists
    """
    # Create the state graph with AgentState schema
    builder = StateGraph(AgentState)

    # Add the three nodes to the graph
    builder.add_node("input", input_node)
    builder.add_node("llm", llm_node)
    builder.add_node("response", response_node)

    # Define entry point (first node to execute)
    builder.set_entry_point("input")

    # Add edges (control flow between nodes)
    builder.add_edge("input", "llm")  # After input_node, go to llm_node
    builder.add_edge("llm", "response")  # After llm_node, go to response_node

    # Define finish point (final node, graph stops here)
    builder.set_finish_point("response")

    # Compile the graph into an executable form
    # After compilation, the graph is immutable and thread-safe
    graph = builder.compile()

    return graph
