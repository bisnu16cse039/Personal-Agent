"""
Input node for the Personal Assistant Agent graph.

This node handles the entry point of user queries, converting them into
LangChain message format for LLM processing.
"""

from langchain_core.messages import HumanMessage


def input_node(state: dict) -> dict:
    """
    Convert raw user query into HumanMessage for LLM input.

    This node extracts the user query from the state configuration,
    wraps it in a LangChain HumanMessage, and returns it for the LLM node
    to process. It is the entry point to the graph.

    Args:
        state (dict): The agent state containing:
            - _config (dict): Configuration dict with user_query
            - messages (list): Existing message history (typically empty on first invocation)
            - Other state fields managed by LangGraph

    Returns:
        dict: Updated state with HumanMessage appended to messages list:
            - messages: [HumanMessage(content=query)]

    Raises:
        ValueError: If user_query is not found in state["_config"]
        KeyError: If _config is missing from state

    Example:
        >>> state = {
        ...     "_config": {"user_query": "What is AI?"},
        ...     "messages": [],
        ...     "thread_id": "test",
        ... }
        >>> result = input_node(state)
        >>> result["messages"][0].content
        'What is AI?'
    """
    # Extract user query from state config
    config = state.get("_config", {})
    user_query = config.get("user_query")

    if not user_query:
        raise ValueError(
            "No user_query found in state['_config']. "
            "Ensure state is initialized with: "
            "state['_config'] = {'user_query': 'your query'}"
        )

    # Wrap query in HumanMessage for LLM
    human_message = HumanMessage(content=user_query)

    # Return updated state with message added
    # The add_messages reducer will append this to the messages list
    return {"messages": [human_message]}
