"""
Response node for the Personal Assistant Agent graph.

This node formats the final AI response and metadata for display to the user.
It extracts the last AI message and associated metadata, then packages them
in a user-friendly format.
"""

from langchain_core.messages import AIMessage


def response_node(state: dict) -> dict:
    """
    Format AI response and metadata for terminal output (Phase 3+).

    This node is the exit point of the graph. It extracts the final response
    from the AI message, combines it with latency and token metadata, includes
    the thinking layer reasoning, and returns a formatted dictionary suitable
    for display to the user.

    Args:
        state (dict): The agent state containing:
            - messages (list): Full conversation history with final AIMessage
            - thinking (str): Hidden reasoning from thinking_node (Phase 3)
            - _config (dict): Configuration including metadata dict
            - Other state fields managed by LangGraph

    Returns:
        dict: Updated state with formatted response:
            - final_response: {
                "response": str,  # The AI's text response
                "thinking": str,  # Hidden reasoning from thinking layer (Phase 3)
                "metadata": {     # Execution metadata
                    "latency_ms": float,
                    "input_tokens": int,
                    "output_tokens": int,
                }
              }

    Raises:
        ValueError: If no messages exist or last message is not AIMessage
        KeyError: If metadata is missing from _config

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> state = {
        ...     "messages": [AIMessage(content="AI is intelligence")],
        ...     "_config": {
        ...         "metadata": {
        ...             "latency_ms": 2500.5,
        ...             "input_tokens": 10,
        ...             "output_tokens": 50,
        ...         }
        ...     },
        ...     "thread_id": "test",
        ... }
        >>> result = response_node(state)
        >>> result["final_response"]["response"]
        'AI is intelligence'
        >>> result["final_response"]["metadata"]["latency_ms"]
        2500.5
    """
    # Extract messages from state
    messages = state.get("messages", [])

    if not messages:
        raise ValueError(
            "No messages found in state. Graph must execute LLM node before response node."
        )

    # Get last message (should be AIMessage from llm_node)
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Last message is not AIMessage, got {type(last_message).__name__}. "
            f"Ensure llm_node runs before response_node."
        )

    # Extract response content
    response_text = last_message.content

    # Extract thinking from state (Phase 3+)
    thinking = state.get("thinking", None)

    # Extract metadata from config
    config = state.get("_config", {})
    metadata = config.get("metadata", {})

    if not metadata:
        raise ValueError(
            "No metadata found in state['_config']. "
            "Ensure llm_node populates metadata with latency_ms, token counts."
        )

    # Format final response for user
    formatted_response = {
        "response": response_text,
        "thinking": thinking,  # Include reasoning for transparency/debugging
        "metadata": {
            "latency_ms": metadata.get("latency_ms", 0),
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
        },
    }

    # Return updated state with final_response field
    return {"final_response": formatted_response}
