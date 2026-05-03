"""
LLM node for the Personal Assistant Agent graph.

This node calls the local Ollama LLM, implements retry logic for timeouts,
and tracks execution metadata (latency, token counts).
"""

import time
from typing import Dict, Any

from langchain_core.messages import AIMessage
from src.config.settings import get_ollama_client


def llm_node(state: dict) -> dict:
    """
    Call Ollama LLM and track latency and token usage.

    This node is the core reasoning component. It takes the conversation history
    from state["messages"], sends it to the local Ollama LLM, and appends the
    response as an AIMessage. It implements retry logic (up to 2 attempts) for
    timeout errors and captures execution metadata.

    State flow:
        Input:  {"messages": [HumanMessage("What is AI?")], ...}
        Output: {
            "messages": [..., AIMessage("AI is intelligence")],
            "_config": {
                "metadata": {
                    "latency_ms": 2500.5,
                    "input_tokens": 10,
                    "output_tokens": 50,
                }
            }
        }

    Args:
        state (dict): The agent state containing:
            - messages (list): Conversation history to send to LLM
            - _config (dict): Configuration (updated with metadata)
            - Other state fields managed by LangGraph

    Returns:
        dict: Updated state with:
            - messages: Previous messages plus new AIMessage appended
            - _config: Updated with metadata dict containing latency and tokens

    Raises:
        RuntimeError: If LLM call fails after 2 retry attempts (timeout, API error, etc.)
        ConnectionError: If unable to get Ollama client
        ValueError: If LLM returns empty or invalid response

    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> state = {
        ...     "messages": [HumanMessage(content="What is AI?")],
        ...     "_config": {},
        ...     "thread_id": "test",
        ... }
        >>> result = llm_node(state)
        >>> isinstance(result["messages"][-1], AIMessage)
        True
        >>> result["_config"]["metadata"]["latency_ms"] > 0
        True
    """
    # Extract conversation history from state and validate input
    messages = state.get("messages", [])

    if not messages:
        raise ValueError(
            "No messages in state. input_node must run before llm_node to populate messages."
        )

    # Get Ollama client (with validation from settings.py)
    try:
        client = get_ollama_client()
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Ollama client: {str(e)}") from e

    # Retry logic: up to 2 attempts for timeout/transient errors
    max_retries = 2
    response_text = None
    latency_ms = 0.0
    last_error = None

    for attempt in range(max_retries):
        try:
            # Measure latency
            start_time = time.perf_counter()
            response = client.invoke(messages)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000  # Convert to ms

            # Extract response text
            response_text = response.content

            if not response_text:
                raise ValueError("LLM returned empty response")

            # Success - break out of retry loop
            break

        except TimeoutError as e:
            last_error = e
            if attempt == max_retries - 1:
                # Last attempt failed
                raise RuntimeError(
                    f"LLM call timed out after {max_retries} attempts. "
                    f"Ollama may be slow or unresponsive."
                ) from e
            # Retry on next iteration
            continue

        except ValueError as e:
            # ValueError (empty response) should fail immediately, no retry
            raise

        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                # Last attempt failed
                raise RuntimeError(
                    f"LLM call failed after {max_retries} attempts: {str(e)}"
                ) from e
            # Retry on next iteration
            continue

    # Create AI message from response
    ai_message = AIMessage(content=response_text)

    # Extract token counts if available (Ollama may not expose these)
    # This is a safe fallback pattern
    input_tokens = 0
    output_tokens = 0

    if hasattr(response, "usage") and response.usage:
        input_tokens = response.usage.get("prompt_tokens", 0)
        output_tokens = response.usage.get("completion_tokens", 0)

    # Build metadata dict
    metadata: Dict[str, Any] = {
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    # Return updated state with AIMessage and metadata
    return {
        "messages": [ai_message],  # add_messages reducer will append to existing list
        "_config": {"metadata": metadata},
    }
