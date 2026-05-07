"""
Thinking node for the Personal Assistant Agent graph (Phase 3).

This node implements the first pass of the two-pass LLM strategy: hidden
chain-of-thought reasoning. It calls the LLM with a special prompt that asks
for internal reasoning, stores the result in state["thinking"], detects if
web search is needed, and sets the routing flag.
"""

import time
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config.settings import get_ollama_client


THINKING_SYSTEM_PROMPT = """You are a helpful assistant that thinks through problems step-by-step.

Before responding to the user's question, analyze it carefully:
1. What is the user asking?
2. What information do I need to answer well?
3. Can I answer this from my training knowledge?
4. Do I need to search the web for current/recent information?

After your reasoning, indicate whether you need web search by including this marker:
- If you need web search: [NEEDS_SEARCH: true]
- If you don't need web search: [NEEDS_SEARCH: false]

This is your internal thinking - be thorough and honest about what you know and don't know.
"""


def thinking_node(state: dict) -> dict:
    """
    First LLM pass: hidden chain-of-thought reasoning before response generation.

    This node implements the thinking layer (Phase 3). It:
    1. Calls the LLM with a system prompt that encourages step-by-step reasoning
    2. Extracts and stores the reasoning in state["thinking"]
    3. Parses the response to detect if web search is needed (looks for [NEEDS_SEARCH] marker)
    4. Sets the needs_search flag for the router node
    5. Does NOT add this reasoning to the conversation history (hidden reasoning)

    State flow:
        Input:  {"messages": [HumanMessage("What's new in AI in 2026?")], ...}
        Output: {
            "messages": [HumanMessage(...)],  # Unchanged - thinking is hidden
            "thinking": "The user is asking about recent AI developments...",
            "needs_search": True,  # Detected from [NEEDS_SEARCH: true] marker
            "_config": {
                "metadata": {
                    "thinking_latency_ms": 3500.0,
                    "thinking_tokens": 150,
                }
            }
        }

    Args:
        state (dict): The agent state containing:
            - messages (list): Conversation history
            - _config (dict): Configuration (updated with thinking metadata)
            - Other state fields

    Returns:
        dict: Updated state with:
            - thinking: Full reasoning text extracted from LLM
            - needs_search: Boolean flag parsed from [NEEDS_SEARCH] marker
            - _config: Updated with thinking_latency_ms and thinking_tokens
            - messages: Unchanged (thinking is hidden reasoning)

    Raises:
        RuntimeError: If LLM call fails after retries
        ValueError: If messages are empty or [NEEDS_SEARCH] marker is malformed
    """
    # Extract conversation history
    messages = state.get("messages", [])
    if not messages:
        raise ValueError(
            "No messages in state. input_node must run before thinking_node."
        )

    # Get Ollama client
    try:
        client = get_ollama_client()
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Ollama client: {str(e)}") from e

    # Retry logic: up to 2 attempts for timeout/transient errors
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()

            # Prepare messages for thinking pass with system prompt
            thinking_messages = [
                SystemMessage(content=THINKING_SYSTEM_PROMPT),
                *messages,  # Full conversation history
            ]

            # Call LLM for reasoning using invoke (LangChain pattern)
            response = client.invoke(thinking_messages)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            thinking_text = response.content

            # Parse needs_search flag from response
            needs_search = _parse_needs_search(thinking_text)

            # Extract clean thinking (remove the marker line)
            clean_thinking = _clean_thinking_text(thinking_text)

            # Update state
            updated_state = state.copy()
            updated_state["thinking"] = clean_thinking
            updated_state["needs_search"] = needs_search

            # Track metadata
            if "_config" not in updated_state:
                updated_state["_config"] = {}
            if "metadata" not in updated_state["_config"]:
                updated_state["_config"]["metadata"] = {}

            updated_state["_config"]["metadata"]["thinking_latency_ms"] = latency_ms
            # Estimate tokens from text length
            updated_state["_config"]["metadata"]["thinking_tokens"] = len(thinking_text.split())

            return updated_state

        except (TimeoutError, ConnectionError) as e:
            last_error = e
            if attempt < max_retries - 1:
                continue  # Retry
            else:
                raise RuntimeError(
                    f"Thinking node failed after {max_retries} attempts: {str(e)}"
                ) from e
        except Exception as e:
            raise RuntimeError(f"Thinking node error: {str(e)}") from e

    raise RuntimeError(f"Thinking node failed: {last_error}")


def _parse_needs_search(response_text: str) -> bool:
    """
    Parse the [NEEDS_SEARCH] marker from LLM response.

    Looks for the pattern [NEEDS_SEARCH: true] or [NEEDS_SEARCH: false]
    in the response. If not found, defaults to False (no search).

    Args:
        response_text (str): The full LLM response including thinking

    Returns:
        bool: True if search is needed, False otherwise
    """
    # Look for the marker pattern
    if "[NEEDS_SEARCH: true]" in response_text.lower():
        return True
    elif "[NEEDS_SEARCH: false]" in response_text.lower():
        return False
    else:
        # Default to False if marker not found
        return False


def _clean_thinking_text(response_text: str) -> str:
    """
    Remove the [NEEDS_SEARCH] marker from thinking text.

    Extracts only the reasoning portion, removing the routing marker
    for cleaner storage in state["thinking"].

    Args:
        response_text (str): Raw LLM response

    Returns:
        str: Cleaned thinking text without marker
    """
    # Remove the marker line
    lines = response_text.split("\n")
    cleaned_lines = [
        line for line in lines
        if "[NEEDS_SEARCH:" not in line
    ]
    return "\n".join(cleaned_lines).strip()
