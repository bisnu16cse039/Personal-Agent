"""
Unit tests for llm_node.

Tests the LLM node in isolation by mocking the Ollama client,
including happy path, retry logic, and error scenarios.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.llm_node import llm_node


class TestLLMNode:
    """Test suite for llm_node function."""

    def test_llm_node_appends_ai_message_with_metadata(self):
        """
        Test 1: Happy path - LLM call succeeds on first attempt.

        Verifies:
        - AIMessage is created and appended to messages
        - Metadata is populated with latency and token counts
        - State structure is correct
        """
        # Arrange: Mock the Ollama client
        with patch("src.nodes.llm_node.get_ollama_client") as mock_get_client:
            # Create mock response object
            mock_response = MagicMock()
            mock_response.content = "AI is intelligence"
            mock_response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 50,
            }

            # Create mock client that returns our response
            mock_client = MagicMock()
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Create input state with HumanMessage
            state = {
                "messages": [HumanMessage(content="What is AI?")],
                "_config": {},
                "thread_id": "test_thread",
            }

            # Act: Call llm_node
            result = llm_node(state)

            # Assert: Verify structure
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert isinstance(result["messages"][0], AIMessage)
            assert result["messages"][0].content == "AI is intelligence"

            # Assert: Verify metadata
            assert "_config" in result
            assert "metadata" in result["_config"]
            assert result["_config"]["metadata"]["latency_ms"] > 0
            assert result["_config"]["metadata"]["input_tokens"] == 10
            assert result["_config"]["metadata"]["output_tokens"] == 50

            # Assert: Client was called once
            mock_client.invoke.assert_called_once()

    def test_llm_node_retries_on_timeout(self):
        """
        Test 2: Retry logic - First call times out, second succeeds.

        Verifies:
        - TimeoutError on first attempt triggers retry
        - Second attempt succeeds and returns valid response
        - Metadata is captured from successful attempt
        - Total invocation count is 2
        """
        # Arrange: Mock client that times out once, then succeeds
        with patch("src.nodes.llm_node.get_ollama_client") as mock_get_client:
            # Create mock response for successful second attempt
            mock_response = MagicMock()
            mock_response.content = "AI is intelligence"
            mock_response.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 50,
            }

            # Mock client with side_effect: first call raises TimeoutError, second returns response
            mock_client = MagicMock()
            mock_client.invoke.side_effect = [
                TimeoutError("Connection timeout"),
                mock_response,
            ]
            mock_get_client.return_value = mock_client

            # Create input state
            state = {
                "messages": [HumanMessage(content="What is AI?")],
                "_config": {},
                "thread_id": "test_thread",
            }

            # Act: Call llm_node (should handle retry internally)
            result = llm_node(state)

            # Assert: Verify retry worked and got valid response
            assert result["messages"][0].content == "AI is intelligence"
            assert result["_config"]["metadata"]["latency_ms"] > 0

            # Assert: Verify client was called exactly 2 times (1 failed + 1 success)
            assert mock_client.invoke.call_count == 2

    def test_llm_node_fails_after_max_retries(self):
        """
        Test 3: Error handling - Both retry attempts fail.

        Verifies:
        - TimeoutError on both attempts raises RuntimeError
        - Error message is clear and helpful
        - No partial state is returned
        """
        # Arrange: Mock client that always times out
        with patch("src.nodes.llm_node.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.invoke.side_effect = TimeoutError("Connection timeout")
            mock_get_client.return_value = mock_client

            state = {
                "messages": [HumanMessage(content="What is AI?")],
                "_config": {},
                "thread_id": "test_thread",
            }

            # Act & Assert: Call llm_node, expect RuntimeError after 2 attempts
            with pytest.raises(RuntimeError) as exc_info:
                llm_node(state)

            # Verify error message
            assert "timed out after 2 attempts" in str(exc_info.value)

            # Verify client was called exactly 2 times
            assert mock_client.invoke.call_count == 2

    def test_llm_node_requires_messages_in_state(self):
        """
        Test 4: Input validation - Missing messages raises clear error.

        Verifies:
        - ValueError raised if state has no messages
        - Error message guides user to run input_node first
        """
        # Arrange: State with no messages
        state = {
            "messages": [],
            "_config": {},
            "thread_id": "test_thread",
        }

        # Act & Assert: Call llm_node, expect ValueError
        with pytest.raises(ValueError) as exc_info:
            llm_node(state)

        # Verify error message is helpful
        assert "No messages in state" in str(exc_info.value)
        assert "input_node must run" in str(exc_info.value)

    def test_llm_node_handles_empty_response(self):
        """
        Test 5: Response validation - Empty LLM response raises error.

        Verifies:
        - ValueError raised if LLM returns empty content
        - Error message is clear
        """
        # Arrange: Mock client returns empty response
        with patch("src.nodes.llm_node.get_ollama_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.content = ""  # Empty response
            mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 0}

            mock_client = MagicMock()
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            state = {
                "messages": [HumanMessage(content="What is AI?")],
                "_config": {},
                "thread_id": "test_thread",
            }

            # Act & Assert: Call llm_node, expect ValueError
            with pytest.raises(ValueError) as exc_info:
                llm_node(state)

            # Verify error message
            assert "empty response" in str(exc_info.value)

    def test_llm_node_handles_missing_token_counts(self):
        """
        Test 6: Graceful degradation - LLM response without usage info.

        Verifies:
        - No error if response has no usage/token info
        - Token counts default to 0
        - Rest of functionality works normally
        """
        # Arrange: Mock response without usage info
        with patch("src.nodes.llm_node.get_ollama_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.content = "AI is intelligence"
            # No usage attribute on response
            del mock_response.usage

            mock_client = MagicMock()
            mock_client.invoke.return_value = mock_response
            mock_get_client.return_value = mock_client

            state = {
                "messages": [HumanMessage(content="What is AI?")],
                "_config": {},
                "thread_id": "test_thread",
            }

            # Act: Call llm_node (should handle missing usage gracefully)
            result = llm_node(state)

            # Assert: Verify it didn't crash and defaulted to 0
            assert result["messages"][0].content == "AI is intelligence"
            assert result["_config"]["metadata"]["input_tokens"] == 0
            assert result["_config"]["metadata"]["output_tokens"] == 0


# Run tests with: pytest tests/test_llm_node.py -v
