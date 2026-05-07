"""
Router node for the Personal Assistant Agent graph (Phase 3-4).

This node implements conditional routing based on whether web search is needed.
It examines the thinking layer's needs_search flag and routes execution to either:
- search_node (Phase 4, when needs_search=True)
- response_node (when needs_search=False)

This is the entry point to the conditional fan-out topology.
"""


def router_node(state: dict) -> dict:
    """
    Route to search or response based on the needs_search flag.

    This node is a no-op node that validates routing conditions and participates
    in the graph topology. The actual routing decision is made by the conditional
    edges function. This node returns an empty dict (no state modification).

    State flow:
        Input:  {"thinking": "...", "needs_search": True, ...}
        Output: {}  # Empty dict - routing decided by conditional edges function

        Input:  {"thinking": "...", "needs_search": False, ...}
        Output: {}  # Empty dict - routing decided by conditional edges function

    Args:
        state (dict): The agent state containing:
            - needs_search (bool): Flag set by thinking_node indicating if search is needed
            - thinking (str): The reasoning from the thinking layer
            - Other state fields

    Returns:
        dict: Empty dict (no state changes). The routing decision is made by the
            conditional edges function which reads state["needs_search"].

    Raises:
        KeyError: If needs_search is not in state (thinking_node must run first)
        ValueError: If needs_search is not a boolean

    Example:
        >>> state = {"needs_search": True, "thinking": "I need current info"}
        >>> result = router_node(state)
        >>> result
        {}

        >>> state = {"needs_search": False, "thinking": "I can answer this"}
        >>> result = router_node(state)
        >>> result
        {}
    """
    # Validate preconditions (thinking_node must have run)
    needs_search = state.get("needs_search")

    if needs_search is None:
        raise KeyError(
            "needs_search not found in state. thinking_node must run before router_node."
        )

    if not isinstance(needs_search, bool):
        raise ValueError(
            f"needs_search must be bool, got {type(needs_search).__name__}"
        )

    # No state modifications - routing is handled by conditional edges function
    return {}
