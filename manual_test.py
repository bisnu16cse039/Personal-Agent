from src.graph.graph import build_graph
from src.graph.state import create_initial_state

# Build graph once
graph = build_graph()

# Define test queries
test_queries = [
    "Location of Dinajpur in Bangladesh?",
    "What is AI?"
    # "Two years ago, a father was 5 times as old as his son. In 6 years, he will be 3 times as old.Find their current ages.",
    # "You have only 3 coins, and they add up to 30. One of them is not a 10-unit coin. What are the coins?",
    # "Find three consecutive integers such that the square of the middle number is 4 more than the product of the other two.", 
]

# Run all tests
results = []
for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"Query {i}: {query}")
    print('='*70)

    try:
        # Create state
        state = create_initial_state(f"test_thread_{i}")
        state["_config"] = {"user_query": query}

        # Invoke graph
        result = graph.invoke(state)

        # Extract response
        response = result['final_response']['response']
        metadata = result['final_response']['metadata']

        # Print results
        print(f"Response:\n{response}...")  
        print(f"\nMetadata:")
        print(f"  Latency: {metadata['latency_ms']:.1f}ms")
        print(f"  Input tokens: {metadata['input_tokens']}")
        print(f"  Output tokens: {metadata['output_tokens']}")

        # Check quality
        is_coherent = len(response) > 20 and response[0].isupper()
        status = "✅ PASS" if is_coherent else "❌ FAIL"
        print(f"{status} - Response is {'coherent' if is_coherent else 'incoherent'}")

        results.append({
            "query": query,
            "status": status,
            "latency_ms": metadata['latency_ms'],
            "response_length": len(response),
        })

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        results.append({
            "query": query,
            "status": "❌ ERROR",
            "error": str(e),
        })

# Summary
print(f"\n\n{'='*70}")
print("SUMMARY")
print('='*70)

passed = sum(1 for r in results if "✅" in r["status"])
total = len(results)

print(f"Tests passed: {passed}/{total}")
print(f"\nDetails:")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['status']} - {r['query'][:40]}...")

if passed == total:
    print("\n🎉 All manual tests passed!")
else:
    print(f"\n⚠️ {total - passed} test(s) failed")