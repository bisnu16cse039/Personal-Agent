# Personal Agent — Design Guide & Learning Roadmap

> **Purpose of this document:**  
> This is your single source of truth for planning, building, and evolving your local personal AI assistant.  
> It captures your goals, every architectural decision you need to make, the options available, the terminology you will encounter, and a learning roadmap to understand each concept deeply before writing code.

---

## 1. Project Vision

### What you are building
A local, privacy-first personal assistant agent that:
- Runs entirely on your **MacBook Pro M4 (24 GB RAM)** — no data leaves your machine
- Uses **LangGraph** as the agent orchestration framework
- Uses **Ollama** or **LM Studio** as the local inference server
- Gets progressively smarter as you evaluate, monitor, and update it over time
- Serves as a genuine daily-use tool, not just a demo

### Your two goals (and why they are complementary)
| Goal | What it means in practice |
|---|---|
| **Learn agent development deeply** | Understand every design decision — not just copy-paste code |
| **Build a useful daily agent** | Each phase adds real capability you can use immediately |

These goals reinforce each other. A well-understood system is one you can actually improve. An agent you use daily gives you real evaluation data.

### Guiding constraints
- **Privacy first** — all inference, storage, and retrieval is local
- **Build iteratively** — each phase is a runnable, testable system on its own
- **Evaluate continuously** — no phase is "done" until you have a way to measure it
- **Understand before automating** — use GitHub Copilot to accelerate, not replace, understanding

---

## 2. The Six-Phase Roadmap

Each phase builds on the last. You should have a working, runnable agent after every single phase — not just at the end.

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6
  │             │             │             │             │             │
Design      Core Loop     Thinking      Tools        Memory          UI
Schema &    Basic QA      Layer         (Search)     Short &       Chainlit +
Topology    via LLM       Chain-of-     Web +        Long-term     Evaluation
                          Thought       Privacy      ChromaDB      Dashboard
```

### Phase summaries

**Phase 1 — Architecture & System Design**  
No code yet. You design the state schema, graph topology, and pick your local model. This phase exists because fixing a wrong architecture decision in Phase 4 is expensive. Get the blueprint right first.

**Phase 2 — Core Loop (Basic QA)**  
Your first runnable graph. User query goes in, local LLM responds, answer comes out. The simplest possible thing that actually works end-to-end. You will spend time here understanding how LangGraph compiles and invokes a graph.

**Phase 3 — The Thinking Layer**  
Add a hidden reasoning step before the final response. The model "thinks" first (chain-of-thought), writes its reasoning to state, and then produces a cleaner final answer. This is also where you detect whether a web search is needed.

**Phase 4 — Tool Augmentation**  
Add a conditional branch: if the thinking node says knowledge is insufficient, route to a DuckDuckGo search node, retrieve results, and inject them into the final response. First encounter with conditional edges.

**Phase 5 — Memory & Persistence**  
Two types of memory: short-term (SQLite checkpointer — conversation history across sessions) and long-term (ChromaDB vector store — specific facts about you and your preferences, retrieved via semantic search).

**Phase 6 — Local UI & Evaluation**  
Wrap the graph in a Chainlit chat interface with streaming. Add an evaluation layer: LLM-as-judge scoring, subjective ratings, and a Grafana or custom dashboard for monitoring quality over time.

---

## 3. The Four Architectural Decisions (Phase 1)

These are the decisions you must make before writing any graph code. Each one has lasting consequences on every phase that follows.

---

### Decision 1 — State Schema Design Philosophy

**What it is:**  
The `AgentState` TypedDict is the shared memory of your graph. Every node reads from it and writes back to it. Nodes do not call each other directly — they only communicate through state.

**Why it matters:**  
The schema you define in Phase 1 is the interface contract for every node you write in Phases 2–6. Changing it later is possible but painful, especially once the checkpointer is persisting it to disk.

#### Option A — Minimal state
Only add fields when you need them. Start with just `messages`.

| | |
|---|---|
| **Pros** | Fastest to understand; easiest to debug; forces you to feel the friction of each limitation |
| **Cons** | Refactoring state mid-project is risky; each phase requires touching state.py |
| **Best for** | Learning the mechanics of LangGraph state flow |

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

#### Option B — Rich state upfront *(recommended)*
Define all fields in Phase 1 but leave most as `None` or empty. Fill them in as you build each phase.

| | |
|---|---|
| **Pros** | No refactors; nodes are decoupled from the start; checkpointer persists rich context; evaluation fields exist before you need them |
| **Cons** | Many fields are unused early on; slightly intimidating to see `None` everywhere |
| **Best for** | Building a system you intend to maintain and grow |

```python
class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    thinking:       Optional[str]
    tool_calls:     Annotated[list[dict], lambda x, y: x + y]
    retrieved_docs: list[str]
    memory_context: Optional[str]
    thread_id:      str
    needs_search:   bool
    final_response: Optional[str]
```

#### Option C — Nested sub-states *(advanced — skip for now)*
Different parts of the state are typed as separate TypedDicts and owned by different sub-graphs.

| | |
|---|---|
| **Pros** | Very clean separation of concerns; scales to multi-agent systems |
| **Cons** | LangGraph reducers don't merge nested dicts cleanly by default; debugging requires unpacking layers |
| **Best for** | Phase 6+ when you introduce supervisor agents |

---

### Decision 2 — Graph Topology

**What it is:**  
The topology is the shape of your graph — which nodes exist, how control moves between them, and whether the graph can loop back on itself. This is defined through edges in LangGraph.

**Why it matters:**  
Unlike the state schema, topology changes require re-wiring the entire graph. Choose deliberately.

#### Option A — Linear chain
Nodes execute one after another in a fixed sequence. No branching, no loops.

```
START → input_node → llm_node → END
```

| | |
|---|---|
| **Pros** | Completely predictable; trivial to debug; no risk of infinite loops |
| **Cons** | Every query costs the same compute; no ability to skip or retry |
| **Best for** | Phase 2 learning only |

#### Option B — Conditional fan-out *(recommended)*
A router node inspects the state and returns a string naming the next node. This creates branches.

```
START → memory_node → thinking_node
          ↓
       router_node ── needs_search=True ──→ search_node → respond_node → END
                   ── needs_search=False ──────────────→ respond_node → END
```

| | |
|---|---|
| **Pros** | Fast path for simple queries; natural extension point for more tools; easy to understand and test |
| **Cons** | Router logic can become messy if-else chains over time |
| **Best for** | A daily-use assistant — this is the right production shape |

#### Option C — ReAct loop *(advanced)*
The agent loops: it takes an action, observes the result, decides whether to act again, and repeats until it decides it is done.

```
START → agent_node → should_continue?
                         ↓ yes
                      tool_node → agent_node (loop)
                         ↓ no
                        END
```

| | |
|---|---|
| **Pros** | Autonomous multi-step reasoning; handles open-ended tasks naturally |
| **Cons** | Risk of infinite loops; unpredictable latency; overkill for single-turn Q&A |
| **Best for** | Tasks like "research this topic and write a report" — retrofit later |

#### Option D — Hierarchical / supervisor *(advanced — skip for now)*
A supervisor agent breaks a task into sub-tasks and delegates to specialist sub-agents (a search agent, a code agent, a writing agent).

| | |
|---|---|
| **Pros** | Scales to complex multi-agent workflows |
| **Cons** | Significant complexity; hard to debug sub-graph state |
| **Best for** | Phase 6+ territory |

---

### Decision 3 — Node Granularity

**What it is:**  
A node is a Python function. Granularity is how much responsibility each function has. There is no technical constraint — a single node could do everything. The question is what makes your agent maintainable and evaluable.

**Why it matters:**  
If a node does five things, you cannot tell which of those five things caused a bad output. Evaluation requires isolation.

#### Option A — Monolithic node
One node handles retrieval, reasoning, search, and response.

| | |
|---|---|
| **Pros** | Fast to write initially |
| **Cons** | Untestable in isolation; one bug breaks everything; impossible to profile or swap one concern |
| **Best for** | Never. This is the temptation to resist. |

#### Option B — Atomic single-responsibility *(recommended)*
Each node has one job with a clear input → output contract.

| | |
|---|---|
| **Pros** | Unit-testable with mock state; easy to swap (replace search_node without touching reasoning); LangSmith traces show exactly where time is spent |
| **Cons** | More files; risk of over-splitting |
| **Best for** | Any agent you intend to evaluate and improve |

Mental test: can you write a one-sentence docstring that fully describes what this node does? If not, it is doing too much.

#### Option C — Functional grouping
Group nodes that always run together (e.g., `memory_retrieve` and `memory_inject` become one `memory_node`).

| | |
|---|---|
| **Pros** | Pragmatic balance; reasonable number of graph nodes |
| **Cons** | Cohesion boundaries are subjective and drift over time |
| **Best for** | A middle-ground pragmatic approach |

---

### Decision 4 — LLM Call Strategy

**What it is:**  
How many times do you call the local model per user turn, and for what purpose each time?

**Why it matters:**  
Every extra LLM call adds latency (~2–4 seconds on M4). But a single call cannot simultaneously reason carefully and respond polished output. This is a direct quality vs. latency trade-off.

#### Option A — Single call per turn *(start here)*
One prompt goes in, one response comes out. The model must do everything at once.

| | |
|---|---|
| **Pros** | Lowest latency; simplest to debug |
| **Cons** | No separation between reasoning and response; harder to evaluate reasoning quality |
| **Best for** | Phase 2. You will feel the quality ceiling naturally. |

#### Option B — Two-pass: think then respond *(recommended)*
Call 1: hidden chain-of-thought reasoning → writes to `state.thinking`  
Call 2: final response using the reasoning as internal context

| | |
|---|---|
| **Pros** | Reasoning and polish are independently improvable; `thinking` field is the most valuable evaluation signal you have; natural home for `needs_search` detection |
| **Cons** | 2× LLM calls; ~2–4s extra latency per query |
| **Best for** | A daily-use assistant where answer quality matters more than raw speed |

#### Option C — Different models per node *(advanced)*
Use a small fast model (e.g., `llama3.1:8b`) for routing and a large quality model (e.g., `qwen3:14b`) for reasoning and response.

| | |
|---|---|
| **Pros** | Optimal cost/quality at each step |
| **Cons** | On 24 GB unified memory, loading two large models simultaneously causes memory swapping; complex prompt format differences between models |
| **Best for** | After you have profiled memory usage in Phase 3 |

---

## 4. Hardware & Model Reference

### Your hardware
- **Chip:** Apple M4
- **RAM:** 24 GB unified memory (shared between CPU and GPU — no discrete VRAM)
- **Key implication:** Loading a model plus ChromaDB plus the OS leaves roughly 12–14 GB for model weights

### Model recommendations

| Model | Disk size | RAM at Q5_K_M | Best for |
|---|---|---|---|
| `qwen3:14b` | ~9 GB | ~11 GB | Primary reasoner — best instruction following + tool use for this budget |
| `qwen3:30b-a3b` (MoE) | ~18 GB | ~20 GB | Near-GPT-4 quality; MoE means faster than its size suggests; tight on 24 GB |
| `deepseek-r1:14b` | ~9 GB | ~11 GB | Strong chain-of-thought; slightly slower than Qwen3 |
| `llama3.1:8b` | ~5 GB | ~6 GB | Fast router or fallback model |
| `nomic-embed-text` | ~274 MB | negligible | Local embeddings for ChromaDB RAG |

### What quantization means

A model's weights are stored as numbers. Full precision is 32-bit floating point (FP32). Quantization compresses these numbers to fewer bits, reducing RAM usage at a small quality cost.

| Quantization | Bits | Quality | RAM usage |
|---|---|---|---|
| `Q8_0` | 8-bit | Near-lossless | High |
| `Q5_K_M` | 5-bit | Excellent — recommended | Medium |
| `Q4_K_M` | 4-bit | Good — noticeable quality drop | Low |
| `Q2_K` | 2-bit | Degraded — use only if desperate | Very low |

**Your choice:** `Q5_K_M` — best quality-to-RAM ratio for 24 GB unified memory.

```bash
ollama pull qwen3:14b          # Defaults to Q5_K_M
ollama pull nomic-embed-text   # Embedding model for ChromaDB
```

---

## 5. Key Terms Glossary

This section is your self-study reference. For each term, a plain-language definition is given, followed by what to search or read to go deeper.

---

### LangGraph core concepts

**StateGraph**  
The main class in LangGraph. You add nodes and edges to it, then compile it into a runnable object. Think of it as a directed graph where each node is a Python function and edges define the allowed transitions.  
*Study:* LangGraph quickstart docs → `StateGraph` class reference

**TypedDict**  
A Python standard library class that lets you define a dictionary with typed keys. LangGraph uses it to define the state schema. Unlike a dataclass, it's still a plain dict at runtime — no class overhead.  
*Study:* PEP 589 (TypedDict specification) — a 15-minute read

**Annotated**  
A Python typing construct that lets you attach metadata to a type hint. LangGraph uses it to attach *reducer functions* to state fields, telling LangGraph how to merge updates from nodes.  
*Study:* Python `typing.Annotated` docs + LangGraph "reducers" concept page

**Reducer**  
A function that tells LangGraph how to combine the existing value of a state field with a new value returned by a node. The default reducer replaces. `add_messages` is a built-in reducer that appends. You can write custom reducers (e.g., `lambda x, y: x + y` to append to a list).  
*Study:* LangGraph docs → "State reducers" section

**Node**  
A Python function that takes the current `AgentState` as input and returns a dictionary of fields to update. Nodes are the workers of your graph. They do not call each other — they only read and write state.  
*Study:* LangGraph docs → "Nodes" section

**Edge**  
A directed connection between two nodes. A normal edge always transitions from node A to node B. A conditional edge calls a function, and the return value (a string) determines which node runs next.  
*Study:* LangGraph docs → `add_edge` vs `add_conditional_edges`

**Conditional edge**  
An edge where the next node is determined at runtime by a router function. The router receives the current state and returns a string that maps to one of several possible next nodes. This is how branching and tool routing work.  
*Study:* LangGraph conditional edges example

**Checkpointer**  
A persistence layer that saves a snapshot of the full graph state after every node execution. The `SqliteSaver` stores these snapshots in a local SQLite file. This is what enables multi-turn conversation memory — on the next turn, the graph restores its state from the snapshot matching `thread_id`.  
*Study:* LangGraph docs → "Persistence" and "Checkpointers"

**Thread ID**  
A string key that identifies a conversation session. The checkpointer uses it to store and retrieve state snapshots. Same user, different topic → different thread ID. You control this from your UI.  
*Study:* LangGraph "Configuration" docs — the `configurable` key

**Compile**  
The step that converts your `StateGraph` builder into an executable graph. You call `builder.compile()` once. After compilation, the graph is immutable. This step validates your node/edge wiring.  
*Study:* LangGraph docs → `compile()` method

---

### Agent reasoning patterns

**Chain-of-Thought (CoT)**  
A prompting technique where you ask the model to reason step-by-step before giving a final answer. The intermediate reasoning is often called a "scratchpad". The key insight: models produce better answers when they reason out loud first.  
*Study:* Wei et al. 2022 "Chain-of-Thought Prompting" (Google Research paper) — freely available

**ReAct (Reason + Act)**  
A pattern where the model alternates between reasoning about a situation and taking an action (calling a tool). It loops: reason → act → observe result → reason again → act again → ... until it decides it is done.  
*Study:* Yao et al. 2022 "ReAct: Synergizing Reasoning and Acting in Language Models"

**Hidden reasoning / thinking trace**  
Keeping the model's chain-of-thought internal to the agent — stored in state but never shown in the UI. This gives you the quality benefit of CoT without cluttering the chat interface. Qwen3 supports a native `<think>...</think>` format for this.  
*Study:* Qwen3 model card — the "thinking mode" section

**Tool use / function calling**  
The ability for an LLM to output a structured request to call an external function (a tool), rather than just outputting text. The graph intercepts this, executes the tool, and feeds the result back to the model.  
*Study:* LangChain docs → "Tool calling" + OpenAI function calling spec (same concept)

**Router / intent classifier**  
A node or function that looks at the current state (often the user query or the thinking trace) and decides which path the graph should take. In Phase 4 your router decides: does this query need a web search or not?  
*Study:* LangGraph conditional edges docs

---

### Memory concepts

**Short-term memory (in-context)**  
Everything in the `messages` list that gets passed to the LLM in its context window. Limited by the model's context length (typically 8k–128k tokens). Managed automatically by the `add_messages` reducer.

**Short-term memory (across sessions)**  
Implemented via the `SqliteSaver` checkpointer. Even if you close and reopen the app, the graph can restore the full state of a conversation using its `thread_id`.  
*Study:* LangGraph persistence docs

**Long-term memory (RAG)**  
Facts, preferences, and past exchanges stored in a vector database (ChromaDB). When a new query arrives, the agent retrieves the most semantically similar stored memories and injects them into the prompt as context.  
*Study:* LangChain RAG conceptual guide + ChromaDB quickstart

**RAG (Retrieval-Augmented Generation)**  
A pattern where you retrieve relevant documents or facts from a store and inject them into the LLM's context before generation. The model then uses this retrieved context to give more accurate, personalised answers.  
*Study:* Lewis et al. 2020 "Retrieval-Augmented Generation" paper + LangChain RAG tutorial

**Vector database**  
A database that stores text as high-dimensional numerical vectors (embeddings) rather than plain text. Queries are also converted to vectors, and the database returns the stored items whose vectors are most similar (cosine similarity). ChromaDB is a local, file-backed vector DB.  
*Study:* ChromaDB docs + "What is a vector database" — Pinecone blog (good conceptual explainer)

**Embedding**  
A numerical representation of a piece of text as a list of ~768 or ~1536 floating-point numbers. Semantically similar texts produce vectors that are close together in this high-dimensional space. You use `nomic-embed-text` (running via Ollama) to generate these locally.  
*Study:* Jay Alammar "The Illustrated Word2Vec" (intuition) + nomic-embed-text model card

**Cosine similarity**  
A measure of how similar two vectors are, regardless of their length. Used by ChromaDB to rank retrieved documents. Returns 1.0 for identical, 0.0 for unrelated. This is what "semantic search" means under the hood.  
*Study:* Any linear algebra resource — 3Blue1Brown "Essence of Linear Algebra" (YouTube)

**Context window**  
The maximum number of tokens (roughly words) a model can process in one call. Everything — the system prompt, conversation history, retrieved documents, and the current query — must fit inside this window.  
*Study:* Your chosen model's model card for exact context length

---

### Infrastructure concepts

**Ollama**  
A local model server that runs LLMs on your machine and exposes an OpenAI-compatible REST API at `http://localhost:11434`. LangChain's `ChatOpenAI` class can point to it directly.  
*Study:* ollama.com quickstart

**OpenAI-compatible API**  
A REST API that follows the same request/response schema as OpenAI's `/v1/chat/completions` endpoint. Ollama, LM Studio, and many other local servers implement this, meaning you can use the standard `openai` Python client with them.  
*Study:* OpenAI API reference → chat completions endpoint

**LangChain**  
The broader Python library ecosystem that LangGraph is part of. Provides LLM wrappers (`ChatOpenAI`), message types (`HumanMessage`, `SystemMessage`), tool definitions, and many integrations. LangGraph is LangChain's graph execution layer.  
*Study:* LangChain conceptual guide — start with "LangChain Expression Language (LCEL)"

**LangSmith**  
Observability and tracing platform for LangChain/LangGraph agents. Every node execution, LLM call, tool call, and state mutation is recorded as a trace. Can be self-hosted locally. Essential for debugging and evaluation.  
*Study:* LangSmith quickstart (the cloud version is free up to a limit; the self-hosted version is fully private)

**ChromaDB**  
A local, open-source vector database that persists to disk. You add documents (text chunks), it generates and stores embeddings, and you query it by semantic similarity.  
*Study:* ChromaDB docs → "Getting started" guide

**Chainlit**  
A Python framework for building chat interfaces. Has native support for async streaming, step indicators, and LangGraph/LangChain integration. Renders in a browser at `localhost:8000`.  
*Study:* Chainlit docs → "Integrations → LangGraph"

**Streamlit** *(alternative UI)*  
A Python framework for building data apps and simple UIs. Easier to learn than Chainlit but lacks native streaming — you have to implement it manually. Good if you want a dashboard alongside the chat.  
*Study:* Streamlit docs → "Build a chatbot"

**SQLite**  
A local, file-backed relational database built into Python's standard library. LangGraph's `SqliteSaver` checkpointer uses it to persist conversation state. No server required.  
*Study:* Python `sqlite3` module docs (understand the basics, you won't write SQL directly)

---

### Evaluation concepts

**LLM-as-judge**  
Using an LLM to score another LLM's output on dimensions like faithfulness, relevance, and helpfulness. Your local `qwen3:14b` can be both the assistant and the judge — no data leaves the machine.  
*Study:* Zheng et al. 2023 "Judging LLM-as-a-Judge with MT-Bench" paper

**RAGAS**  
An open-source Python library for evaluating RAG pipelines. Metrics include: faithfulness (does the answer match the retrieved context?), answer relevancy, context precision, and context recall.  
*Study:* ragas.io documentation

**Faithfulness**  
An evaluation metric: does the model's answer only contain claims that are supported by the retrieved context? A faithfulness score of 1.0 means no hallucination relative to the provided sources.  
*Study:* RAGAS docs → "Faithfulness" metric

**Hallucination**  
When a model states something confidently that is factually incorrect or not supported by its context. The primary failure mode you are evaluating against.  
*Study:* "Survey of Hallucination in Natural Language Generation" (Ji et al. 2023)

**Subjective evaluation**  
Human (in this case, you) rating the quality of agent responses on a scale (e.g., 1–5) along dimensions like helpfulness, tone, and accuracy. The ground truth that LLM-as-judge tries to approximate.  
*Study:* Chatbot Arena paper — "LMSYS Chatbot Arena: Benchmarking LLMs in the Wild"

---

## 6. Learning Resources (Prioritised)

Work through these in order. Each one unlocks understanding for the next phase.

### Before Phase 1
- [ ] **LangGraph quickstart** — `python.langchain.com/docs/langgraph` — build the minimal graph in ~30 mins
- [ ] **Python TypedDict** — PEP 589 — 15 mins
- [ ] **Ollama quickstart** — pull and run `qwen3:14b` from the terminal before any code

### Before Phase 2
- [ ] **LangChain ChatOpenAI** — understand `invoke`, `stream`, and message types
- [ ] **LangGraph nodes and edges** — official "How-to guides → Nodes"
- [ ] **LangGraph state management** — "How-to guides → State"

### Before Phase 3
- [ ] **Chain-of-Thought paper** (Wei et al. 2022) — read the abstract and examples section
- [ ] **Qwen3 model card** — understand the `<think>` tag format
- [ ] **LangGraph conditional edges** — official how-to

### Before Phase 4
- [ ] **LangChain tools** — "How to create tools" guide
- [ ] **DuckDuckGo Search Python library** — PyPI page + examples
- [ ] **LangGraph tool node** — built-in `ToolNode` class docs

### Before Phase 5
- [ ] **LangGraph persistence / checkpointers** — official persistence guide
- [ ] **ChromaDB getting started** — chromadb.io docs
- [ ] **RAG conceptual guide** — LangChain RAG tutorial (Part 1 and 2)
- [ ] **Nomic Embed Text** — model card on Ollama hub

### Before Phase 6
- [ ] **Chainlit getting started** — chainlit.io docs
- [ ] **LangSmith quickstart** — smith.langchain.com
- [ ] **RAGAS documentation** — ragas.io

---

## 7. Project File Structure (Target)

This is what your project will look like at Phase 6. Build it incrementally — create each folder and file only when you need it.

```
local-assistant/
│
├── graph/
│   ├── __init__.py
│   ├── state.py          # AgentState TypedDict — Phase 1
│   └── graph.py          # StateGraph builder — grows each phase
│
├── nodes/
│   ├── __init__.py
│   ├── llm_node.py       # Core LLM inference — Phase 2
│   ├── thinking_node.py  # Chain-of-thought — Phase 3
│   ├── router_node.py    # Conditional edge logic — Phase 4
│   ├── search_node.py    # DuckDuckGo search — Phase 4
│   └── memory_node.py    # ChromaDB retrieve + persist — Phase 5
│
├── memory/
│   ├── __init__.py
│   └── vector_store.py   # ChromaDB client + query helpers — Phase 5
│
├── ui/
│   ├── __init__.py
│   └── app.py            # Chainlit interface — Phase 6
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py      # LLM-as-judge scorer — Phase 6
│   └── test_cases.json   # Your personal eval dataset
│
├── data/
│   ├── checkpoints.db    # SQLite conversation state (gitignored)
│   └── chroma_db/        # ChromaDB persisted vector store (gitignored)
│
├── .env                  # LANGCHAIN_* env vars (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 8. Your Design Decisions (Fill This In)

After working through Section 3, record your choices here. This becomes the specification for your Phase 1 code.

| Decision | Options | Your choice | Your reasoning |
|---|---|---|---|
| State schema | Minimal / Rich upfront / Nested | **B — Rich upfront** | No refactors; nodes decoupled; checkpointer persists rich context |
| Graph topology | Linear / Conditional fan-out / ReAct / Hierarchical | **B — Conditional fan-out** | Natural extension; fast path for simple queries; ReAct deferred to Phase 4 |
| Node granularity | Monolithic / Atomic / Functional grouping | **C — Atomic single-responsibility** | Unit-testable; easy to swap; essential for evaluation and profiling |
| LLM call strategy | Single call / Two-pass / Multi-model | **B — Two-pass: think then respond** | Quality over latency; thinking field is evaluation signal |
| Primary model | qwen3:14b / qwen3:30b-a3b / deepseek-r1:14b | **qwen3:14b** | Best instruction following for budget; ~11GB at Q5_K_M fits 24GB RAM; upgrade path to 30b-a3b clear |
| UI framework | Chainlit / Streamlit | *To decide* | |

---

## 9. Evaluation & Monitoring Strategy

Evaluation is not a Phase 6 feature — it is a mindset you adopt from Phase 2 onward. Start logging immediately.

### Metrics to track per phase

| Phase | What to measure | How |
|---|---|---|
| 2 | Response latency, response length | `time.time()` wrapper around `graph.invoke()` |
| 3 | Does `thinking` actually detect search-needed queries correctly? | Manual review of `state.thinking` |
| 4 | Search trigger rate, result relevance | Review `state.tool_calls` logs |
| 5 | Memory retrieval precision — are the right memories being fetched? | Manual review of `state.retrieved_docs` |
| 6 | LLM-as-judge scores, your personal ratings | RAGAS + custom eval runner |

### The evaluation loop (your continuous improvement cycle)

```
Use the agent → Notice a bad response → Inspect the trace in LangSmith
     ↓
Identify which node caused the failure
     ↓
Write a test case that reproduces the failure
     ↓
Fix the node (prompt, logic, or retrieval)
     ↓
Re-run eval suite → Confirm improvement → Update this doc
```

---

## 10. Changelog

Use this section to record decisions you changed and why. This is your learning journal.

| Date | Phase | What changed | Why |
|---|---|---|---|
| | | | |

---

*Last updated: Phase 1 planning — pre-implementation*  
*Hardware: MacBook Pro M4 · 24 GB unified memory*  
*Primary model: qwen3:14b at Q5_K_M via Ollama*
