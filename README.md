# Simple LangGraph Chatbot

A deliberately small chat application built to **learn LangGraph**, not to show off.

The entire "agent" is one node:

```
START ──▶ call_model (DeepSeek) ──▶ END
```

That is on purpose. Before you can reason about multi-node agents, tools, loops and human-in-the-loop
flows, you need to see what a graph, a state schema, a reducer, a checkpointer and a thread actually
do. Every one of those ideas is present here, in ~240 lines of readable Python.

## Features

- **One-node LangGraph workflow** — `START → call_model → END`, compiled with a SQLite checkpointer.
- **SQLite persistence** — every conversation is stored on disk; restart the app and nothing is lost.
- **Threads** — a collapsible sidebar with *New chat*, the list of stored conversations, and a 🗑 delete
  action per thread.
- **Reopening a thread** loads its full message history from the database.
- **Token streaming** — the assistant's reply appears word by word via `stream_mode="messages"`.
- **Streamlit UI** — native `st.chat_message` / `st.chat_input` chat elements, no custom CSS or JS.
- **8 focused tests** — they prove the reducer keeps history, that threads are isolated, that streaming
  works, and that deleting a thread clears both stores.

## How it works

```
┌──────────────────────── Streamlit (src/chatbot/ui.py) ────────────────────────┐
│  sidebar: New chat · thread list · 🗑         chat: transcript · chat_input     │
└───────────────┬────────────────────────────────────────────▲──────────────────┘
                │ HumanMessage                                │ tokens
                ▼                                             │
┌────────────────────── LangGraph app (src/chatbot/graph.py) ──┴─────────────────┐
│   START ──▶ call_model ──▶ END                                                │
│                 │  llm.invoke([SystemMessage, *state["messages"]])            │
└─────────────────┼─────────────────────────────────────────────────────────────┘
                  │                                    state saved after every step
                  ▼                                                    │
        ChatDeepSeek (api.deepseek.com)                                ▼
                                                     ┌──────────────────────────┐
                                                     │ data/chatbot.db          │
                                                     │  threads                 │ ← this app
                                                     │  checkpoints, writes     │ ← LangGraph
                                                     └──────────────────────────┘
```

Three layers, each with one job:

| Layer | File | Responsibility |
|---|---|---|
| UI | `app.py`, `src/chatbot/ui.py` | Draw the sidebar and chat, stream tokens, decide which thread is active. Knows nothing about prompts or SQL. |
| Workflow | `src/chatbot/state.py`, `src/chatbot/graph.py` | Define the state schema, the node, the edges, and the LLM call. |
| Storage | `src/chatbot/db.py` + LangGraph's checkpointer | Two kinds of data: LangGraph's *checkpoints* (conversation state) and this app's *threads* table (titles + ordering). |

### Project structure

```
simple-chatbot-01/
├── app.py                  # Streamlit entry point — just calls ui.render()
├── pyproject.toml          # dependencies, Python version, pytest + ruff config
├── uv.lock                 # exact versions, created by uv
├── .env.example            # copy to .env and add your API key
├── data/                   # SQLite database is created here (git-ignored)
├── src/chatbot/
│   ├── config.py           # env settings, file paths, system prompt
│   ├── state.py            # ChatState — this is where the reducer lives
│   ├── graph.py            # the graph: node, edges, LLM, checkpointer
│   ├── db.py               # the `threads` table (create/list/delete)
│   └── ui.py               # sidebar, transcript, streaming send
└── tests/
    ├── test_graph.py       # reducer, thread isolation, streaming, delete
    └── test_db.py          # threads table behaviour
```

## The LangGraph / LangChain features used

| Feature | Where | What it does for us |
|---|---|---|
| `StateGraph` | `graph.py` | The graph builder. You declare nodes and edges on it. |
| **State schema** (`TypedDict`) | `state.py` | Declares *what memory the graph has*. Here: a list of messages. |
| **Reducer** (`add_messages`) | `state.py` | Tells LangGraph how to *combine* an update with the existing value instead of replacing it. Explained at length [below](#why-reducers-are-needed). |
| Nodes | `graph.py` → `call_model` | A plain Python function that receives the state and returns a **partial update**. |
| Edges (`START`, `END`) | `graph.py` | `START` is the entry point, `END` terminates. Our only node sits between them. |
| `compile(checkpointer=...)` | `graph.py` | Turns the builder into a runnable graph **and switches persistence on**. |
| Checkpointer (`SqliteSaver`) | `graph.py` → `get_checkpointer` | Saves the state of every step into SQLite. This is what makes conversations survive restarts. |
| `thread_id` / `config` | `graph.py` → `thread_config` | A thread is one conversation. The checkpointer keys everything by `thread_id`, so it doubles as the chat's primary key. |
| `stream_mode="messages"` | `ui.py` → `stream_reply` | Streams LLM tokens as `(message_chunk, metadata)` tuples while the graph runs. |
| `get_state(config)` | `graph.py` → `load_history` | Reads the last saved state of a thread — how "open an old chat" works. |
| `ChatDeepSeek` (`BaseChatModel`) | `graph.py` → `get_llm` | LangChain's chat-model interface for DeepSeek. Any other `BaseChatModel` is a drop-in replacement (the tests use a fake one). |
| `st.cache_resource` | `ui.py` → `load_graph`, `load_checkpointer` | Builds the graph and opens the SQLite connection once per server process, not on every Streamlit rerun. |

### Why reducers are needed

LangGraph state is not a bag of variables you mutate. A node **returns a partial update**, and LangGraph
merges that update into the current state. "How to merge" is decided per key, by that key's **reducer**.

The default reducer is *last write wins*: whatever the update contains replaces the old value.

That is fine for a value like `summary: str`, and fatal for `messages: list`.

Our state declares the merge strategy explicitly:

```python
# src/chatbot/state.py
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` is a function `(left: list, right: list) -> list` that **appends** instead of replacing.

`MessagesState` from `langgraph.graph` is the built-in shortcut for exactly this field —
it is written out by hand here so you can see the reducer instead of importing it.

#### What happens without a reducer

Say the state declared `messages: list[AnyMessage]` (no `Annotated`), and you chat twice on one thread:

| Step | What is sent | State after the step (no reducer) |
|---|---|---|
| turn 1 input | `{"messages": [Human("hi")]}` | `[Human("hi")]` |
| turn 1 node | returns `{"messages": [AI("hello!")]}` | **`[AI("hello!")]`** — the human message was overwritten |
| turn 2 input | `{"messages": [Human("how are you?")]}` | **`[Human("how are you?")]`** |
| turn 2 LLM prompt | `[System, Human("how are you?")]` | 🐛 the bot has no idea what was said before |

Nothing raised an error. The chatbot simply forgets everything, every turn — the classic bug.

#### What happens with `add_messages`

| Step | What is sent | State after the step (with reducer) |
|---|---|---|
| turn 1 input | `{"messages": [Human("hi")]}` | `[Human("hi")]` |
| turn 1 node | returns `{"messages": [AI("hello!")]}` | `[Human("hi"), AI("hello!")]` |
| turn 2 input | `{"messages": [Human("how are you?")]}` | `[Human("hi"), AI("hello!"), Human("how are you?")]` |
| turn 2 LLM prompt | `[System, Human("hi"), AI("hello!"), Human("how are you?")]` | ✅ the model sees the whole conversation |

Note *where* the reducer fires: not only on the node's return value, but also when the **graph input**
is applied to the already-checkpointed state. Because the checkpointer restores turn 1 before turn 2
runs, the reducer is what appends the new `HumanMessage` to the restored list.

`tests/test_graph.py::test_messages_accumulate_across_invocations_on_one_thread` asserts both halves of
this: the stored history *and* the exact list of messages handed to the model on the second turn.

#### "It is a one-node graph — do I still need a reducer?"

Yes, and it is worth being precise about why. The number of **supersteps** in a graph has nothing to do
with it. What matters is the number of **invocations** of the same thread. Every message the user sends
is a new invocation against the same checkpoint, so `messages` accumulates across invocations, and
accumulating keys need a reducer.

#### Two more things `add_messages` gives you for free

- **Upsert by id** — messages are keyed by their `id`, so returning a message with an existing id updates
  it in place instead of duplicating it.
- **Deletion** — return a `RemoveMessage(id=...)` and the reducer removes that message. This is how
  "edit the conversation" / "clear history" features are built later on.

## What happens when you send a message

1. `ui.render_chat` has already painted the stored transcript by calling
   `load_history(graph, thread_id)` → `graph.get_state(config).values["messages"]`.
2. You type into `st.chat_input`. Streamlit reruns the script; `prompt` is not empty.
3. `db.ensure_thread(thread_id, prompt)` inserts the row in `threads` on the first message (the title is
   the first 40 characters of that message) or just bumps `updated_at` afterwards.
4. `stream_reply` calls `graph.stream({"messages": [HumanMessage(prompt)]}, config, stream_mode="messages")`.
5. LangGraph loads the thread's checkpoint, the `add_messages` reducer appends the human message,
   and `call_model` runs.
6. `call_model` builds `[SystemMessage, *state["messages"]]` and calls `llm.invoke(...)`, then returns
   `{"messages": [AIMessage(...)]}` — again merged by the reducer.
7. `stream_mode="messages"` yields the model's chunks as they arrive; `stream_reply` keeps the text and
   `st.write_stream` prints it into the assistant bubble.
8. The checkpointer writes the new state before the step finishes, so the reply is already persisted.
9. `st.rerun()` repaints the page from the database, which also refreshes the sidebar title and order.

## Streaming

`stream_mode="messages"` yields tuples of `(message_chunk, metadata)`:

```python
for chunk, metadata in graph.stream(inputs, config, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk):
        print(chunk.text, end="")
```

`metadata["langgraph_node"]` tells you which node produced the chunk — handy once you have several LLM
calls and want to stream only one of them. `chunk.text` flattens the message content to plain text.

## SQLite database schema

Everything lives in **one** file (`data/chatbot.db` by default, override with `CHATBOT_DB_PATH`), which
holds two families of tables:

### 1. `threads` — owned by this app (`src/chatbot/db.py`)

```sql
CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,          -- the LangGraph thread_id (a UUID)
    title      TEXT NOT NULL,             -- first user message, truncated to 40 chars
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads (updated_at DESC);
```

`created_at` / `updated_at` are written by the app as ISO-8601 UTC strings
(`2026-11-14T09:41:12.345678+00:00`), which also sort correctly as text; `updated_at` drives the
sidebar ordering.

A checkpointer stores *state*, not titles, so the sidebar needs this small table. It is also the place
to add `user_id`, `archived`, `pinned`, … later.

### 2. `checkpoints` and `writes` — owned by LangGraph (`SqliteSaver.setup()`)

Created automatically the first time the app runs:

```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,                      -- the serialised state (messages live here)
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

- One `checkpoints` row = one saved state of a thread. LangGraph keeps them all, so you can inspect or
  replay history (`graph.get_state_history(config)`).
- `writes` holds the intermediate results of each node run, which is what resumes a graph cleanly.
- `thread_id` is the join key between your `threads` table and LangGraph's tables. Deleting a thread
  therefore means: `checkpointer.delete_thread(id)` **and** `DELETE FROM threads WHERE id = ?` — both
  are wrapped in `ui.delete_thread`.

### Poking at the database

Any SQLite viewer works (DB Browser for SQLite, the `sqlite3` CLI, or the SQLite extension for VS Code):

```sql
SELECT id, title, updated_at FROM threads ORDER BY updated_at DESC;
SELECT thread_id, COUNT(*) AS saved_states FROM checkpoints GROUP BY thread_id;
```

The checkpoint payloads are BLOBs, so read them through LangGraph instead:

```python
from chatbot.graph import build_graph, get_checkpointer, get_llm, load_history

graph = build_graph(get_llm(), get_checkpointer())
for message in load_history(graph, "<paste-a-thread-id>"):
    print(message.type, "→", message.text)
```

## Getting started

### Prerequisites

- **Python 3.11+** (developed on 3.14.7).
- **[uv](https://docs.astral.sh/uv/)** — install it once:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A **DeepSeek API key** from <https://platform.deepseek.com/api_keys>.

### 1. Install dependencies

```powershell
uv sync
```

This creates `.venv/`, installs everything from `uv.lock`, and installs this project in editable mode
(so `import chatbot` works from anywhere in the project).

### 2. Configure the environment

```powershell
Copy-Item .env.example .env      # macOS/Linux: cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | ✅ | — | Your DeepSeek key. Read automatically by `ChatDeepSeek`. |
| `DEEPSEEK_MODEL` | — | `deepseek-flash` | `deepseek-flash` (DeepSeek-V4.1-Flash) or `deepseek-v4-pro` (bigger, slower). Both support thinking mode. |
| `CHATBOT_DB_PATH` | — | `data/chatbot.db` | Where the SQLite file lives. A relative path is resolved from the project root. |

`.env` is git-ignored — never commit your key.

> **A note on thinking mode:** DeepSeek-V4.1-Flash answers with thinking enabled by default, which delays
> the first streamed token. `get_llm()` therefore passes `extra_body={"thinking": {"type": "disabled"}}`.
> Drop that argument if you want the model to reason before answering (this app streams only the final
> `content`, never the reasoning text).

### 3. Run the app

```powershell
uv run streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>) and opens your browser. Type a message
and the reply streams in. Click **➕ New chat**, chat again, then switch between the two threads in the
sidebar — both conversations are read back from SQLite.

### 4. Run the tests

```powershell
uv run pytest
```

The tests use a fake chat model, so they need no API key and make no network calls.

### 5. Lint and format

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. It is configured
in `pyproject.toml` (`[tool.ruff]`): line length 100, and the `E`, `W`, `F`, `I`, `UP` and `B` rule sets.

```powershell
uv run ruff check .            # report problems
uv run ruff check --fix .      # fix what ruff can fix automatically (unused imports, import order, …)
uv run ruff format .           # apply formatting
uv run ruff format --check .   # verify formatting without changing files
```

What each rule set catches: `F` undefined/unused names, `E`/`W` style problems (long lines, whitespace),
`I` import ordering and grouping, `UP` outdated syntax that newer Python replaces (e.g. `datetime.UTC`),
`B` common bug patterns. Run `uv run ruff check . --statistics` to see the rule codes in use, or
`uv run ruff rule <CODE>` (e.g. `uv run ruff rule E501`) to read what a specific rule means.

### 6. Type check

Static types are checked with [ty](https://docs.astral.sh/ty/), Astral's type checker. It needs no
configuration: it discovers the project `.venv` and reads `requires-python` from `pyproject.toml`.

```powershell
uv run ty check                        # check the whole project
uv run ty check --output-format concise  # one line per problem
uv run ty check --watch                # re-check while you edit
```

There is exactly one suppression in the codebase — `StateGraph(ChatState)  # ty: ignore[invalid-argument-type]`
in `src/chatbot/graph.py` — because ty 0.0.104 does not yet match `TypedDict` state classes against
LangGraph's `StateLike` protocol (LangGraph's own `MessagesState` fails the same way). Drop the comment
once ty supports it.

### 7. Start over with an empty history

```powershell
Remove-Item data\chatbot.db*       # macOS/Linux: rm data/chatbot.db*
```

The tables are recreated on the next launch.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `DEEPSEEK_API_KEY is not set` in the UI | Create `.env` from `.env.example`, add the key, restart Streamlit. |
| `AuthenticationError` / HTTP 401 | The key is wrong, revoked, or has no credit. |
| `sqlite3.OperationalError: database is locked` | Something else is writing the DB — close other SQLite clients, or delete the file to start fresh. |
| Port already in use | `uv run streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError: chatbot` | Run `uv sync` from the project root, and start the app with `uv run`. |

## Where to go next

The point of this project is to grow the graph. Small, safe next steps, in order:

1. **Add a second node** — e.g. `call_model → save_summary → END`, or a router node with a conditional
   edge (`builder.add_conditional_edges`). The state schema is the only thing you have to extend, and you
   will immediately feel when a field needs a reducer and when plain last-write-wins is correct.
2. **Give it tools** — a node that binds `llm.bind_tools([...])`, plus a `ToolNode` and a conditional
   edge back to the model. This is the classic ReAct loop, and the persistence you already have keeps
   working.
3. **Stream more** — try `stream_mode="updates"` (see each node's output) or `["messages", "updates"]`
   to watch the graph and the tokens together.
4. **Inspect history** — `graph.get_state_history(config)` returns every checkpoint of a thread, which is
   the basis for undo, branching, or a "time travel" debugger.
5. **Multi-user** — add a `user_id` column to `threads` and filter the sidebar by the logged-in user.
