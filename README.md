# 📚 RAG Study Buddy — LLM Course Notes Assistant

*A Retrieval-Augmented Generation (RAG) CLI App using Gemini + Local Embeddings*

This application is a lightweight, fully local + API-powered **RAG (Retrieval-Augmented Generation)** system.

It allows a student to ask questions about  **pre-uploaded PDF notes** , and receive answers grounded *only* in those documents.

This project satisfies all requirements of the  **LLM App Assignment** : core feature, RAG enhancement, safety guardrails, telemetry, offline evaluation, reproducibility, and UX polish.

---

# Features

### Architecture Diagram

```
User Question
      |
      v
+-------------------------+
| Safety Layer            |
| - Length guard          |
| - Prompt injection check|
+-------------------------+
      |
      v
+-------------------------+
| Retrieval (RAG)         |
| - embed query           |
| - cosine similarity     |
| - top-k chunks          |
+-------------------------+
      |
      v
+-------------------------+
| Prompt Builder          |
+-------------------------+
      |
      v
+-------------------------+
| Gemini LLM              |
+-------------------------+
      |
      v
Grounded Answer Returned

```

### ✅ Core Feature

* CLI app where a user asks: **“What do my documents say about X?”**
* The app retrieves the most relevant text chunks from embedded course PDFs and uses Gemini to produce an answer.

### ✅ Enhancement: **RAG**

* Local embeddings using **SentenceTransformers (all-MiniLM-L6-v2)**
* Cosine similarity search
* Context-injected prompts to Gemini for grounded answers
* Full index built once and cached

### ✅ Safety & Robustness

* System prompt with explicit **DO/DON’T rules**
* **Input length guard** (questions > 2000 chars are rejected)
* **Prompt-injection detection** for phrases like: *“ignore previous instructions”, “reveal your system prompt”*
* **Error fallback message** for any LLM/API failure
* If an answer is not supported by the documents, the model must say: **“I don’t know based on the docs.”**

### ✅ Telemetry

Per-request JSON logs stored at:

<pre class="overflow-visible!" data-start="1710" data-end="1735"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>logs/requests.log
</span></span></code></div></div></pre>

Each entry contains:

* timestamp
* pathway (`rag`, `blocked_prompt_injection`, `too_long`, `error`, `cache_hit`)
* latency (ms)
* rough token estimates
* model name

### ✅ Offline Evaluation

* `tests/tests.json` containing ≥ 15 test cases
* `app/offline_eval.py` runs all tests and prints a **pass rate**
* Covers length guard, injection guard, unknown questions, repeated query (cache), and RAG queries

### ✅ Reproducibility

Project includes:

* `README.md` (this file)
* `requirements.txt`
* `.env.example`
* Seed PDFs in `data/docs`
* One-command run script: `python -m app.rag_cli`
* Offline eval script: `python -m app.offline_eval`
* Deterministic embeddings index at `embeddings/index.pkl`

### ⭐ Bonus Implemented

* Cached embedding model (faster retrieval)
* Response caching (identical questions → instant answers)
* Rich CLI formatting (colored prompts, soft wrapping)

# Project Structure

```

.
├── app
│   ├── build_index.py         # Builds embedding index from PDFs
│   ├── rag_core.py            # RAG pipeline, LLM calls, safety, telemetry
│   ├── rag_cli.py             # Interactive CLI app
│   └── offline_eval.py        # Offline test runner
│
├── data
│   └── docs/                  # Seed PDF files (user-provided)
│
├── embeddings/
│   └── index.pkl              # Generated embedding index
│
├── logs/
│   └── requests.log           # Telemetry logs
│
├── tests
│   └── tests.json             # ≥15 test cases
│
├── requirements.txt
├── .env.example
└── README.md
```

# ⚙️ Installation & Setup

## 1. Clone the repository

```
git clone <your-repo-url>
cd <repo-folder>
```

## 2. Install requirements

<pre class="overflow-visible!" data-start="3603" data-end="3646"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt</span></span></code></div></div></pre>

## 3. Add your Gemini API key (done)

Create a `.env` file based on `.env.example`:

```
GEMINI_API_KEY=your-key-here or
copy .env.example .env
```

## 4. Add documents(done)

Place your `.pdf`, `.txt`, or `.md` files into: data/docs folder

## 5. Build the Embedding Index (One-Time) (done)

Before first run:

```
python -m app.build_index

```

## 6. Running the App

```
python -m app.rag_cli
```

You will see:

![1764709560161](image/README/1764709560161.png)

## 7. Run Offline Evaluation

```
python -m app.offline_eval
```

This will:

* Load `tests/tests.json`
* Run each question through the RAG pipeline
* Compare against expected substrings
* Print PASS/FAIL for each test
* Print an overall **pass rate (%)**

  ![1764709646575](image/README/1764709646575.png)

  ![1764709675375](image/README/1764709675375.png)
