# Day 11 — Local RAG & Semantic Retrieval

## Goal

Day 10 finished the web research pipeline:

```text
search → fetch → extract → chunk → context → LLM
```

Day 11 introduces **embeddings and a vector store**, but with one important architectural decision:

> Do not use RAG for the internet yet. Use RAG to give Jarvis its own persistent knowledge.

A typical RAG pipeline chunks documents, embeds those chunks, stores the vectors, embeds the user's query, retrieves similar chunks, and passes them to the LLM as context. citeturn0search3turn0search9

## 1. Choose the first knowledge source

Start with a small local directory:

```text
knowledge/
├── notes/
├── docs/
└── projects/
```

For V1, support only Markdown and plain text.

## 2. Create a document ingestion pipeline

```text
Document
   ↓
Read
   ↓
Clean
   ↓
Chunk
   ↓
Embed
   ↓
Store
```

## 3. Define a `Document`

```python
@dataclass
class Document:
    id: str
    path: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
```

## 4. Generalize the Day 10 chunk model

Reuse the same abstraction for web content and local documents:

```python
@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
```

## 5. Add an embedding model

Embeddings turn text into numerical vectors representing semantic information.

For a free/local-first Jarvis, evaluate a small local embedding model and hide it behind:

```python
class Embedder:
    def embed(self, text: str) -> list[float]:
        ...
```

Do not scatter model-specific code throughout the application.

## 6. Understand document vs query embeddings

Document:

```text
"Redis can implement rate limiting..."
        ↓
document embedding
```

Query:

```text
"How can I restrict API request frequency?"
        ↓
query embedding
```

Then:

```text
query vector
     ↓
vector search
     ↓
nearest document vectors
```

Use the embedding model's recommended query/document modes if it provides them.

## 7. Choose a vector store

For V1, don't deploy a distributed vector database.

### Recommended: PostgreSQL + pgvector

```text
PostgreSQL
 ├── documents
 ├── chunks
 └── embeddings
```

pgvector adds vector types, distance operators, and approximate indexes such as HNSW to PostgreSQL. citeturn0search2

Minimal schema:

```sql
CREATE EXTENSION vector;

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(EMBEDDING_DIM),
    metadata JSONB
);
```

The embedding dimension must match your model.

For a small corpus, exact scans can be perfectly adequate; pgvector notes that a table scan may be faster when the table is small. citeturn0search2

## 8. Store metadata alongside vectors

Store:

```text
chunk_id
document_id
content
embedding
title
path
position
created_at
updated_at
```

The vector tells you what is semantically similar. Metadata tells you where it came from.

## 9. Implement ingestion

Create:

```bash
python -m jarvis.ingest knowledge/
```

Flow:

```text
knowledge/
     ↓
find documents
     ↓
read
     ↓
chunk
     ↓
embed
     ↓
store
```

Example output:

```text
Found 4 documents

golang.md
  12 chunks

redis.md
  8 chunks

system-design.md
  17 chunks

jarvis.md
  6 chunks

Total:
43 chunks indexed
```

## 10. Make ingestion incremental

Don't re-embed everything every time.

```text
Document
 ↓
calculate hash
 ↓
hash unchanged?
 ├── yes → skip
 └── no  → re-index
```

Store:

```text
document_hash
```

For example:

```python
content_hash = sha256(content.encode()).hexdigest()
```

## 11. Delete stale chunks

When a document changes:

```text
old chunks
    ↓
delete by document_id
    ↓
re-chunk
    ↓
re-embed
    ↓
insert
```

This prevents stale information from remaining searchable.

## 12. Implement semantic search

Create:

```python
search_knowledge(query)
```

Flow:

```text
User question
      ↓
Embedding
      ↓
Vector search
      ↓
Top K chunks
```

Start with:

```text
TOP_K = 5
```

A standard vector retriever embeds the query and compares it with stored vectors using a similarity/distance metric. citeturn0search3

## 13. Return similarity information

Example:

```text
Chunk                         Score

Redis rate limiting             0.91
Gubernator implementation       0.87
Kafka consumer groups           0.42
Python decorators               0.18
```

Don't blindly choose a universal threshold such as `0.70`. Measure retrieval quality first because the useful threshold depends on the embedding model, metric, corpus, and chunking strategy.

## 14. Test semantic search manually

Put this in your knowledge base:

```text
Redis can implement rate limiting using token bucket algorithms.
```

Query:

```text
How can I restrict API request frequency?
```

The Redis chunk should appear near the top even though the query doesn't contain the exact words `Redis`, `token`, or `bucket`.

## 15. Compare keyword search vs vector search

Example:

```text
"prevent duplicate payment"
```

Keyword search might find:

```text
duplicate payment
```

Semantic search may additionally find:

```text
idempotency keys
exactly-once processing
transaction deduplication
```

This comparison demonstrates what embeddings add.

## 16. Don't build hybrid retrieval yet

Eventually:

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Keyword Search     Vector Search
          │                 │
          └────────┬────────┘
                   ▼
              Rank Fusion
                   │
                   ▼
             Final Results
```

Hybrid retrieval can combine lexical and semantic retrieval. citeturn0search1

But Day 11 should remain:

> **Vector search only.**

## 17. Connect RAG to the LLM

```python
async def answer_with_knowledge(query):
    chunks = await search_knowledge(query)

    context = build_context(chunks)

    return await llm.answer(
        query=query,
        context=context,
    )
```

Final flow:

```text
User
 ↓
LLM / Router
 ↓
knowledge_search()
 ↓
Embedding
 ↓
Vector DB
 ↓
Relevant chunks
 ↓
Context Builder
 ↓
LLM
 ↓
Answer
```

This is the standard RAG pattern: retrieve relevant context first, then provide it to the generation model. citeturn0search9

## 18. Add `search_knowledge` as an LLM capability

Eventually:

```json
{
  "name": "search_knowledge",
  "description": "Search Jarvis's private knowledge base",
  "parameters": {
    "query": {
      "type": "string"
    }
  }
}
```

Jarvis now has two information sources:

```text
                    Jarvis
                       │
              ┌────────┴────────┐
              │                 │
        search_web()      search_knowledge()
              │                 │
           SearXNG           Vector DB
              │                 │
          Internet          Your data
```

## 19. Teach the router when to use each

### Web search

Use for:

```text
latest
today
current
news
price
recent release
weather
live information
```

### Private knowledge

Use for:

```text
my notes
my project
my documentation
what I wrote
our architecture
my code
previous research
```

### Both

For:

> What did I write about rate limiting, and is that still the recommended approach?

Use:

```text
search_knowledge()
        +
search_web()
        ↓
combine
        ↓
LLM
```

Standard RAG is appropriate for fixed retrieval flows; more dynamic source selection can later be handled by an agent/router. citeturn0search8

## 20. Preserve provenance

A RAG result should retain:

```json
{
  "chunk_id": "chunk_001",
  "document_id": "doc_redis",
  "title": "Redis Notes",
  "path": "knowledge/redis.md",
  "content": "Redis can...",
  "score": 0.91
}
```

Then Jarvis can say:

> According to your Redis notes...

and optionally show the source path.

## 21. Protect private knowledge

Keep private knowledge as a separate data path:

```text
PRIVATE KNOWLEDGE
      │
      ▼
 Local retrieval
      │
      ▼
 Relevant chunks
      │
      ▼
 LLM
```

Don't send the entire knowledge base to an external model. Only provide the retrieved chunks needed for the answer.

## 22. Add document ownership to the schema

Even for a single-user V1, include:

```text
owner_id
```

Eventually:

```text
documents
 ├── owner_id
 ├── visibility
 └── ...
```

You don't need multi-user functionality yet.

## 23. Create an evaluation dataset

Create:

```text
tests/rag/
```

Example questions:

```text
Q1:
How does our payment service handle duplicate requests?

Expected:
payment.md

Q2:
What did I write about Kafka consumer retries?

Expected:
kafka.md

Q3:
What is our Redis caching strategy?

Expected:
architecture.md
redis.md

Q4:
What does my documentation say about authentication?

Expected:
auth.md
```

Record:

```text
query
expected documents
expected chunks
actual chunks
score
```

## 24. Measure retrieval quality

Track:

```text
Top-1 accuracy
Top-3 recall
Top-5 recall
```

Example:

```text
10 test questions

Correct document in top 1:
7

Correct document in top 3:
9

Correct document in top 5:
10
```

This gives you a baseline for evaluating later improvements.

## 25. Day 11 architecture

```text
                         ┌──────────────┐
                         │     User     │
                         └──────┬───────┘
                                │
                                ▼
                               STT
                                │
                                ▼
                              LLM
                                │
                   ┌────────────┴────────────┐
                   │                         │
              search_web()           search_knowledge()
                   │                         │
                   ▼                         ▼
                SearXNG                 Embed Query
                   │                         │
                   ▼                         ▼
              Web Research             Vector DB
                   │                         │
                   │                    Top K Chunks
                   │                         │
                   └────────────┬────────────┘
                                ▼
                         Context Builder
                                │
                                ▼
                               LLM
                                │
                                ▼
                               TTS
```

Knowledge ingestion:

```text
              knowledge/
                   │
                   ▼
              Documents
                   │
                   ▼
                Chunking
                   │
                   ▼
              Embedding
                   │
                   ▼
             Vector Store
                   │
                   ▼
             Searchable KB
```

# Day 11 Definition of Done

- [ ] `Document` model implemented.
- [ ] Generic `Chunk` model created/reused.
- [ ] Local Markdown/text ingestion implemented.
- [ ] Local embedding model evaluated.
- [ ] `Embedder` abstraction implemented.
- [ ] Vector store selected.
- [ ] Vector store running locally.
- [ ] Document embeddings stored.
- [ ] Metadata stored alongside vectors.
- [ ] Incremental document ingestion implemented.
- [ ] Document hashing implemented.
- [ ] Stale chunks removed when documents change.
- [ ] `search_knowledge()` implemented.
- [ ] Query embeddings implemented.
- [ ] Top-K vector search implemented.
- [ ] Similarity scores returned.
- [ ] Semantic search manually tested.
- [ ] Keyword vs semantic retrieval compared.
- [ ] RAG context builder connected to LLM.
- [ ] `search_knowledge` exposed as an LLM capability.
- [ ] Web search and private knowledge kept separate.
- [ ] Provenance preserved.
- [ ] Basic privacy boundary implemented.
- [ ] RAG evaluation dataset created.
- [ ] Retrieval metrics measured.
- [ ] No hybrid search yet.
- [ ] No reranker yet.
- [ ] No complicated agent loops yet.

# Day 11 Milestone

Jarvis should handle:

> **"What did I write about rate limiting?"**

```text
User
 ↓
STT
 ↓
LLM
 ↓
search_knowledge()
 ↓
Embed:
"What did I write about rate limiting?"
 ↓
Vector DB
 ↓
Top 5 chunks
 ↓
Context Builder
 ↓
LLM
 ↓
"According to your notes..."
 ↓
TTS
```

And:

> **"What's the latest information about rate limiting?"**

should use:

```text
search_web()
 ↓
SearXNG
 ↓
Current sources
 ↓
LLM
```

The interesting case:

> **"What did I write about rate limiting, and is that still the recommended approach?"**

```text
                    User
                      │
                      ▼
                     LLM
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
   search_knowledge()      search_web()
            │                   │
            ▼                   ▼
        Your notes          Current web
            │                   │
            └─────────┬─────────┘
                      ▼
                   Context
                      │
                      ▼
                     LLM
                      │
                      ▼
                  Comparison
```

## Important architectural decision

At the end of Day 11, Jarvis should have three distinct layers:

```text
                    ┌───────────────┐
                    │      LLM      │
                    │ General World │
                    │   Knowledge   │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       ┌─────────────┐             ┌──────────────┐
       │ Web Search  │             │ Private RAG  │
       │             │             │              │
       │ Current     │             │ Your data    │
       │ information │             │ Your notes   │
       │             │             │ Your docs    │
       └─────────────┘             └──────────────┘
```

Don't mix these together prematurely.

- LLM pretrained knowledge → general questions
- Web search → freshness
- RAG → information specific to you

## What comes after Day 11?

Don't immediately add every RAG optimization.

A sensible progression is:

```text
Day 11
Local documents
    ↓
Embeddings
    ↓
Vector DB
    ↓
Semantic retrieval
    ↓
LLM
```

Then later:

```text
Hybrid search
    ↓
Reranking
    ↓
Query rewriting
    ↓
Conversation-aware retrieval
    ↓
Agentic retrieval
```

A current open-source RAG architecture demonstrates this kind of progression with query rewriting, hybrid dense/BM25 retrieval, deduplication, optional reranking, prompt construction, and generation. citeturn0search1

For Jarvis, **Day 11 should stop at basic semantic retrieval**. Get it working and measurable before adding sophisticated retrieval machinery.

## References

- AWS Prescriptive Guidance — RAG retrieval architecture. citeturn0search3turn0search9
- pgvector — PostgreSQL vector similarity search and indexing. citeturn0search2
- Cloud.gov — practical pgvector RAG example using Markdown documents. citeturn0search0
- Microsoft Azure Architecture Center — standard vs agentic RAG. citeturn0search8
