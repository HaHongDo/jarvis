# Day 12 — Hybrid Retrieval & Better RAG Search

## Goal

Day 11 established basic RAG:

```text
Documents
   ↓
Chunk
   ↓
Embedding
   ↓
pgvector
   ↓
Semantic search
   ↓
LLM
```

Day 12 improves retrieval rather than adding another major feature.

The problem with pure vector search is that semantic similarity is not always enough. If the user asks:

> "What is the `authz_version` column?"

You want an exact match for `authz_version`. Semantic search may instead return documents about authorization generally.

The solution is:

> **Hybrid search = keyword search + vector search + rank fusion**

pgvector officially supports combining PostgreSQL full-text search with vector search and recommends Reciprocal Rank Fusion (RRF) or a cross-encoder for combining results.

---

## 1. Day 12 architecture

Change:

```text
                    Query
                      │
                      ▼
                Vector Search
                      │
                      ▼
                   Top K
```

into:

```text
                    Query
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Keyword Search     Vector Search
             │                 │
             ▼                 ▼
       Keyword Results     Vector Results
             │                 │
             └────────┬────────┘
                      ▼
                 Rank Fusion
                      │
                      ▼
                 Final Results
                      │
                      ▼
                     LLM
```

| Search | Good at |
|---|---|
| Keyword | Exact names, identifiers, terminology |
| Vector | Meaning, paraphrases, concepts |
| Hybrid | Both |

---

## 2. Add PostgreSQL full-text search

Since Jarvis is already using PostgreSQL + pgvector, don't introduce Elasticsearch.

Add a searchable `tsvector` representation:

```sql
ALTER TABLE chunks
ADD COLUMN textsearch tsvector;
```

Populate it from chunk content.

Then query:

```sql
SELECT *
FROM chunks
WHERE textsearch @@ plainto_tsquery('rate limiting');
```

PostgreSQL full-text search can rank matching documents using `ts_rank_cd`.

---

## 3. Add a GIN index

```sql
CREATE INDEX chunks_textsearch_idx
ON chunks
USING GIN(textsearch);
```

Your chunks table now has two retrieval indexes:

```text
chunks
 ├── GIN
 │    └── keyword search
 │
 └── HNSW / vector index
      └── semantic search
```

For a small dataset, exact vector search may still be enough. Don't spend Day 12 tuning vector indexes. pgvector's default exact search provides perfect recall; approximate HNSW/IVFFlat indexes trade some recall for speed.

---

## 4. Implement keyword retrieval

Create:

```python
def keyword_search(query: str, limit: int = 10):
    ...
```

Return the same structure as vector search:

```python
@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    source: str
```

---

## 5. Keep vector search separate

You should now have:

```python
vector_search(query)
keyword_search(query)
```

Test them independently first.

Example:

```text
Query:
"What is authz_version?"
```

Vector search might return:

```text
1. Authorization architecture
2. JWT permissions
3. API authentication
4. Versioning
```

Keyword search might return:

```text
1. authz_version implementation
2. Authorization cache
3. Token invalidation
```

---

## 6. Standardize the result format

Both searches should return:

```python
@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    source: str
```

Important:

> **Don't directly compare raw vector and keyword scores.**

They use different scales. RRF combines ranks instead of raw scores.

---

## 7. Implement Reciprocal Rank Fusion

RRF is the main feature of Day 12.

If vector search returns:

```text
A → rank 1
B → rank 2
C → rank 3
```

and keyword search returns:

```text
C → rank 1
A → rank 2
D → rank 3
```

use:

```text
RRF score = 1 / (k + rank)
```

Start with:

```text
k = 60
```

A result that ranks well in both systems gets a strong combined ranking.

---

## 8. Implement RRF yourself

Don't install another retrieval framework.

For V1:

```python
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
):
    scores = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = (
                scores.get(result.chunk_id, 0)
                + 1 / (k + rank)
            )

    ...
```

Then sort by RRF score descending.

---

## 9. Create `hybrid_search()`

```python
def hybrid_search(query: str, limit: int = 5):
    vector_results = vector_search(query, limit=20)
    keyword_results = keyword_search(query, limit=20)

    return reciprocal_rank_fusion(
        [
            vector_results,
            keyword_results,
        ]
    )[:limit]
```

Conceptually:

```text
vector → candidates
keyword → candidates
             ↓
            RRF
             ↓
        final top K
```

Retrieve more candidates than you ultimately return so the fusion stage has enough candidates to work with.

---

## 10. Test exact terminology

Create:

```text
The payment service uses authz_version to invalidate
stale authorization tokens.
```

Ask:

```text
What is authz_version?
```

Keyword retrieval should strongly favor the document containing the exact identifier.

Hybrid search should preserve that exact match.

---

## 11. Test paraphrasing

Document:

```text
The system prevents duplicate payment processing
using idempotency keys.
```

Query:

```text
How do we make sure the same payment isn't processed twice?
```

There may be very little lexical overlap.

Vector search should perform better here, while hybrid search should still retain the semantic result.

---

## 12. Test a mixed query

```text
How does authz_version prevent stale authorization?
```

Expected:

```text
keyword search
    ↓
authz_version

vector search
    ↓
stale authorization / token invalidation

          ↓

       hybrid
          ↓
      best result
```

---

## 13. Add document metadata filtering

Add basic metadata:

```text
source_type
owner_id
document_type
project
```

Eventually:

```python
hybrid_search(
    query="authentication",
    project="jarvis",
)
```

The database becomes:

```text
                  Query
                    │
          ┌─────────┴─────────┐
          │                   │
       Filter              Search
          │                   │
          ▼                   ▼
      Jarvis docs       vector + keyword
```

pgvector supports filtering vector searches using normal PostgreSQL conditions.

---

## 14. Apply filters inside both retrieval paths

Prefer:

```text
filter
  ↓
keyword/vector search
  ↓
fusion
  ↓
final results
```

not:

```text
search everything
      ↓
top 100
      ↓
filter
```

Otherwise relevant candidates may be discarded before fusion.

---

## 15. Add a retrieval debug mode

Create:

```bash
python -m jarvis.search "How does authz_version work?" --debug
```

Example:

```text
QUERY
-----
How does authz_version work?


VECTOR RESULTS
--------------

1. authorization.md
   score: 0.89

2. authentication.md
   score: 0.83

3. token-cache.md
   score: 0.79


KEYWORD RESULTS
---------------

1. authorization.md
   rank: 1
   score: 12.4

2. api-versioning.md
   rank: 2
   score: 8.1


RRF RESULTS
-----------

1. authorization.md
   rrf: 0.0325

2. authentication.md
   rrf: 0.0161

3. api-versioning.md
   rrf: 0.0159
```

This makes it possible to understand why Jarvis retrieved something.

---

## 16. Add retrieval evaluation

For every evaluation query:

```text
query
expected_document
```

Run:

```text
vector search
keyword search
hybrid search
```

Then compare:

```text
                 Top-1    Top-3    Top-5

Vector             70%      90%      95%
Keyword            65%      82%      90%
Hybrid             80%      95%     100%
```

Those numbers are examples. Your actual measurements are what matter.

Track at least:

- Top-1 accuracy
- Top-3 recall
- Top-5 recall

---

## 17. Don't add a reranker yet

You may see:

```text
Query
 ↓
Hybrid Search
 ↓
Reranker
 ↓
LLM
```

A reranker adds another model and inference step.

For Day 12:

```text
Query
 ↓
Keyword + Vector
 ↓
RRF
 ↓
LLM
```

That's enough.

---

## 18. Don't change the LLM yet

Keep:

```text
retrieval
    ↓
context builder
    ↓
LLM
```

Only replace:

```text
vector_search()
```

with:

```text
hybrid_search()
```

This lets you measure retrieval improvements without changing generation.

---

## 19. Improve the context builder

Instead of:

```text
[chunk 1]
[chunk 2]
[chunk 3]
```

include provenance:

```text
SOURCE: redis.md

Redis can implement rate limiting...

---

SOURCE: architecture.md

The API gateway applies...
```

The LLM can now distinguish where retrieved information came from.

---

## 20. Deduplicate retrieved chunks

If:

```text
Vector:
A
B
C

Keyword:
A
D
E
```

the final context should be:

```text
A
B
C
D
E
```

not:

```text
A
B
C
A
D
E
```

Use `chunk_id` as the identity.

---

## 21. Add a context budget

Don't blindly send every retrieved chunk to the LLM.

```python
MAX_CONTEXT_TOKENS = ...
```

Then:

```text
Top results
    ↓
ranked
    ↓
add chunks
    ↓
token budget reached
    ↓
stop
```

---

## 22. Keep retrieval modular

At the end of Day 12:

```text
RetrievalService
│
├── VectorRetriever
│
├── KeywordRetriever
│
└── HybridRetriever
       │
       └── RRF
```

The application should call:

```python
retriever.search(query)
```

The LLM/router should not know about:

```text
pgvector
tsvector
RRF
HNSW
GIN
```

Those are retrieval implementation details.

---

## 23. Updated architecture

```text
                         USER
                           │
                           ▼
                          STT
                           │
                           ▼
                         ROUTER
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        search_web()            search_knowledge()
              │                         │
              ▼                         ▼
           SearXNG               HybridRetriever
              │                         │
              │                ┌────────┴────────┐
              │                ▼                 ▼
              │          Keyword Search     Vector Search
              │                │                 │
              │                └────────┬────────┘
              │                         ▼
              │                        RRF
              │                         │
              │                         ▼
              │                  Relevant chunks
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

---

## 24. Day 12 Definition of Done

### PostgreSQL

- [ ] PostgreSQL full-text search enabled.
- [ ] `tsvector` representation added.
- [ ] GIN index created.
- [ ] Vector search still working.
- [ ] Existing pgvector storage unchanged.

### Retrieval

- [ ] `keyword_search()` implemented.
- [ ] `vector_search()` implemented.
- [ ] Both return the same `SearchResult` structure.
- [ ] Search results contain rank and provenance.
- [ ] `hybrid_search()` implemented.
- [ ] Reciprocal Rank Fusion implemented.
- [ ] Duplicate chunks removed.
- [ ] Metadata filters implemented.
- [ ] Filters applied before final ranking.
- [ ] Context token budget implemented.

### Testing

- [ ] Exact identifier query tested.
- [ ] Semantic/paraphrased query tested.
- [ ] Mixed query tested.
- [ ] Vector vs keyword vs hybrid compared.
- [ ] Retrieval evaluation dataset expanded.
- [ ] Debug retrieval mode implemented.
- [ ] Top-1/Top-3/Top-5 metrics recorded.

### LLM integration

- [ ] Context builder updated with provenance.
- [ ] LLM generation pipeline unchanged.
- [ ] Hybrid retrieval replaces vector-only retrieval.
- [ ] No reranker yet.
- [ ] No query rewriting yet.
- [ ] No agentic retrieval yet.

---

# Day 12 Milestone

Jarvis should handle:

> **"What is authz_version?"**

```text
             Query
               │
       ┌───────┴────────┐
       ▼                ▼
   Keyword           Vector
       │                │
       ▼                ▼
authz_version       authorization
       │                │
       └───────┬────────┘
               ▼
              RRF
               │
               ▼
        authorization.md
               │
               ▼
              LLM
               │
               ▼
             Answer
```

And:

> **"How do we make sure the same payment isn't processed twice?"**

```text
             Query
               │
       ┌───────┴────────┐
       ▼                ▼
   Keyword           Vector
       │                │
       │          idempotency
       │             keys
       │                │
       └───────┬────────┘
               ▼
              RRF
               │
               ▼
        payment.md
               │
               ▼
              LLM
```

This gives Jarvis both exact technical terminology retrieval and semantic/paraphrase retrieval.

---

# What to deliberately postpone

Do not add these yet:

```text
❌ Cross-encoder reranker
❌ Query rewriting
❌ Multi-query retrieval
❌ Parent/child chunking
❌ Agentic RAG
❌ Graph RAG
❌ Elasticsearch
❌ Dedicated vector DB
❌ Automatic document classification
❌ Complex relevance scoring
```

Keep the progression:

```text
Day 11
Basic vector RAG
      ↓
Day 12
Hybrid retrieval
      ↓
Day 13
Retrieval quality / reranking
      ↓
Day 14+
Agent/router behavior
```

The important thing is that **Day 12 makes retrieval better without making Jarvis fundamentally more complicated**.

## References

- pgvector official repository — vector search, full-text hybrid search, RRF, and approximate indexing.
- pgvector Python official RRF example — concrete PostgreSQL + pgvector + full-text hybrid implementation.
- pgvector reference — filtering and hybrid-search guidance.
