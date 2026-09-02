# Day 7 — Real Web Search Integration

## Goal

Replace the fake `search()` tool from Day 6 with a real SearXNG-backed search pipeline.

```text
User
 ↓
STT
 ↓
LLM
 ↓
search(query)
 ↓
Search Tool
 ↓
SearXNG
 ↓
Normalized Results
 ↓
LLM
 ↓
Natural-language answer
 ↓
TTS
```

SearXNG exposes an HTTP search API through `/` and `/search`. JSON output can be requested with `format=json` when JSON is enabled in the instance configuration. It also supports parameters such as `q`, `categories`, `language`, `pageno`, `time_range`, and `safesearch`.

## 1. Set up SearXNG locally

Run SearXNG locally rather than relying on a public instance.

```text
Jarvis
   │
   ▼
Search Tool
   │
   ▼
http://localhost:8080
   │
   ▼
SearXNG
   ├── Search engine A
   ├── Search engine B
   └── Search engine C
```

## 2. Enable the JSON API

Configure SearXNG so JSON is enabled in `settings.yml`.

Test:

```bash
curl "http://localhost:8080/search?q=rust&format=json"
```

If JSON is not enabled, SearXNG can return HTTP 403.

## 3. Create a SearXNG client

Add:

```text
search/
├── __init__.py
├── searxng.py
├── models.py
└── service.py
```

```python
class SearXNGClient:

    def search(
        self,
        query: str,
        *,
        category: str = "general",
        language: str = "en",
        time_range: str | None = None,
        page: int = 1,
    ):
        ...
```

Its responsibility should only be HTTP communication with SearXNG.

## 4. Define your own SearchResult model

Do not pass raw SearXNG JSON directly to the LLM.

```python
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str | None
    published_at: str | None
```

Pipeline:

```text
SearXNG JSON
 ↓
SearchResult[]
 ↓
LLM
```

## 5. Build a normalizer

Create:

```text
search/normalizer.py
```

```python
class SearchNormalizer:

    def normalize(self, response) -> list[SearchResult]:
        ...
```

Handle:

- missing titles
- missing URLs
- missing snippets
- duplicate URLs
- empty results
- malformed results

## 6. Deduplicate results

Normalize URLs before comparing them:

```text
URL
 ↓
remove obvious tracking parameters
 ↓
normalize hostname
 ↓
normalize trailing slash
 ↓
deduplicate
```

Do not build an advanced ranking system yet.

## 7. Limit results

Start with:

```text
SearXNG
 ↓
10 results
 ↓
normalize
 ↓
top 5
 ↓
LLM
```

For example:

```python
MAX_RESULTS = 5
```

Tune this later based on token usage, answer quality, latency, and context size.

## 8. Convert results into LLM-friendly text

Do not use embeddings yet.

Start with structured text:

```text
## Web Search Results

Query: Rust programming language

### Result 1
Title: Rust Programming Language
URL: https://www.rust-lang.org/
Source: rust-lang.org
Summary: Rust is a language empowering everyone to build reliable
and efficient software.

### Result 2
Title: Rust - Wikipedia
URL: https://en.wikipedia.org/wiki/Rust
Source: wikipedia.org
Summary: Rust is a multi-paradigm general-purpose programming language...
```

Then:

```text
Search results
 ↓
Prompt context
 ↓
LLM
```

## 9. Create a SearchContext formatter

Create:

```text
search/context.py
```

```python
class SearchContextBuilder:

    def build(self, query, results) -> str:
        ...
```

Keep this boundary clean:

```text
SearchResult[]
     ↓
SearchContext
     ↓
LLM
```

## 10. Update the search tool

Replace the Day 6 fake implementation with:

```text
search()
   ↓
SearXNGClient
   ↓
Normalizer
   ↓
Deduplicator
   ↓
Result limiter
   ↓
Context builder
   ↓
LLM
```

The LLM should still see:

```text
search(query)
```

## 11. Improve the search tool description

Tell the model:

```text
Search the internet for current or externally verifiable information.

Use this tool when:
- the user asks for current information
- the user asks about recent events
- the answer is likely outside your knowledge
- the user explicitly asks you to search the web

Do not use this tool for:
- basic reasoning
- simple calculations
- casual conversation
- questions you can confidently answer without external information
```

Do not build a separate classifier yet.

## 12. Test explicit search

Test:

> Search the web for the latest Rust release.

Expected:

```text
User
 ↓
LLM
 ↓
search("latest Rust release")
 ↓
SearXNG
 ↓
Results
 ↓
LLM
 ↓
Answer
```

Also test:

> Search for the latest news about OpenAI.

## 13. Test when search should NOT happen

Ask:

> What is a binary search tree?

The LLM should answer directly.

Desired behavior:

```text
Question
 ↓
LLM
 ├── Can answer directly → Answer
 └── Needs external information → Search
```

Not:

```text
Every question
 ↓
Internet search
```

## 14. Add search metadata

Retain metadata such as:

```json
{
  "query": "latest Rust release",
  "results": [],
  "result_count": 5,
  "searched_at": "2026-08-29T22:00:00Z"
}
```

This will later help with caching, debugging, observability, and freshness decisions.

## 15. Add a basic cache

Do not implement semantic caching yet.

Use normalized exact-query caching:

```text
"latest rust release"
        ↓
normalize
        ↓
hash
        ↓
cache
```

Architecture:

```text
Search request
      │
      ▼
   Cache?
   /    \
 HIT    MISS
  │       │
  │       ▼
  │    SearXNG
  │       │
  │       ▼
  │     Cache
  │       │
  └───┬───┘
      ▼
    LLM
```

Initial TTL ideas:

```text
general search: 5–30 minutes
news: shorter
static technical documentation: potentially longer
```

## 16. Don't use embeddings for search caching yet

For example:

```text
"What is the current Bitcoin price?"
```

and:

```text
"What was the Bitcoin price in 2020?"
```

are semantically similar but require different results.

Start with normalized exact-query caching.

Semantic caching can come later.

## 17. Preserve source attribution

Keep the source URL associated with each result:

```text
According to the Rust website [1], Rust ...

[1] https://www.rust-lang.org/
```

You do not need to speak the URL through TTS. Preserve it for future UI citations and debugging.

## 18. Test failure cases

### SearXNG unavailable

Expected:

```text
"I couldn't reach the web search service right now."
```

### No search results

Expected:

```text
"I couldn't find any useful results for that."
```

### SearXNG timeout

Return a controlled tool error instead of crashing the assistant.

## 19. Measure the pipeline

Record:

```text
STT latency
LLM tool-decision latency
SearXNG latency
normalization latency
LLM answer latency
TTS latency
```

Example:

```text
STT       300 ms
LLM       800 ms
Search    600 ms
LLM       900 ms
TTS       ...
----------------
Total     ...
```

## 20. Final architecture

```text
                         ┌──────────────┐
                         │  Microphone  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ Wake Word    │
                         └──────┬───────┘
                                │
                                ▼
                              STT
                                │
                                ▼
                         ┌──────────────┐
                         │     LLM      │
                         └──────┬───────┘
                                │
                         tool call: search
                                │
                                ▼
                     ┌────────────────────┐
                     │    Search Tool     │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │    Search Cache    │
                     └─────────┬──────────┘
                               │
                         cache miss
                               │
                               ▼
                     ┌────────────────────┐
                     │      SearXNG       │
                     └─────────┬──────────┘
                               │
                               ▼
                     Raw Search Results
                               │
                               ▼
                     ┌────────────────────┐
                     │    Normalizer      │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │  Dedup / Limit     │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │  Context Builder   │
                     └─────────┬──────────┘
                               │
                               ▼
                              LLM
                               │
                               ▼
                       Response Stream
                               │
                               ▼
                            Kokoro
                               │
                               ▼
                           Speaker
```

# Day 7 Definition of Done

- [ ] SearXNG running locally.
- [ ] JSON API enabled.
- [ ] `SearXNGClient` implemented.
- [ ] `SearchResult` model created.
- [ ] SearXNG responses normalized.
- [ ] Duplicate URLs removed.
- [ ] Results limited to a reasonable number.
- [ ] Search context formatter implemented.
- [ ] Fake `search()` tool replaced with real implementation.
- [ ] LLM can decide to call the search tool.
- [ ] Explicit web-search requests work.
- [ ] Normal questions don't unnecessarily trigger search.
- [ ] Search failures are handled gracefully.
- [ ] Basic query cache implemented.
- [ ] Search latency logged.
- [ ] Source URLs preserved.
- [ ] Voice response works end-to-end.

# Key Design Decision

**Do not turn search results into embeddings on Day 7.**

Start with:

```text
SearXNG JSON
     ↓
Normalized SearchResult[]
     ↓
Markdown/text context
     ↓
LLM
```

SearXNG already provides an HTTP API and aggregates results from configured search engines. Your application layer should focus on making those results small, clean, deduplicated, and useful to the LLM.

After this works, Day 8 should tackle the more interesting problem: deciding whether Jarvis actually needs to search the internet, together with smarter caching and query classification.

## References

- SearXNG Search API: https://docs.searxng.org/dev/search_api.html
- SearXNG Developer Documentation: https://docs.searxng.org/dev/index.html
- SearXNG JSON Engine: https://docs.searxng.org/dev/engines/json_engine.html
