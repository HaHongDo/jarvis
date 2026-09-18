# Day 13 — Speech Normalizer Implementation Plan

## Goal

Day 13 focuses entirely on improving speech-to-text accuracy **without switching to a bigger STT model**.

The Speech Normalizer sits between STT and the rest of Jarvis:

```text
Microphone
    ↓
VAD
    ↓
STT
    ↓
Raw Transcript
    ↓
Speech Normalizer
    ↓
Normalized Transcript
    ↓
Router / LLM
```

The main problem being solved is technical vocabulary being transcribed incorrectly:

```text
"gore teen" → "goroutine"
"jay vee em" → "JVM"
"post gres" → "PostgreSQL"
"gee rpc" → "gRPC"
```

Whisper supports `initial_prompt` for custom vocabulary/proper-noun context, and faster-whisper exposes `hotwords`. These can be integrated later, but the main Day 13 work is the post-STT normalization layer.

---

## 1. Define the responsibilities

The Speech Normalizer should do only three things:

### Fix known transcription mistakes

```text
go routine → goroutine
go-routine → goroutine
jay vee em → JVM
post gres → PostgreSQL
```

### Normalize technical terminology

```text
kubernetes → Kubernetes
grpc → gRPC
web socket → WebSocket
postgres → PostgreSQL
```

### Detect uncertain corrections

If a correction is ambiguous, preserve the original rather than aggressively changing the user's words.

The normalizer should be conservative.

---

## 2. Create the module

Suggested structure:

```text
jarvis/
├── speech/
│   ├── stt.py
│   ├── normalizer.py
│   └── vocabulary.py
│
├── llm/
├── retrieval/
├── router/
└── main.py
```

The rest of Jarvis should only need:

```python
result = normalizer.normalize(raw_text)
```

---

## 3. Define the result type

Do not return only a string.

```python
@dataclass
class NormalizationResult:
    original: str
    normalized: str
    changes: list[Correction]
```

And:

```python
@dataclass
class Correction:
    original: str
    replacement: str
    reason: str
    confidence: float
```

Example:

```python
NormalizationResult(
    original="how does a gore teen communicate",
    normalized="how does a goroutine communicate",
    changes=[
        Correction(
            original="gore teen",
            replacement="goroutine",
            reason="technical vocabulary",
            confidence=0.98,
        )
    ],
)
```

---

## 4. Build the vocabulary store

Create:

```text
speech/vocabulary.py
```

Start with a small vocabulary:

```python
TECH_TERMS = {
    "goroutine": [
        "go routine",
        "go-routine",
        "gore teen",
        "goreteen",
    ],

    "JVM": [
        "jvm",
        "j v m",
        "jay vee em",
    ],

    "PostgreSQL": [
        "postgres",
        "post gres",
        "postgre sql",
    ],

    "Kubernetes": [
        "kubernetes",
        "kubernetees",
    ],

    "gRPC": [
        "grpc",
        "g r p c",
        "gee rpc",
    ],
}
```

Do not try to create thousands of terms yet. Start with approximately 30–50 terms that you actually encounter.

---

## 5. Separate canonical terms from aliases

Prefer a structure that can eventually contain metadata:

```python
@dataclass
class TechnicalTerm:
    canonical: str
    aliases: list[str]
    category: str
    related_terms: list[str]
```

Example:

```python
TechnicalTerm(
    canonical="goroutine",
    aliases=[
        "go routine",
        "go-routine",
        "gore teen",
    ],
    category="golang",
    related_terms=[
        "channel",
        "mutex",
        "waitgroup",
        "scheduler",
    ],
)
```

This is better than maintaining only:

```python
"go routine": "goroutine"
```

because later you can make the vocabulary context-aware.

---

## 6. Implement deterministic normalization first

Create:

```python
def normalize_known_terms(text: str) -> str:
    ...
```

Example:

```text
Input:
"I want to create a go routine"

Output:
"I want to create a goroutine"
```

Do not involve an LLM yet.

---

## 7. Do not use naive `.replace()`

Avoid:

```python
text.replace("go routine", "goroutine")
```

because it can eventually produce incorrect replacements.

For example:

```text
"I have a routine written in Go"
```

must remain unchanged.

Use phrase matching with word boundaries.

For example:

```python
pattern = r"go[\s-]+routine"
```

Use case-insensitive matching.

---

## 8. Preserve canonical capitalization

Technical terminology often has meaningful capitalization.

Normalize:

```text
jvm
Jvm
jay vee em
```

to:

```text
JVM
```

Likewise:

```text
grpc
gRPC
g r p c
```

to:

```text
gRPC
```

---

## 9. Introduce vocabulary context

The same speech fragment can mean different things depending on the current topic.

Create:

```python
@dataclass
class VocabularyContext:
    domain: str | None
    terms: list[TechnicalTerm]
```

Then:

```python
normalizer.normalize(
    text,
    vocabulary_context=context,
)
```

For a Go conversation:

```text
goroutine
channel
mutex
WaitGroup
select
context
scheduler
```

For a Java conversation:

```text
JVM
JIT
GC
heap
thread
G1
ZGC
```

This makes corrections safer because the surrounding topic provides evidence.

---

## 10. Add deterministic confidence

Not all corrections are equally reliable.

Example:

```text
"jay vee em" → "JVM"
```

is highly reliable.

But:

```text
"gorteen" → "goroutine"
```

may be less certain.

Use levels such as:

```text
HIGH
MEDIUM
LOW
```

or numeric confidence:

```text
0.95+
0.75–0.95
<0.75
```

Suggested behavior:

```text
>= 0.95
    automatic correction

0.75–0.95
    context-dependent correction

<0.75
    preserve original
```

Determine the exact thresholds from your test corpus.

---

## 11. Build the normalizer pipeline

The first implementation should be:

```text
Raw STT
   ↓
Whitespace normalization
   ↓
Known terminology matching
   ↓
Context-aware matching
   ↓
Confidence check
   ↓
Normalized transcript
```

Keep this deterministic initially.

---

## 12. Add an LLM fallback only after deterministic matching works

Eventually:

```text
                 Raw STT
                    │
                    ▼
            Exact term matcher
                    │
             ┌──────┴──────┐
             │             │
        confident       uncertain
             │             │
             ▼             ▼
          replace       LLM check
             │             │
             └──────┬──────┘
                    ▼
              final transcript
```

The LLM should **not answer the user's question**.

Its only job is to determine whether a suspicious phrase is probably a transcription error.

---

## 13. Give the correction LLM a strict contract

Example:

```text
You are a speech transcription correction system.

The transcript came from an English speech-to-text model.

Correct only likely transcription errors.

Do not:
- answer the user's question
- rewrite sentences
- improve grammar unnecessarily
- change the user's meaning
- invent information

The user is currently discussing Go programming.

Known technical vocabulary:
goroutine
channel
mutex
WaitGroup
scheduler

Transcript:
"How does the gore teen scheduler work in go?"

Return only the corrected transcript.
```

Expected:

```text
How does the goroutine scheduler work in Go?
```

---

## 14. Prefer structured LLM output

If an LLM fallback is used:

```json
{
  "changed": true,
  "text": "How does the goroutine scheduler work in Go?",
  "changes": [
    {
      "from": "gore teen",
      "to": "goroutine",
      "confidence": 0.98
    }
  ]
}
```

Validate the result before accepting it.

For example, reject a result if the LLM suddenly changes many words in a short sentence.

---

## 15. Add a maximum-correction rule

The normalizer must never become a general rewriting system.

Input:

```text
Explain how goroutines communicate.
```

Unacceptable output:

```text
Explain the differences between goroutines and threads.
```

That changes the user's meaning.

For V1:

```text
Only replace suspicious terms.
Do not add clauses.
Do not remove clauses.
Do not change the question.
Do not answer the question.
```

---

## 16. Always preserve the raw transcript

Never discard the original STT output.

Store:

```text
raw:
"How does a gore teen communicate?"

normalized:
"How does a goroutine communicate?"
```

This is essential for debugging whether a problem came from:

```text
STT
or
Speech Normalizer
or
LLM
```

---

## 17. Add a correction log

During development:

```text
[SpeechNormalizer]

RAW:
How does a gore teen communicate with another gore teen?

MATCH:
"gore teen" → "goroutine"

MATCH:
"gore teen" → "goroutine"

OUTPUT:
How does a goroutine communicate with another goroutine?
```

Make debug logging configurable so it can later be disabled.

---

## 18. Build a speech normalization test corpus

Create:

```text
tests/
└── speech/
    └── terminology.json
```

Example:

```json
[
  {
    "input": "how does a gore teen work",
    "expected": "how does a goroutine work"
  },
  {
    "input": "how does the jay vee em work",
    "expected": "how does the JVM work"
  },
  {
    "input": "how does post gres store data",
    "expected": "how does PostgreSQL store data"
  }
]
```

Then run:

```bash
pytest tests/speech/
```

---

## 19. Add negative tests

This is extremely important.

Don't only test things that should be corrected.

Example:

```text
Input:
"I wrote a routine in Go"

Expected:
"I wrote a routine in Go"
```

Not:

```text
"I wrote a routine in goroutine"
```

Other examples:

```text
"Go is a programming language"
```

must remain unchanged.

Evaluate:

```text
correct corrections
+
correctly unchanged sentences
+
false corrections
+
missed corrections
```

---

## 20. Build an accuracy metric

Your test output should eventually look like:

```text
Speech Normalizer Evaluation
============================

Total cases:        100

Corrected:           72
Correctly unchanged: 24
Incorrectly changed:  3
Missed corrections:   1

Correction precision: 96%
Correction recall:    98%
```

For this module, precision is particularly important.

It is better for Jarvis to occasionally miss:

```text
"gore teen"
```

than to confidently rewrite normal sentences incorrectly.

---

## 21. Learn from your actual STT mistakes

Suppose you repeatedly observe:

```text
STT:
gore teen

Correct:
goroutine
```

After manually verifying it, add the mapping to your vocabulary.

Over time:

```text
Your speech
     ↓
Your STT model
     ↓
Your recurring errors
     ↓
Speech Normalizer vocabulary
```

This makes the normalizer adapt to your actual speaking style and STT behavior.

Do not automatically add corrections to the vocabulary yet. Verify them first.

---

## 22. Feed vocabulary back into STT later

Once the vocabulary is reliable, you can also use it upstream.

OpenAI Whisper's `initial_prompt` is intended to provide contextual vocabulary/proper nouns, while faster-whisper exposes `hotwords` as hint phrases.

The eventual architecture can therefore be:

```text
             Vocabulary Store
                 │
          ┌──────┴──────┐
          ▼             ▼
         STT       Normalizer
    initial_prompt   vocabulary
    / hotwords           │
          │              │
          └──────┬───────┘
                 ▼
          normalized text
```

Keep this as a later enhancement. Day 13 should first make the post-STT normalizer reliable.

---

## 23. Day 13 API

Aim for:

```python
class SpeechNormalizer:

    def normalize(
        self,
        text: str,
        context: VocabularyContext | None = None,
    ) -> NormalizationResult:
        ...
```

Internally:

```text
normalize()
    │
    ├── whitespace normalization
    │
    ├── exact vocabulary matching
    │
    ├── context-aware matching
    │
    ├── confidence calculation
    │
    └── optional LLM fallback
```

---

## 24. Final Day 13 architecture

```text
                       Microphone
                           │
                           ▼
                          VAD
                           │
                           ▼
                          STT
                           │
                           ▼
                    Raw Transcript
                           │
                           ▼
                 ┌──────────────────┐
                 │ SpeechNormalizer │
                 └────────┬─────────┘
                          │
                 ┌────────┴─────────┐
                 │                  │
                 ▼                  ▼
          Vocabulary Matcher    LLM Fallback
                 │                  │
                 └────────┬─────────┘
                          ▼
                  Normalized Text
                          │
                          ▼
                        Router
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
         Web Search                  RAG
             │                         │
             └────────────┬────────────┘
                          ▼
                          LLM
                           │
                           ▼
                          TTS
```

---

# 25. Implementation order

## Phase 1 — Foundation

- [ ] Create `speech/normalizer.py`
- [ ] Create `speech/vocabulary.py`
- [ ] Define `NormalizationResult`
- [ ] Define `Correction`
- [ ] Define `TechnicalTerm`
- [ ] Define `VocabularyContext`

## Phase 2 — Deterministic normalization

- [ ] Case-insensitive matching
- [ ] Word-boundary matching
- [ ] Multi-word phrase matching
- [ ] Canonical capitalization
- [ ] Basic confidence scoring
- [ ] Preserve original transcript

## Phase 3 — Vocabulary

Start with roughly 30–50 terms:

```text
goroutine
channel
mutex
WaitGroup
JVM
JIT
PostgreSQL
Redis
Kafka
Kubernetes
Docker
gRPC
WebSocket
OAuth
OIDC
PKCE
pgvector
FastAPI
SQLAlchemy
Elasticsearch
```

Add terms based on the vocabulary you actually speak.

## Phase 4 — Context

- [ ] Add `VocabularyContext`
- [ ] Add Go vocabulary
- [ ] Add Java vocabulary
- [ ] Add Python/backend vocabulary
- [ ] Select vocabulary based on current conversation

## Phase 5 — Testing

- [ ] 50 positive cases
- [ ] 30 negative cases
- [ ] Test multi-word corrections
- [ ] Test capitalization
- [ ] Test ambiguous terms
- [ ] Measure false corrections
- [ ] Measure missed corrections

## Phase 6 — LLM fallback

Only after the deterministic version works:

- [ ] Detect uncertain terms
- [ ] Send only suspicious cases to LLM
- [ ] Require structured output
- [ ] Reject large rewrites
- [ ] Preserve original if confidence is low

## Phase 7 — STT integration

Finally:

```text
Vocabulary Store
      │
      ├──→ STT initial_prompt / hotwords
      │
      └──→ SpeechNormalizer
```

---

# Day 13 Definition of Done

You should be able to say:

> "How does a gore teen communicate with another gore teen?"

and get:

```text
Raw STT:
How does a gore teen communicate with another gore teen?

        ↓

Speech Normalizer

        ↓

Normalized:
How does a goroutine communicate with another goroutine?
```

while:

> "I wrote a routine in Go."

remains:

```text
I wrote a routine in Go.
```

The core objective is:

> **Safe correction, not maximum correction.**

The normalizer should fix technical terminology when there is strong evidence, but otherwise leave the user's words alone.
