"""Technical vocabulary data for the Speech Normalizer (Day 13 plan items 4-5, 10)
and the Day 14 Vocabulary Manager (day 14 items 3-5).

Three layers live here, all pure data - the behavior that uses them lives in
`normalizer.py` and `vocabulary_manager.py`:

  TECH_TERMS          every known technical term plus the STT mishearings that
                      should be corrected to it.
  VOCABULARY_DOMAINS  which terms belong to a conversation topic, so the manager
                      can activate a small slice of the vocabulary instead of the
                      whole database.
  DOMAIN_KEYWORDS     the phrases that say "the conversation is about this
                      domain", used for keyword scoring (day 14 item 7).

Confidence is per `TechnicalTerm`, not per alias, so a term is only ever as
trustworthy as its riskiest alias:

  >= 0.95           unambiguous phonetic mishearings / casing fixes for
                     well-known terms (e.g. "gore teen" -> goroutine, "jay vee
                     em" -> JVM) - corrected automatically, no context needed.
  0.75 - 0.95       plausible but with real collision risk against ordinary
                     English (e.g. "red is" for Redis, "my sequel" for MySQL) -
                     only corrected when the active vocabulary confirms the topic.
  < 0.75            too ambiguous to correct deterministically at all (e.g.
                     "g c" for GC, "gill" for GIL) - left alone unless an LLM
                     fallback confirms it.

A term whose aliases straddle two tiers is listed twice, once per tier (see
MySQL), so its unambiguous spelling stays auto-correctable while its ambiguous
one has to wait for context.

Terms are added when a real STT mistake is observed, not by trying to write a
complete dictionary up front (day 14 item 5).
"""

from .models import TechnicalTerm, VocabularyContext

TECH_TERMS: list[TechnicalTerm] = [
    # -- Go -----------------------------------------------------------------
    TechnicalTerm(
        canonical="goroutine",
        aliases=["go routine", "go-routine", "gore teen", "goreteen", "gore teens", "go routines"],
        category="golang",
        confidence=0.97,
        related_terms=["channel", "mutex", "WaitGroup", "scheduler"],
    ),
    TechnicalTerm(
        canonical="WaitGroup",
        aliases=["waitgroup", "wait group", "wait groups"],
        category="golang",
        confidence=0.85,
        related_terms=["goroutine"],
    ),
    TechnicalTerm(
        canonical="RWMutex",
        aliases=["rw mutex", "r w mutex", "read write mutex", "are w mutex"],
        category="golang",
        confidence=0.9,
        related_terms=["mutex"],
    ),
    TechnicalTerm(
        canonical="GOMAXPROCS",
        aliases=["gomaxprocs", "go max procs", "go max proc", "jay max procs", "gomax procs"],
        category="golang",
        confidence=0.95,
        related_terms=["scheduler"],
    ),
    TechnicalTerm(
        canonical="pprof",
        aliases=["pprof", "pee prof", "p prof", "pee proof"],
        category="golang",
        confidence=0.9,
    ),
    # "c go" / "see go" collide with ordinary speech ("see, Go is fast"), so cgo
    # stays context-gated.
    TechnicalTerm(canonical="cgo", aliases=["cgo", "c go", "see go"], category="golang", confidence=0.8),
    TechnicalTerm(canonical="channel", aliases=[], category="golang"),
    TechnicalTerm(canonical="mutex", aliases=[], category="golang"),
    TechnicalTerm(canonical="scheduler", aliases=[], category="golang"),
    TechnicalTerm(canonical="select", aliases=[], category="golang"),
    TechnicalTerm(canonical="context", aliases=[], category="golang"),
    TechnicalTerm(canonical="sync", aliases=[], category="golang"),
    TechnicalTerm(canonical="atomic", aliases=[], category="golang"),
    TechnicalTerm(canonical="defer", aliases=[], category="golang"),
    TechnicalTerm(canonical="panic", aliases=[], category="golang"),
    TechnicalTerm(canonical="recover", aliases=[], category="golang"),
    TechnicalTerm(canonical="slice", aliases=[], category="golang"),
    TechnicalTerm(canonical="struct", aliases=[], category="golang"),
    TechnicalTerm(canonical="interface", aliases=[], category="golang"),
    TechnicalTerm(canonical="pointer", aliases=[], category="golang"),
    # -- Java ---------------------------------------------------------------
    TechnicalTerm(canonical="JVM", aliases=["jvm", "j v m", "jay vee em"], category="java", confidence=0.97),
    TechnicalTerm(canonical="JIT", aliases=["jit", "j i t", "jay eye tee"], category="java", confidence=0.95),
    TechnicalTerm(canonical="GC", aliases=["g c"], category="java", confidence=0.6),
    TechnicalTerm(canonical="G1", aliases=["g one", "gee one"], category="java", confidence=0.8),
    TechnicalTerm(canonical="ZGC", aliases=["z g c", "zee gee cee"], category="java", confidence=0.85),
    # "spring" is an ordinary English word; only corrected once Java is the topic.
    TechnicalTerm(canonical="Spring", aliases=["spring"], category="java", confidence=0.8),
    TechnicalTerm(
        canonical="Spring Boot",
        aliases=["spring boot", "springboot"],
        category="java",
        confidence=0.92,
    ),
    TechnicalTerm(canonical="heap", aliases=[], category="java"),
    TechnicalTerm(canonical="thread", aliases=[], category="java"),
    TechnicalTerm(canonical="classloader", aliases=[], category="java"),
    TechnicalTerm(canonical="garbage collection", aliases=[], category="java"),
    # -- Python -------------------------------------------------------------
    TechnicalTerm(canonical="FastAPI", aliases=["fastapi", "fast api"], category="python", confidence=0.96),
    TechnicalTerm(
        canonical="SQLAlchemy",
        aliases=["sqlalchemy", "sql alchemy", "sequel alchemy"],
        category="python",
        confidence=0.95,
    ),
    TechnicalTerm(
        canonical="Pydantic",
        aliases=["pydantic", "pie dantic", "pi dantic"],
        category="python",
        confidence=0.95,
    ),
    TechnicalTerm(canonical="uvicorn", aliases=["uvicorn", "you vee corn"], category="python", confidence=0.95),
    TechnicalTerm(
        canonical="asyncio", aliases=["asyncio", "async io", "a sync io"], category="python", confidence=0.8
    ),
    # Deliberately close to "goroutine": which one a mishearing means depends
    # entirely on whether the conversation is about Go or Python.
    TechnicalTerm(
        canonical="coroutine",
        aliases=["co routine", "co-routine", "core routine"],
        category="python",
        confidence=0.9,
    ),
    TechnicalTerm(canonical="GIL", aliases=["gil", "g i l", "gill"], category="python", confidence=0.6),
    TechnicalTerm(canonical="event loop", aliases=[], category="python"),
    TechnicalTerm(canonical="await", aliases=[], category="python"),
    TechnicalTerm(canonical="async", aliases=[], category="python"),
    TechnicalTerm(canonical="decorator", aliases=[], category="python"),
    TechnicalTerm(canonical="generator", aliases=[], category="python"),
    # -- Databases ------------------------------------------------------------
    TechnicalTerm(
        canonical="PostgreSQL",
        aliases=["postgres", "post gres", "postgre sql", "post gray sequel"],
        category="database",
        confidence=0.96,
    ),
    TechnicalTerm(canonical="Redis", aliases=["redis", "red is", "reddis"], category="database", confidence=0.8),
    TechnicalTerm(
        canonical="pgvector",
        aliases=["pgvector", "pg vector", "p g vector"],
        category="database",
        confidence=0.95,
    ),
    TechnicalTerm(
        canonical="SQLite", aliases=["sqlite", "sql lite", "sequel lite"], category="database", confidence=0.96
    ),
    TechnicalTerm(
        canonical="MongoDB",
        aliases=["mongodb", "mongo db", "mango db"],
        category="database",
        confidence=0.95,
    ),
    TechnicalTerm(canonical="MySQL", aliases=["mysql"], category="database", confidence=0.96),
    # "my sql query is slow" / "my sequel to that post" are both real sentences,
    # so these spellings only become MySQL once databases are the topic.
    TechnicalTerm(canonical="MySQL", aliases=["my sql", "my sequel"], category="database", confidence=0.85),
    TechnicalTerm(canonical="index", aliases=[], category="database"),
    TechnicalTerm(canonical="transaction", aliases=[], category="database"),
    TechnicalTerm(canonical="replica", aliases=[], category="database"),
    TechnicalTerm(canonical="sharding", aliases=[], category="database"),
    TechnicalTerm(canonical="partition", aliases=[], category="database"),
    TechnicalTerm(canonical="query planner", aliases=[], category="database"),
    # -- Infrastructure -------------------------------------------------------
    TechnicalTerm(
        canonical="Kubernetes",
        aliases=["kubernetes", "kubernetees", "koo ber net eez", "coopernetes", "cooper netties"],
        category="infrastructure",
        confidence=0.96,
    ),
    TechnicalTerm(
        canonical="Docker", aliases=["docker", "doecker", "dockker"], category="infrastructure", confidence=0.97
    ),
    TechnicalTerm(canonical="Terraform", aliases=["terraform"], category="infrastructure", confidence=0.96),
    TechnicalTerm(
        canonical="Prometheus", aliases=["prometheus", "prometheous"], category="infrastructure", confidence=0.95
    ),
    TechnicalTerm(canonical="Grafana", aliases=["grafana", "gra fauna"], category="infrastructure", confidence=0.95),
    TechnicalTerm(canonical="YAML", aliases=["yaml", "yammel"], category="infrastructure", confidence=0.95),
    TechnicalTerm(
        canonical="Cilium", aliases=["cilium", "silly um", "see lium"], category="infrastructure", confidence=0.9
    ),
    TechnicalTerm(
        canonical="Traefik", aliases=["traefik", "tray fik", "tray feek"], category="infrastructure", confidence=0.9
    ),
    TechnicalTerm(canonical="Ingress", aliases=["ingress"], category="infrastructure", confidence=0.85),
    # "at the helm" is ordinary English; context-gated.
    TechnicalTerm(canonical="Helm", aliases=["helm"], category="infrastructure", confidence=0.8),
    TechnicalTerm(canonical="container", aliases=[], category="infrastructure"),
    TechnicalTerm(canonical="pod", aliases=[], category="infrastructure"),
    TechnicalTerm(canonical="sidecar", aliases=[], category="infrastructure"),
    TechnicalTerm(canonical="namespace", aliases=[], category="infrastructure"),
    # -- Networking / auth --------------------------------------------------
    TechnicalTerm(canonical="gRPC", aliases=["grpc", "g r p c", "gee rpc"], category="networking", confidence=0.96),
    TechnicalTerm(
        canonical="WebSocket",
        aliases=["web socket", "web sockets", "websocket"],
        category="networking",
        confidence=0.95,
    ),
    # "engine x" collides with ordinary English ("start the engine, x factor...");
    # medium confidence so it only fires with a confirmed "networking" topic.
    TechnicalTerm(
        canonical="NGINX", aliases=["nginx", "engine x", "en jinx"], category="networking", confidence=0.85
    ),
    TechnicalTerm(canonical="reverse proxy", aliases=[], category="networking"),
    TechnicalTerm(canonical="load balancer", aliases=[], category="networking"),
    TechnicalTerm(canonical="OAuth", aliases=["oauth", "o auth", "oh auth"], category="auth", confidence=0.95),
    TechnicalTerm(canonical="OIDC", aliases=["oidc", "o i d c"], category="auth", confidence=0.8),
    TechnicalTerm(canonical="PKCE", aliases=["pkce", "pixy", "pixie"], category="auth", confidence=0.75),
    # "jot" is an ordinary word ("jot that down"), so JWT waits for auth context.
    TechnicalTerm(canonical="JWT", aliases=["jwt", "j w t", "jay w t", "jot"], category="auth", confidence=0.9),
    TechnicalTerm(canonical="access token", aliases=[], category="auth"),
    # -- Backend / messaging -----------------------------------------------
    TechnicalTerm(canonical="Kafka", aliases=["kafka"], category="backend", confidence=0.97),
    TechnicalTerm(
        canonical="RabbitMQ",
        aliases=["rabbitmq", "rabbit m q", "rabbit mq"],
        category="backend",
        confidence=0.95,
    ),
    TechnicalTerm(canonical="JSON", aliases=["json", "jay son"], category="backend", confidence=0.96),
    TechnicalTerm(canonical="message queue", aliases=[], category="backend"),
    TechnicalTerm(canonical="consumer group", aliases=[], category="backend"),
    TechnicalTerm(canonical="backpressure", aliases=[], category="backend"),
    # -- Search ---------------------------------------------------------------
    TechnicalTerm(
        canonical="Elasticsearch",
        aliases=["elasticsearch", "elastic search"],
        category="search",
        confidence=0.96,
    ),
    TechnicalTerm(canonical="inverted index", aliases=[], category="search"),
    TechnicalTerm(canonical="vector search", aliases=[], category="search"),
    # -- Distributed systems ------------------------------------------------
    TechnicalTerm(canonical="Raft", aliases=["raft"], category="distributed_systems", confidence=0.8),
    TechnicalTerm(
        canonical="Paxos", aliases=["paxos", "pack sauce"], category="distributed_systems", confidence=0.9
    ),
    TechnicalTerm(canonical="quorum", aliases=[], category="distributed_systems"),
    TechnicalTerm(canonical="consensus", aliases=[], category="distributed_systems"),
    TechnicalTerm(canonical="leader election", aliases=[], category="distributed_systems"),
    TechnicalTerm(canonical="eventual consistency", aliases=[], category="distributed_systems"),
    TechnicalTerm(canonical="consistent hashing", aliases=[], category="distributed_systems"),
    TechnicalTerm(canonical="split brain", aliases=[], category="distributed_systems"),
    # -- General programming --------------------------------------------------
    TechnicalTerm(canonical="API", aliases=["api", "a p i"], category="general_programming", confidence=0.9),
    TechnicalTerm(canonical="CLI", aliases=["cli", "c l i"], category="general_programming", confidence=0.9),
    TechnicalTerm(canonical="SDK", aliases=["sdk", "s d k"], category="general_programming", confidence=0.9),
    TechnicalTerm(canonical="race condition", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="deadlock", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="stack trace", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="unit test", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="refactor", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="benchmark", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="throughput", aliases=[], category="general_programming"),
    TechnicalTerm(canonical="latency", aliases=[], category="general_programming"),
]


_TERMS_BY_CANONICAL: dict[str, list[TechnicalTerm]] = {}
for _term in TECH_TERMS:
    _TERMS_BY_CANONICAL.setdefault(_term.canonical, []).append(_term)


# Terms that belong to a domain without living there. A term's `category` is its
# home domain; these entries let one term show up in several conversations -
# Kafka is a "backend" term, but someone talking about distributed systems or
# infrastructure wants it in their active vocabulary too (day 14 item 6).
_DOMAIN_EXTRAS: dict[str, list[str]] = {
    "golang": ["GC", "gRPC", "race condition", "deadlock", "throughput", "latency"],
    "java": ["Kafka", "gRPC", "garbage collection", "thread", "throughput", "latency"],
    "python": ["JSON", "Redis", "PostgreSQL", "SQLAlchemy", "unit test"],
    "database": ["SQLAlchemy", "pgvector", "consistent hashing", "eventual consistency"],
    "infrastructure": ["NGINX", "Kafka", "Redis", "Elasticsearch", "gRPC", "YAML"],
    "networking": ["Kubernetes", "Traefik", "WebSocket"],
    "auth": ["JSON", "OAuth"],
    "backend": ["Redis", "PostgreSQL", "Elasticsearch", "gRPC"],
    "search": ["Elasticsearch", "index", "vector search", "pgvector"],
    "distributed_systems": [
        "Kafka",
        "RabbitMQ",
        "Redis",
        "gRPC",
        "Kubernetes",
        "sharding",
        "replica",
        "partition",
        "throughput",
        "latency",
    ],
    "general_programming": ["JSON", "YAML", "API"],
}


def _build_domains() -> dict[str, list[str]]:
    """domain -> canonical terms. Built from each term's home category plus the
    cross-domain extras above, so a term never has to be spelled out twice."""
    domains: dict[str, list[str]] = {}
    for term in TECH_TERMS:
        bucket = domains.setdefault(term.category, [])
        if term.canonical not in bucket:
            bucket.append(term.canonical)

    for domain, extras in _DOMAIN_EXTRAS.items():
        bucket = domains.setdefault(domain, [])
        for canonical in extras:
            if canonical not in _TERMS_BY_CANONICAL:
                raise ValueError(f"_DOMAIN_EXTRAS['{domain}'] references unknown term '{canonical}'")
            if canonical not in bucket:
                bucket.append(canonical)

    return domains


VOCABULARY_DOMAINS: dict[str, list[str]] = _build_domains()


# Human-readable topic names, used to open the STT vocabulary hint
# ("The conversation is about Go programming.", day 14 item 11).
DOMAIN_LABELS: dict[str, str] = {
    "golang": "Go programming",
    "java": "Java",
    "python": "Python",
    "database": "databases",
    "infrastructure": "infrastructure and Kubernetes",
    "networking": "networking",
    "auth": "authentication and authorization",
    "backend": "backend services and messaging",
    "search": "search infrastructure",
    "distributed_systems": "distributed systems",
    "general_programming": "software engineering",
}


# What the user calls a domain out loud, for explicit topic switches
# ("let's switch to Java", day 14 item 9). Bare "go" is safe here *only* because
# it has to follow a switch cue - it is deliberately absent from the scoring
# keywords below, where "let's go" would otherwise vote for Go.
DOMAIN_ALIASES: dict[str, list[str]] = {
    "golang": ["golang", "go lang", "go programming", "go concurrency", "go"],
    "java": ["java", "jvm", "spring boot"],
    "python": ["python", "asyncio", "fastapi"],
    "database": ["databases", "database", "postgres", "postgresql", "sql", "mysql", "mongodb"],
    "infrastructure": ["infrastructure", "infra", "devops", "kubernetes", "k8s", "docker", "terraform"],
    "networking": ["networking", "network", "grpc", "http"],
    "auth": ["auth", "authentication", "authorization", "oauth", "identity"],
    "backend": ["backend", "message queues", "messaging", "kafka"],
    "search": ["search", "elasticsearch", "full text search"],
    "distributed_systems": ["distributed systems", "distributed system", "distributed computing", "consensus"],
    "general_programming": ["programming", "software engineering", "coding", "general programming"],
}


# Decisive topic signals, worth a full point each in `detect_domain` scoring
# (day 14 item 7). Weaker signals are derived automatically from each domain's
# own vocabulary - see `vocabulary_manager.domain_keywords`.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "golang": [
        "golang",
        "go lang",
        "go concurrency",
        "go programming",
        "go runtime",
        "go scheduler",
        "go module",
        "goroutine",
        "goroutines",
        "go routine",
        "gore teen",
        "waitgroup",
        "wait group",
        "gomaxprocs",
        "go max procs",
    ],
    "java": [
        "java",
        "jvm",
        "jay vee em",
        "jit",
        "garbage collector",
        "garbage collection",
        "spring boot",
        "hotspot",
        "class loader",
        "classloader",
    ],
    "python": [
        "python",
        "asyncio",
        "async io",
        "coroutine",
        "co routine",
        "event loop",
        "fastapi",
        "fast api",
        "pydantic",
        "django",
        "pip install",
    ],
    "database": [
        "database",
        "databases",
        "postgres",
        "postgresql",
        "post gres",
        "sql query",
        "mysql",
        "mongodb",
        "mongo db",
        "query planner",
        "primary key",
        "foreign key",
    ],
    "infrastructure": [
        "kubernetes",
        "coopernetes",
        "kubectl",
        "docker",
        "dockerfile",
        "helm chart",
        "ingress",
        "terraform",
        "prometheus",
        "grafana",
    ],
    "networking": [
        "nginx",
        "engine x",
        "reverse proxy",
        "load balancer",
        "grpc",
        "gee rpc",
        "websocket",
        "web socket",
    ],
    "auth": [
        "oauth",
        "oidc",
        "authentication",
        "authorization",
        "access token",
        "refresh token",
        "jwt",
        "pkce",
        "single sign on",
    ],
    "backend": [
        "kafka",
        "rabbitmq",
        "rabbit mq",
        "message queue",
        "consumer group",
        "backpressure",
        "event driven",
    ],
    "search": [
        "elasticsearch",
        "elastic search",
        "inverted index",
        "full text search",
        "bm25",
        "vector search",
    ],
    "distributed_systems": [
        "distributed system",
        "distributed systems",
        "raft",
        "paxos",
        "consensus",
        "quorum",
        "leader election",
        "eventual consistency",
        "consistent hashing",
        "split brain",
        "cap theorem",
    ],
    "general_programming": [
        "refactor",
        "unit test",
        "code review",
        "stack trace",
        "race condition",
        "deadlock",
        "big o",
        "time complexity",
    ],
}


# Words that are technical terms *and* ordinary English. They stay in the
# vocabulary (they describe a domain and seed the STT hint) but are excluded
# from the keywords derived automatically from each domain's terms, so "please
# change the channel" or "I need to go" doesn't quietly make Go the active
# topic. A phrase listed in DOMAIN_KEYWORDS above is a curated override and
# still scores.
AMBIGUOUS_KEYWORDS: frozenset[str] = frozenset(
    {
        "go",
        "map",
        "select",
        "context",
        "interface",
        "struct",
        "pointer",
        "slice",
        "defer",
        "panic",
        "recover",
        "sync",
        "atomic",
        "index",
        "transaction",
        "partition",
        "replica",
        "async",
        "await",
        "container",
        "pod",
        "node",
        "namespace",
        "spring",
        "raft",
        "helm",
        "jot",
        "gill",
        "g c",
        "pixy",
        "pixie",
        "red is",
        "my sql",
        "my sequel",
        "c go",
        "see go",
        "api",
        "cli",
        "sdk",
        "a p i",
        "c l i",
        "s d k",
        "generator",
        "decorator",
        "benchmark",
        "latency",
        "throughput",
        "recover",
        "thread",
        "heap",
    }
)


def terms_for_domain(domain: str) -> list[TechnicalTerm]:
    """Every `TechnicalTerm` in `domain`, in declaration order. A canonical listed
    in several confidence tiers (MySQL) yields all of its entries."""
    terms: list[TechnicalTerm] = []
    for canonical in VOCABULARY_DOMAINS.get(domain, []):
        terms.extend(_TERMS_BY_CANONICAL.get(canonical, []))
    return terms


def context_for_domain(domain: str) -> VocabularyContext:
    """Builds a `VocabularyContext` for a single domain (see Day 13 plan item 9).
    Used both to gate context-dependent corrections and to list "known technical
    vocabulary" for the LLM fallback prompt (Day 13 plan item 13)."""
    return VocabularyContext(domain=domain, terms=terms_for_domain(domain))


def known_domains() -> list[str]:
    return list(VOCABULARY_DOMAINS)


GO_CONTEXT = context_for_domain("golang")
JAVA_CONTEXT = context_for_domain("java")
PYTHON_CONTEXT = context_for_domain("python")
