"""Technical vocabulary for the Speech Normalizer (see plan items 4-5, 10).

Confidence is per `TechnicalTerm`, not per alias, so a term is only ever as
trustworthy as its riskiest alias:

  >= 0.95           unambiguous phonetic mishearings / casing fixes for
                     well-known terms (e.g. "gore teen" -> goroutine, "jay vee
                     em" -> JVM) - corrected automatically, no context needed.
  0.75 - 0.95       plausible but with real collision risk against ordinary
                     English (e.g. "red is" for Redis, "g one" for G1) - only
                     corrected when a `VocabularyContext` confirms the topic.
  < 0.75            too ambiguous to correct deterministically at all (e.g.
                     "g c" for GC, "gill" for GIL) - left alone unless an LLM
                     fallback confirms it.
"""

from .models import TechnicalTerm, VocabularyContext

TECH_TERMS: list[TechnicalTerm] = [
    # -- Go ---------------------------------------------------------------
    TechnicalTerm(
        canonical="goroutine",
        aliases=["go routine", "go-routine", "gore teen", "goreteen", "gore teens"],
        category="golang",
        confidence=0.97,
        related_terms=["channel", "mutex", "waitgroup", "scheduler"],
    ),
    TechnicalTerm(
        canonical="WaitGroup",
        aliases=["waitgroup", "wait group", "wait groups"],
        category="golang",
        confidence=0.85,
        related_terms=["goroutine"],
    ),
    TechnicalTerm(canonical="channel", aliases=[], category="golang"),
    TechnicalTerm(canonical="mutex", aliases=[], category="golang"),
    TechnicalTerm(canonical="scheduler", aliases=[], category="golang"),
    TechnicalTerm(canonical="select", aliases=[], category="golang"),
    TechnicalTerm(canonical="context", aliases=[], category="golang"),
    # -- Java ---------------------------------------------------------------
    TechnicalTerm(canonical="JVM", aliases=["jvm", "j v m", "jay vee em"], category="java", confidence=0.97),
    TechnicalTerm(canonical="JIT", aliases=["jit", "j i t", "jay eye tee"], category="java", confidence=0.95),
    TechnicalTerm(canonical="GC", aliases=["g c"], category="java", confidence=0.6),
    TechnicalTerm(canonical="G1", aliases=["g one", "gee one"], category="java", confidence=0.8),
    TechnicalTerm(canonical="ZGC", aliases=["z g c", "zee gee cee"], category="java", confidence=0.85),
    TechnicalTerm(canonical="heap", aliases=[], category="java"),
    TechnicalTerm(canonical="thread", aliases=[], category="java"),
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
    # -- DevOps / infra ---------------------------------------------------
    TechnicalTerm(
        canonical="Kubernetes",
        aliases=["kubernetes", "kubernetees", "koo ber net eez"],
        category="devops",
        confidence=0.96,
    ),
    TechnicalTerm(canonical="Docker", aliases=["docker"], category="devops", confidence=0.97),
    TechnicalTerm(canonical="Terraform", aliases=["terraform"], category="devops", confidence=0.96),
    TechnicalTerm(
        canonical="Prometheus", aliases=["prometheus", "prometheous"], category="devops", confidence=0.95
    ),
    TechnicalTerm(canonical="YAML", aliases=["yaml", "yammel"], category="devops", confidence=0.95),
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
    TechnicalTerm(canonical="OAuth", aliases=["oauth", "o auth", "oh auth"], category="auth", confidence=0.95),
    TechnicalTerm(canonical="OIDC", aliases=["oidc", "o i d c"], category="auth", confidence=0.8),
    TechnicalTerm(canonical="PKCE", aliases=["pkce", "pixy", "pixie"], category="auth", confidence=0.75),
    # -- Python / backend -----------------------------------------------
    TechnicalTerm(canonical="FastAPI", aliases=["fastapi", "fast api"], category="python", confidence=0.96),
    TechnicalTerm(
        canonical="SQLAlchemy",
        aliases=["sqlalchemy", "sql alchemy", "sequel alchemy"],
        category="python",
        confidence=0.95,
    ),
    TechnicalTerm(
        canonical="asyncio", aliases=["asyncio", "async io", "a sync io"], category="python", confidence=0.8
    ),
    TechnicalTerm(canonical="GIL", aliases=["gil", "g i l", "gill"], category="python", confidence=0.6),
    TechnicalTerm(canonical="Kafka", aliases=["kafka"], category="backend", confidence=0.97),
    TechnicalTerm(
        canonical="RabbitMQ",
        aliases=["rabbitmq", "rabbit m q", "rabbit mq"],
        category="backend",
        confidence=0.95,
    ),
    TechnicalTerm(canonical="JSON", aliases=["json", "jay son"], category="backend", confidence=0.96),
    TechnicalTerm(
        canonical="Elasticsearch",
        aliases=["elasticsearch", "elastic search"],
        category="search",
        confidence=0.96,
    ),
]


def context_for_domain(domain: str) -> VocabularyContext:
    """Builds a `VocabularyContext` from every `TECH_TERMS` entry in `domain` (see
    plan item 9). Used both to gate context-dependent corrections and to list
    "known technical vocabulary" for the LLM fallback prompt (plan item 13)."""
    return VocabularyContext(domain=domain, terms=[t for t in TECH_TERMS if t.category == domain])


GO_CONTEXT = context_for_domain("golang")
JAVA_CONTEXT = context_for_domain("java")
PYTHON_CONTEXT = context_for_domain("python")
