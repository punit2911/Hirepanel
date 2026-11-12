"""Skill taxonomy: canonical skill -> aliases. Extend freely; no code changes needed."""
from __future__ import annotations

import re
from dataclasses import dataclass

TAXONOMY: dict[str, list[str]] = {
    # languages
    "python": ["python", "python3"], "java": ["java"], "javascript": ["javascript", "js", "ecmascript", "es6"],
    "typescript": ["typescript"], "c++": ["c++", "cpp"], "c#": ["c#", "csharp", "c sharp"],
    "go": ["golang", "go lang"], "rust": ["rust"], "sql": ["sql", "t-sql", "pl/sql", "plsql"],
    "scala": ["scala"], "kotlin": ["kotlin"], "swift": ["swift"], "php": ["php"], "ruby": ["ruby"],
    "r": ["r language", "rstudio", "r programming"], "matlab": ["matlab"],
    "bash": ["bash", "shell scripting", "shell script"],
    # web / backend
    "django": ["django"], "flask": ["flask"], "fastapi": ["fastapi"], "spring": ["spring boot", "spring"],
    "react": ["react", "react.js", "reactjs"], "angular": ["angular", "angularjs"], "vue": ["vue", "vue.js", "vuejs"],
    "node.js": ["node.js", "nodejs", "node"], "next.js": ["next.js", "nextjs"], ".net": [".net", "dotnet", "asp.net"],
    "html": ["html", "html5"], "css": ["css", "css3", "sass", "tailwind"], "redux": ["redux"],
    "rails": ["ruby on rails", "rails"], "graphql": ["graphql"], "rest api": ["rest api", "rest apis", "restful", "rest"],
    "grpc": ["grpc"], "microservices": ["microservices", "microservice"], "websockets": ["websockets", "websocket"],
    # data / ML
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning", "neural networks", "neural network"],
    "pandas": ["pandas"], "numpy": ["numpy"], "scipy": ["scipy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"], "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf.keras"], "keras": ["keras"], "jax": ["jax"], "xgboost": ["xgboost", "lightgbm", "catboost"],
    "spark": ["apache spark", "spark", "pyspark"], "hadoop": ["hadoop", "hive"], "airflow": ["airflow", "apache airflow", "prefect", "dagster"],
    "dbt": ["dbt"], "kafka": ["kafka", "apache kafka"], "flink": ["flink"], "mlflow": ["mlflow", "weights & biases", "wandb", "weights and biases", "kubeflow"],
    "hugging face": ["hugging face", "huggingface"], "transformers": ["transformers", "bert", "gpt"],
    "langchain": ["langchain", "llamaindex", "llama index", "langgraph"], "llm": ["llm", "llms", "large language model", "large language models", "generative ai", "genai"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "fine-tuning": ["fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft"],
    "prompt engineering": ["prompt engineering", "prompt design"], "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "opencv", "image classification", "object detection"],
    "reinforcement learning": ["reinforcement learning"], "mlops": ["mlops", "ml ops"],
    "feature store": ["feature store", "feast"], "faiss": ["faiss"], "pinecone": ["pinecone"], "weaviate": ["weaviate"],
    "milvus": ["milvus"], "chromadb": ["chromadb", "chroma"], "pgvector": ["pgvector"],
    "vector database": ["vector database", "vector databases", "vector db", "vector store", "vector search"],
    "time series": ["time series", "forecasting"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling", "statistical modelling", "hypothesis testing", "a/b testing", "ab testing"],
    "data analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "data engineering": ["data engineering", "etl", "elt", "data pipelines", "data pipeline"],
    "data warehousing": ["data warehouse", "data warehousing"],
    "recommendation systems": ["recommendation system", "recommendation systems", "recommender system", "recommender systems"],
    "model serving": ["model serving", "model deployment", "inference serving", "torchserve", "bentoml", "triton", "tf serving"],
    "onnx": ["onnx"], "cuda": ["cuda", "gpu programming"],
    "distributed training": ["distributed training", "deepspeed", "horovod", "fsdp"],
    # cloud / devops
    "aws": ["aws", "amazon web services", "ec2", "s3", "sagemaker", "aws lambda", "eks", "ecs"],
    "gcp": ["gcp", "google cloud", "google cloud platform", "vertex ai"],
    "azure": ["azure", "azure ml"], "docker": ["docker", "dockerfile", "containerization", "containerisation"],
    "kubernetes": ["kubernetes", "k8s", "helm"], "terraform": ["terraform", "cloudformation", "pulumi"],
    "ansible": ["ansible"], "jenkins": ["jenkins"], "github actions": ["github actions", "gitlab ci", "circleci"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous delivery", "continuous deployment"],
    "linux": ["linux", "unix"], "git": ["git"], "prometheus": ["prometheus"], "grafana": ["grafana"],
    "datadog": ["datadog"], "observability": ["observability", "monitoring"], "serverless": ["serverless"], "nginx": ["nginx"],
    # databases
    "postgresql": ["postgresql", "postgres"], "mysql": ["mysql", "mariadb"], "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"], "elasticsearch": ["elasticsearch", "opensearch", "elk"], "snowflake": ["snowflake"],
    "bigquery": ["bigquery"], "redshift": ["redshift"], "cassandra": ["cassandra"], "dynamodb": ["dynamodb"],
    "sqlite": ["sqlite"], "oracle": ["oracle db", "oracle database"], "neo4j": ["neo4j"],
    # practices
    "system design": ["system design"], "distributed systems": ["distributed systems"],
    "data structures": ["data structures"], "algorithms": ["algorithms"],
    "unit testing": ["unit testing", "pytest", "junit", "jest", "test-driven development", "tdd"],
    "agile": ["agile", "scrum", "kanban"],
    "event-driven": ["event-driven", "event driven", "message queue", "message queues", "rabbitmq", "sqs", "pub/sub"],
    "security": ["oauth", "jwt", "owasp", "penetration testing"],
    # BI / business tools
    "tableau": ["tableau"], "power bi": ["power bi", "powerbi"], "looker": ["looker"],
    "excel": ["excel", "microsoft excel", "advanced excel", "vlookup", "pivot tables"],
    "jira": ["jira"], "figma": ["figma"], "salesforce": ["salesforce"], "sap": ["sap"],
    "selenium": ["selenium", "cypress", "playwright"], "postman": ["postman"],
    # soft / leadership
    "leadership": ["leadership", "team lead", "led a team", "led team", "managed a team", "led a group", "tech lead"],
    "mentoring": ["mentoring", "mentored", "mentorship", "coaching"],
    "communication": ["communication skills", "stakeholder communication", "presentations"],
    "stakeholder management": ["stakeholder management", "stakeholder engagement"],
    "project management": ["project management", "program management", "pmp"],
    "product management": ["product management", "product roadmap"],
    "cross-functional collaboration": ["cross-functional", "cross functional"],
    # non-tech roles
    "sales": ["sales", "b2b sales", "business development"], "crm": ["crm", "hubspot"],
    "seo": ["seo", "sem"], "content marketing": ["content marketing"],
    "financial modeling": ["financial modeling", "financial modelling"],
    "accounting": ["accounting", "bookkeeping", "gaap", "ifrs"],
    "customer support": ["customer support", "customer service"],
    "recruiting": ["recruiting", "talent acquisition", "sourcing"],
}

# Skills that are "close enough" for partial credit (used by the trajectory lens only).
RELATED_GROUPS: list[set[str]] = [
    {"pytorch", "tensorflow", "keras", "jax"}, {"aws", "gcp", "azure"},
    {"mysql", "postgresql", "sqlite", "oracle"}, {"react", "angular", "vue"},
    {"spark", "hadoop", "flink", "kafka"}, {"airflow", "dbt", "data engineering"},
    {"mlflow", "mlops", "model serving"}, {"pandas", "numpy", "scipy"},
    {"scikit-learn", "xgboost", "machine learning", "statistics"},
    {"java", "kotlin", "scala", "c#"}, {"terraform", "ansible"},
    {"jenkins", "github actions", "ci/cd"}, {"fastapi", "flask", "django"},
    {"pinecone", "faiss", "weaviate", "milvus", "chromadb", "pgvector", "vector database"},
    {"llm", "rag", "fine-tuning", "langchain", "prompt engineering", "transformers", "nlp", "hugging face"},
    {"docker", "kubernetes"}, {"snowflake", "bigquery", "redshift", "data warehousing"},
    {"tableau", "power bi", "looker"}, {"deep learning", "pytorch", "tensorflow", "computer vision", "nlp"},
]


# A requirement for the category is satisfied by any concrete member.
EXPANSIONS: dict[str, list[str]] = {
    "vector database": ["faiss", "pinecone", "weaviate", "milvus", "chromadb", "pgvector"],
    "data warehousing": ["snowflake", "bigquery", "redshift"],
    "ci/cd": ["jenkins", "github actions"],
    "observability": ["prometheus", "grafana", "datadog"],
    "deep learning": ["pytorch", "tensorflow", "keras", "jax"],
    "cloud": ["aws", "gcp", "azure"],
}


def expand_group(group: list[str]) -> list[str]:
    out = list(group)
    for g in group:
        for member in EXPANSIONS.get(g, []):
            if member not in out:
                out.append(member)
    return out


def related_to(skill: str) -> set[str]:
    out: set[str] = set()
    for g in RELATED_GROUPS:
        if skill in g:
            out |= g
    out.discard(skill)
    return out


_ALIAS2CANON: dict[str, str] = {}
for _canon, _aliases in TAXONOMY.items():
    for _a in _aliases + [_canon]:
        _ALIAS2CANON[_a.lower()] = _canon


def _alias_pattern(alias: str) -> str:
    return r"\s+".join(re.escape(part) for part in alias.split())


_ALL_RX = re.compile(
    r"(?<![A-Za-z0-9+#])(?:"
    + "|".join(_alias_pattern(a) for a in sorted(_ALIAS2CANON, key=len, reverse=True))
    + r")(?![A-Za-z0-9+#])",
    re.I,
)


@dataclass(frozen=True)
class SkillHit:
    canon: str
    start: int
    end: int


def canonical(name: str) -> str | None:
    return _ALIAS2CANON.get(re.sub(r"\s+", " ", name.strip().lower()))


def find_skills(text: str) -> list[SkillHit]:
    hits = []
    for m in _ALL_RX.finditer(text):
        canon = _ALIAS2CANON.get(re.sub(r"\s+", " ", m.group(0).lower()))
        if canon:
            hits.append(SkillHit(canon, m.start(), m.end()))
    return hits


def skills_in(text: str) -> list[str]:
    seen: list[str] = []
    for h in find_skills(text):
        if h.canon not in seen:
            seen.append(h.canon)
    return seen
