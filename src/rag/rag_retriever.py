"""Main RAG retriever that builds context for project generation."""

from src.rag.vector_store import vector_store
from src.rag.knowledge_base import PERFECT_EXAMPLES
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CRITICAL_TEMPLATE_IDS = [
    "fastapi_main_template",
    "database_template",
    "requirements_template",
    "react_index_template",
    "index_css_template",
    "tsconfig_template",
    "package_json_template",
    "index_html_template",
    "backend_env_template",
    "frontend_env_template",
    "declarations_template",
    "pagination_crud",
]


class RAGRetriever:
    def __init__(self):
        self.initialized = False
        logger.info("RAGRetriever initialised")

    def initialize(self):
        if not self.initialized:
            vector_store.initialize()
            self.initialized = True
            logger.info("RAGRetriever ready")

    def get_context_for_project(self, problem_statement: str) -> str:
        """
        Retrieve relevant perfect examples and build context string
        to inject into the LLM prompt.
        """
        self.initialize()

        problem_lower = problem_statement.lower()

        always_include = [
            "fastapi_main_template", "database_template", "requirements_template",
            "react_index_template", "index_css_template", "tsconfig_template",
            "package_json_template", "index_html_template", "backend_env_template",
            "frontend_env_template", "declarations_template", "pagination_crud",
        ]

        if any(w in problem_lower for w in ["auth", "login", "register", "user", "password", "jwt", "saas"]):
            always_include += ["jwt_auth_backend", "user_model_auth", "auth_schemas", "react_auth_context", "login_page"]

        if any(w in problem_lower for w in ["dashboard", "admin", "saas", "analytics", "stats"]):
            always_include += ["dashboard_layout", "stats_cards_component"]

        if any(w in problem_lower for w in ["payment", "stripe", "subscription", "billing"]):
            always_include += ["stripe_payment_backend"]

        if any(w in problem_lower for w in ["real-time", "realtime", "chat", "websocket", "live"]):
            always_include += ["websocket_realtime"]

        if any(w in problem_lower for w in ["upload", "file", "image", "attachment"]):
            always_include += ["file_upload_backend"]

        if any(w in problem_lower for w in ["table", "list", "grid", "data"]):
            always_include += ["data_table_component"]

        critical_examples = [
            e for e in PERFECT_EXAMPLES
            if e['id'] in always_include
        ]

        query_specific = vector_store.retrieve(
            query=problem_statement,
            n_results=5
        )

        context_parts = [
            "=== PERFECT CODE TEMPLATES (MUST FOLLOW EXACTLY) ===\n",
            "These are verified, working templates. Use them as the foundation for every file.\n\n"
        ]

        for example in critical_examples:
            context_parts.append(f"### {example['file_path']}\n")
            context_parts.append(f"# {example['description']}\n")
            if 'important_note' in example:
                context_parts.append(f"# IMPORTANT: {example['important_note']}\n")
            context_parts.append(f"```\n{example['code'].strip()}\n```\n\n")

        context_parts.append("=== PROJECT-SPECIFIC EXAMPLES ===\n\n")

        seen_ids = {e['id'] for e in critical_examples}
        for example in query_specific:
            if example['id'] not in seen_ids:
                context_parts.append(f"### {example['file_path']}\n")
                context_parts.append(f"# {example['description']}\n")
                context_parts.append(f"```\n{example['code'].strip()}\n```\n\n")
                seen_ids.add(example['id'])

        return "".join(context_parts)

    def record_successful_build(self, project_name: str, files: list[dict]):
        """Add successfully built project files to knowledge base."""
        for f in files:
            if f.get('content') and len(f['content']) > 50:
                example_id = f"success_{project_name}_{f['path'].replace('/', '_')}"
                vector_store.add_example({
                    "id": example_id,
                    "description": f"Successfully built {f['path']} for {project_name}",
                    "category": "backend" if "backend" in f['path'] else "frontend",
                    "subcategory": f['path'].split('/')[-1].split('.')[0],
                    "file_path": f['path'],
                    "tags": [project_name, f['path'].split('.')[-1]],
                    "code": f['content']
                })
        logger.info("Recorded %d files from successful build: %s", len(files), project_name)


rag_retriever = RAGRetriever()
