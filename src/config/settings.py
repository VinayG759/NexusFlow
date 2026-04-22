"""NexusFlow application settings.

All values are read from environment variables. A ``.env`` file in the project
root is loaded automatically when the module is imported, so no manual
``load_dotenv()`` call is needed.

Usage::

    from src.config.settings import settings

    print(settings.APP_NAME)
    print(settings.DATABASE_URL)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the NexusFlow application.

    Values are resolved in this order (highest priority first):
    1. Actual environment variables
    2. Variables declared in the ``.env`` file
    3. Defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    # General application identity and runtime mode.

    APP_NAME: str = "NexusFlow"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    # Default directory where NexusFlow saves all generated projects
    OUTPUT_DIRECTORY: str = "C:\\Users\\Priya\\OneDrive\\Desktop\\nexusflowbuiltprojects"

    # ── Database ──────────────────────────────────────────────────────────────
    # PostgreSQL connection settings used by SQLAlchemy.
    # DATABASE_URL must be set in the environment or .env file before startup.
    # Example: postgresql+psycopg2://user:password@localhost:5432/nexusflow

    DATABASE_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ── Agents ────────────────────────────────────────────────────────────────
    # Controls which LLM model agents use by default and how they handle
    # failures and long-running tasks.

    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    MAX_RETRIES: int = 3
    TASK_TIMEOUT: int = 300  # seconds

    # ── Tools ─────────────────────────────────────────────────────────────────
    # API keys and limits for external tools available to agents.
    # WEB_SEARCH_API_KEY should be set in .env and never committed to source control.

    WEB_SEARCH_API_KEY: str = ""
    MAX_SEARCH_RESULTS: int = 10
    # Anthropic Claude API key used by APIConnectorTool to call the LLM.
    # Obtain from https://console.anthropic.com — never commit the real value.
    ANTHROPIC_API_KEY: str = ""
    # Groq API key used by APIConnectorTool.call_groq() for fast LLM inference.
    # Obtain from https://console.groq.com — never commit the real value.
    GROQ_API_KEY: str = ""


# Module-level singleton — import this directly instead of instantiating Settings yourself.
settings = Settings()
