"""Collects training data from every build and debug attempt."""

import json
import re
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.models import BuildAttempt, ErrorPattern, TrainingExample
from src.utils.logger import get_logger
from src.utils.training_data import TRAINING_DATA

logger = get_logger(__name__)

# Known error patterns — ordered most-specific first so detect_error_type() returns
# the best match. More specific TypeScript/StackBlitz/Python patterns come before
# their generic counterparts.
KNOWN_ERROR_PATTERNS = [
    # ── StackBlitz-specific (before generic css/package patterns) ────────────
    {
        "error_type": "stackblitz_missing_css",
        "pattern": r"Import error.*index\.css|can't find file.*index\.css|Import error.*App\.css",
        "fix_strategy": "create_index_css",
        "description": "StackBlitz missing CSS file",
    },
    {
        "error_type": "stackblitz_missing_package",
        "pattern": r"Can't find packages?:\s*\S+",
        "fix_strategy": "add_to_package_json",
        "description": "StackBlitz missing npm package",
    },
    # ── CSS / missing files ──────────────────────────────────────────────────
    {
        "error_type": "missing_css",
        "pattern": r"can't find file.*\.css|Cannot find module.*\.css|import.*\.css.*not found",
        "fix_strategy": "create_index_css",
        "description": "Missing CSS file",
    },
    {
        "error_type": "missing_package",
        "pattern": r"Cannot find module '([^']+)'|can't find packages?: ([^\n]+)",
        "fix_strategy": "add_to_package_json",
        "description": "Missing npm package",
    },
    {
        "error_type": "fake_package",
        "pattern": r"404.*@react-bits|not found.*react-bits|Cannot find.*@react-bits|404 Not Found.*npmjs\.org",
        "fix_strategy": "remove_fake_package",
        "description": "Fake/non-existent npm package",
    },
    {
        "error_type": "wrong_import_path",
        "pattern": r"Module not found.*styles/globals|Cannot find.*styles/index",
        "fix_strategy": "fix_import_path",
        "description": "Wrong CSS import path",
    },
    # ── Python (specific before generic) ────────────────────────────────────
    {
        "error_type": "local_module_false_positive",
        "pattern": r"import '(database|routes|models|schemas|config|auth|crud|deps|middleware|dependencies|core|api|services|utils|helpers|security|serializers|validators|permissions|tasks|repositories)' not found in requirements\.txt",
        "fix_strategy": "skip",
        "description": "Local Python module incorrectly flagged as missing pip package",
    },
    {
        "error_type": "missing_python_package",
        "pattern": r"ModuleNotFoundError: No module named '([^']+)'",
        "fix_strategy": "add_to_requirements",
        "description": "Missing Python package",
    },
    {
        "error_type": "python_import",
        "pattern": r"ImportError: cannot import name|cannot import name '([^']+)'",
        "fix_strategy": "llm_fix",
        "description": "Python import error",
    },
    {
        "error_type": "python_syntax",
        "pattern": r"SyntaxError|IndentationError|NameError|ImportError",
        "fix_strategy": "llm_fix",
        "description": "Python syntax error",
    },
    {
        "error_type": "python_runtime",
        "pattern": r"AttributeError: 'NoneType'|TypeError: object is not|ValueError: too many",
        "fix_strategy": "llm_fix",
        "description": "Python runtime error",
    },
    # ── Missing files ────────────────────────────────────────────────────────
    {
        "error_type": "missing_index_html",
        "pattern": r"public/index\.html.*not found|missing.*index\.html",
        "fix_strategy": "create_index_html",
        "description": "Missing public/index.html",
    },
    {
        "error_type": "missing_env",
        "pattern": r"REACT_APP_.*is not defined|process\.env\..*undefined",
        "fix_strategy": "create_env_file",
        "description": "Missing environment variable",
    },
    # ── Database (specific before generic) ───────────────────────────────────
    {
        "error_type": "db_table_missing",
        "pattern": r"relation '([^']+)' does not exist|Table '([^']+)' doesn't exist|UndefinedTableError",
        "fix_strategy": "run_migrations",
        "description": "Database table not found",
    },
    {
        "error_type": "db_auth",
        "pattern": r"password authentication failed|InvalidPasswordError",
        "fix_strategy": "check_credentials",
        "description": "Database authentication failure",
    },
    {
        "error_type": "db_host",
        "pattern": r"could not translate host name|getaddrinfo.*ENOTFOUND",
        "fix_strategy": "fix_host",
        "description": "Database host resolution failure",
    },
    {
        "error_type": "db_connection",
        "pattern": r"could not connect.*database|connection.*refused.*5432|asyncpg.*connection",
        "fix_strategy": "check_database_url",
        "description": "Database connection error",
    },
    # ── CORS ─────────────────────────────────────────────────────────────────
    {
        "error_type": "cors_preflight",
        "pattern": r"preflight request doesn't pass access control|CORS.*preflight",
        "fix_strategy": "fix_cors",
        "description": "CORS preflight failure",
    },
    {
        "error_type": "cors_error",
        "pattern": r"CORS.*blocked|Access-Control-Allow-Origin|cross-origin",
        "fix_strategy": "fix_cors",
        "description": "CORS configuration error",
    },
    # ── TypeScript (specific before generic) ─────────────────────────────────
    {
        "error_type": "typescript_missing_types",
        "pattern": r"TS7016:.*declaration file for module",
        "fix_strategy": "add_ts_ignore",
        "description": "Missing TypeScript declaration file",
    },
    {
        "error_type": "typescript_missing_import",
        "pattern": r"TS2304: Cannot find name",
        "fix_strategy": "fix_import",
        "description": "Missing TypeScript import",
    },
    {
        "error_type": "typescript_type",
        "pattern": r"TS2339:|TS2345:|TS2322:|TS2531:|TS2532:|TS18048:",
        "fix_strategy": "llm_fix",
        "description": "TypeScript type mismatch",
    },
    {
        "error_type": "typescript_version",
        "pattern": r"TS1139: Type parameter declaration expected|TS2497:",
        "fix_strategy": "add_skip_lib_check",
        "description": "TypeScript version incompatibility",
    },
    {
        "error_type": "typescript_css",
        "pattern": r"TS2561:.*'inset[XY]'|'inset[XY]'.*does not exist",
        "fix_strategy": "fix_css_property",
        "description": "Invalid CSS-in-JS property name",
    },
    {
        "error_type": "typescript_error",
        "pattern": r"TS\d{4}:|TypeScript error|type.*is not assignable",
        "fix_strategy": "fix_typescript",
        "description": "TypeScript type error",
    },
    # ── React ─────────────────────────────────────────────────────────────────
    {
        "error_type": "react_invalid_child",
        "pattern": r"Objects are not valid as a React child|Minified React error",
        "fix_strategy": "llm_fix",
        "description": "Invalid React child type",
    },
    {
        "error_type": "react_undefined",
        "pattern": r"Cannot read properties of undefined \(reading '(?:map|filter|forEach|length|find)'",
        "fix_strategy": "llm_fix",
        "description": "React undefined property access",
    },
    {
        "error_type": "react_key",
        "pattern": r"Each child in a list should have a unique 'key' prop",
        "fix_strategy": "llm_fix",
        "description": "Missing React list key",
    },
    {
        "error_type": "react_cleanup",
        "pattern": r"Can't perform a React state update on an unmounted component",
        "fix_strategy": "llm_fix",
        "description": "React unmounted component state update",
    },
    {
        "error_type": "react_infinite_loop",
        "pattern": r"Too many re-renders",
        "fix_strategy": "llm_fix",
        "description": "React infinite re-render loop",
    },
    # ── FastAPI ───────────────────────────────────────────────────────────────
    {
        "error_type": "fastapi_validation",
        "pattern": r"422 Unprocessable Entity|RequestValidationError",
        "fix_strategy": "llm_fix",
        "description": "FastAPI request validation error",
    },
    {
        "error_type": "fastapi_async",
        "pattern": r"'coroutine' object has no attribute|coroutine was never awaited",
        "fix_strategy": "llm_fix",
        "description": "Missing await in async FastAPI code",
    },
    {
        "error_type": "fastapi_response",
        "pattern": r"ResponseValidationError|Object of type .* is not JSON serializable",
        "fix_strategy": "llm_fix",
        "description": "FastAPI response serialization error",
    },
    {
        "error_type": "fastapi_eventloop",
        "pattern": r"RuntimeError: no running event loop|There is no current event loop",
        "fix_strategy": "llm_fix",
        "description": "No running asyncio event loop",
    },
    # ── SQLAlchemy advanced ───────────────────────────────────────────────────
    {
        "error_type": "sqlalchemy_greenlet",
        "pattern": r"MissingGreenlet|greenlet_spawn has not been called",
        "fix_strategy": "llm_fix",
        "description": "SQLAlchemy async greenlet error",
    },
    {
        "error_type": "sqlalchemy_detached",
        "pattern": r"DetachedInstanceError|Instance.*not bound to a Session",
        "fix_strategy": "llm_fix",
        "description": "SQLAlchemy detached instance",
    },
    {
        "error_type": "sqlalchemy_unique",
        "pattern": r"UniqueViolationError|duplicate key value violates unique constraint",
        "fix_strategy": "llm_fix",
        "description": "Unique constraint violation",
    },
    {
        "error_type": "sqlalchemy_integrity",
        "pattern": r"sqlalchemy\.exc\.IntegrityError|ForeignKeyViolationError|CheckViolationError",
        "fix_strategy": "llm_fix",
        "description": "SQLAlchemy integrity error",
    },
    {
        "error_type": "sqlalchemy_data",
        "pattern": r"sqlalchemy\.exc\.DataError|value too long for type|invalid input syntax for type",
        "fix_strategy": "llm_fix",
        "description": "SQLAlchemy data error",
    },
    {
        "error_type": "sqlalchemy_notnull",
        "pattern": r"NotNullViolationError|null value in column.*violates not-null",
        "fix_strategy": "llm_fix",
        "description": "NOT NULL constraint violation",
    },
    {
        "error_type": "sqlalchemy_lazy",
        "pattern": r"lazy.*load.*async|lazy='dynamic'.*async",
        "fix_strategy": "llm_fix",
        "description": "Lazy loading in async context",
    },
    {
        "error_type": "sqlalchemy_session",
        "pattern": r"Can't operate on a closed Session|transaction has been rolled back due to a previous exception",
        "fix_strategy": "llm_fix",
        "description": "SQLAlchemy session error",
    },
    {
        "error_type": "sqlalchemy_pool",
        "pattern": r"QueuePool limit.*overflow|connection pool.*exhausted|pool.*timed out",
        "fix_strategy": "llm_fix",
        "description": "Connection pool exhausted",
    },
    {
        "error_type": "sqlalchemy_transaction",
        "pattern": r"PendingRollbackError|InFailedSQLTransactionError|transaction is aborted",
        "fix_strategy": "llm_fix",
        "description": "SQLAlchemy transaction error",
    },
    {
        "error_type": "sqlalchemy_n1",
        "pattern": r"N\+1 query|lazy loading triggered.*queries|SELECT.*executed.*times per request",
        "fix_strategy": "llm_fix",
        "description": "N+1 query problem",
    },
    {
        "error_type": "sqlalchemy_cascade",
        "pattern": r"violates foreign key constraint.*still referenced|cascade.*delete.*error",
        "fix_strategy": "llm_fix",
        "description": "Cascade delete error",
    },
    # ── FastAPI advanced ──────────────────────────────────────────────────────
    {
        "error_type": "fastapi_circular_import",
        "pattern": r"partially initialized module.*circular import|circular import.*routers",
        "fix_strategy": "llm_fix",
        "description": "Circular import in FastAPI modules",
    },
    {
        "error_type": "fastapi_dependency",
        "pattern": r"Dependency.*raised an exception|dependency injection.*error|Invalid args for response field",
        "fix_strategy": "llm_fix",
        "description": "FastAPI dependency injection error",
    },
    {
        "error_type": "fastapi_background",
        "pattern": r"Exception in background task|background.*task.*error|BackgroundTasks.*closed",
        "fix_strategy": "llm_fix",
        "description": "FastAPI background task error",
    },
    {
        "error_type": "fastapi_websocket_error",
        "pattern": r"WebSocketDisconnect|Cannot call 'send' once a close message",
        "fix_strategy": "llm_fix",
        "description": "FastAPI WebSocket error",
    },
    {
        "error_type": "fastapi_middleware_error",
        "pattern": r"CORS middleware must be.*outermost|GZipMiddleware.*content-encoding",
        "fix_strategy": "llm_fix",
        "description": "FastAPI middleware order error",
    },
    {
        "error_type": "fastapi_lifespan_error",
        "pattern": r"lifespan.*yields exactly once|ASGI lifespan.*startup",
        "fix_strategy": "llm_fix",
        "description": "FastAPI lifespan error",
    },
    {
        "error_type": "fastapi_router_conflict",
        "pattern": r"Duplicate route path|NoMatchFound.*route.*not included|APIRouter.*path.*conflicts",
        "fix_strategy": "llm_fix",
        "description": "FastAPI router conflict",
    },
    {
        "error_type": "fastapi_duplicate_op",
        "pattern": r"Duplicate operationId|operationId.*used by.*more than one route",
        "fix_strategy": "llm_fix",
        "description": "Duplicate OpenAPI operation ID",
    },
    {
        "error_type": "fastapi_path_param_error",
        "pattern": r"path parameter.*not used in the path|Path parameter.*name mismatch",
        "fix_strategy": "llm_fix",
        "description": "FastAPI path parameter error",
    },
    {
        "error_type": "fastapi_query_type",
        "pattern": r"query parameter.*expected int|value is not a valid.*query parameter",
        "fix_strategy": "llm_fix",
        "description": "FastAPI query parameter type error",
    },
    {
        "error_type": "fastapi_header_error",
        "pattern": r"header.*field required|Authorization header.*missing or malformed",
        "fix_strategy": "llm_fix",
        "description": "FastAPI header validation error",
    },
    {
        "error_type": "fastapi_form_data",
        "pattern": r"Form field.*required.*missing|mixing JSON body and form data",
        "fix_strategy": "llm_fix",
        "description": "FastAPI form data error",
    },
    # ── React advanced ────────────────────────────────────────────────────────
    {
        "error_type": "react_useeffect_loop",
        "pattern": r"Maximum update depth exceeded|useEffect.*infinite loop",
        "fix_strategy": "llm_fix",
        "description": "useEffect infinite loop",
    },
    {
        "error_type": "react_stale_closure",
        "pattern": r"stale closure|stale.*state.*closure",
        "fix_strategy": "llm_fix",
        "description": "React stale closure",
    },
    {
        "error_type": "react_context_missing",
        "pattern": r"useContext.*undefined|Context.*not provided|Provider.*missing",
        "fix_strategy": "llm_fix",
        "description": "React context provider missing",
    },
    {
        "error_type": "react_ref_null",
        "pattern": r"ref\.current is null|Cannot read.*null.*getBoundingClientRect",
        "fix_strategy": "llm_fix",
        "description": "React ref is null",
    },
    {
        "error_type": "react_forwardref",
        "pattern": r"Function components cannot be given refs.*forwardRef|forwardRef.*missing",
        "fix_strategy": "llm_fix",
        "description": "React forwardRef missing",
    },
    {
        "error_type": "react_suspense",
        "pattern": r"component suspended.*no fallback|Suspense.*boundary.*not found|SuspenseException",
        "fix_strategy": "llm_fix",
        "description": "Missing React Suspense boundary",
    },
    {
        "error_type": "react_error_boundary",
        "pattern": r"ErrorBoundary.*not catching|no ErrorBoundary.*catch render crash",
        "fix_strategy": "llm_fix",
        "description": "React error boundary missing",
    },
    {
        "error_type": "react_controlled_input",
        "pattern": r"changing.*uncontrolled.*controlled|changing.*controlled.*uncontrolled|component.*changing an uncontrolled",
        "fix_strategy": "llm_fix",
        "description": "React controlled/uncontrolled input conflict",
    },
    {
        "error_type": "react_defaultprops",
        "pattern": r"defaultProps will be removed|defaultProps.*deprecated.*function component",
        "fix_strategy": "llm_fix",
        "description": "Deprecated React defaultProps",
    },
    # ── Python advanced ───────────────────────────────────────────────────────
    {
        "error_type": "python_coroutine_not_awaited",
        "pattern": r"coroutine '([^']+)' was never awaited|RuntimeWarning.*coroutine.*awaited",
        "fix_strategy": "llm_fix",
        "description": "Coroutine not awaited",
    },
    {
        "error_type": "python_blocking_async",
        "pattern": r"blocking call.*async|time\.sleep.*async|event loop.*blocked",
        "fix_strategy": "llm_fix",
        "description": "Blocking call in async context",
    },
    {
        "error_type": "python_circular_import",
        "pattern": r"partially initialized module.*circular import|circular import.*models",
        "fix_strategy": "llm_fix",
        "description": "Python circular import",
    },
    {
        "error_type": "python_pydantic_v2",
        "pattern": r"'validator' is removed in Pydantic v2|use 'model_dump'|PydanticUserError|PydanticSchemaGenerationError",
        "fix_strategy": "llm_fix",
        "description": "Pydantic v1/v2 incompatibility",
    },
    {
        "error_type": "python_dotenv_order",
        "pattern": r"load_dotenv.*before import|DATABASE_URL.*not loaded.*import time",
        "fix_strategy": "llm_fix",
        "description": "dotenv loaded after import",
    },
    {
        "error_type": "python_encoding",
        "pattern": r"UnicodeDecodeError|UnicodeEncodeError|codec.*can't.*decode|codec.*can't.*encode",
        "fix_strategy": "llm_fix",
        "description": "Python encoding error",
    },
    {
        "error_type": "python_json_serialize",
        "pattern": r"Object of type (?:UUID|datetime|Decimal|bytes) is not JSON serializable",
        "fix_strategy": "llm_fix",
        "description": "JSON serialization error",
    },
    {
        "error_type": "python_datetime_tz",
        "pattern": r"can't compare offset-naive and offset-aware|astimezone.*naive datetime|naive.*aware.*datetime",
        "fix_strategy": "llm_fix",
        "description": "Datetime timezone mismatch",
    },
    {
        "error_type": "python_decimal_float",
        "pattern": r"float.*precision.*money|Decimal.*float.*precision|FloatingPointError",
        "fix_strategy": "llm_fix",
        "description": "Float precision error in financial calc",
    },
    {
        "error_type": "python_bytes_string",
        "pattern": r"bytes-like object is required, not 'str'|'bytes' object has no attribute 'split'",
        "fix_strategy": "llm_fix",
        "description": "Bytes/string type confusion",
    },
    # ── Deployment ────────────────────────────────────────────────────────────
    {
        "error_type": "deploy_port",
        "pattern": r"EADDRINUSE.*PORT|app bound to localhost.*0\.0\.0\.0|failed to listen on port \$PORT",
        "fix_strategy": "fix_port_binding",
        "description": "App not bound to $PORT",
    },
    {
        "error_type": "deploy_database_url",
        "pattern": r"invalid DSN 'postgres://|scheme 'postgres' is not recognized|asyncpg.*requires.*postgresql\+asyncpg",
        "fix_strategy": "fix_database_url_scheme",
        "description": "postgres:// URL must be postgresql+asyncpg://",
    },
    {
        "error_type": "deploy_ssl",
        "pattern": r"SSL connection.*closed unexpectedly|CERTIFICATE_VERIFY_FAILED|SSL required.*server does not support",
        "fix_strategy": "llm_fix",
        "description": "Database SSL required in production",
    },
    {
        "error_type": "deploy_static_files",
        "pattern": r"StaticFiles.*directory.*does not exist|static.*files.*not.*served|React.*blank page.*production",
        "fix_strategy": "llm_fix",
        "description": "Static files not configured",
    },
    {
        "error_type": "deploy_502",
        "pattern": r"502 Bad Gateway|upstream connect error|upstream.*timed out",
        "fix_strategy": "llm_fix",
        "description": "502 Bad Gateway",
    },
    {
        "error_type": "deploy_memory",
        "pattern": r"OOMKilled|JavaScript heap out of memory|MemoryError.*static TLS",
        "fix_strategy": "llm_fix",
        "description": "Deployment memory limit exceeded",
    },
    {
        "error_type": "deploy_timeout",
        "pattern": r"H12 Request Timeout|Gateway Timeout 504|request deadline exceeded",
        "fix_strategy": "llm_fix",
        "description": "Deployment request timeout",
    },
    {
        "error_type": "deploy_build",
        "pattern": r"npm run build.*exit.*code 1|pip install failed.*requirement|build failed.*TypeScript",
        "fix_strategy": "llm_fix",
        "description": "Deployment build failure",
    },
    {
        "error_type": "deploy_env_missing",
        "pattern": r"Missing environment variable|KeyError.*SECRET_KEY|API_KEY.*not found.*production",
        "fix_strategy": "llm_fix",
        "description": "Required env var missing in production",
    },
    # ── Auth / Security ───────────────────────────────────────────────────────
    {
        "error_type": "auth_jwt_expired",
        "pattern": r"ExpiredSignatureError|Signature has expired|JWT.*expired",
        "fix_strategy": "llm_fix",
        "description": "JWT token expired",
    },
    {
        "error_type": "auth_jwt_invalid",
        "pattern": r"InvalidSignatureError|Signature verification failed|DecodeError.*segments",
        "fix_strategy": "llm_fix",
        "description": "JWT signature invalid",
    },
    {
        "error_type": "auth_bcrypt",
        "pattern": r"UnknownHashError.*bcrypt|ValueError: Invalid salt|bcrypt.*checkpw.*failed",
        "fix_strategy": "llm_fix",
        "description": "bcrypt hash error",
    },
    {
        "error_type": "auth_cors_credentials",
        "pattern": r"credentials.*wildcard|Allow-Credentials.*false|withCredentials.*matching origin",
        "fix_strategy": "llm_fix",
        "description": "CORS credentials with wildcard origin",
    },
    {
        "error_type": "auth_cookie",
        "pattern": r"SameSite.*Strict.*cross-site|cookie.*Secure.*HTTP|third-party cookie.*SameSite",
        "fix_strategy": "llm_fix",
        "description": "Cookie SameSite/Secure policy error",
    },
    {
        "error_type": "auth_protected_route",
        "pattern": r"Bearer token not.*sent|auth state not persisted.*reload|protected.*route.*redirect.*login",
        "fix_strategy": "llm_fix",
        "description": "Protected route auth failure",
    },
    {
        "error_type": "auth_rbac",
        "pattern": r"role.*cannot access.*endpoint|RBAC.*check.*failed|Insufficient permissions",
        "fix_strategy": "llm_fix",
        "description": "Role-based access control failure",
    },
    {
        "error_type": "auth_oauth2",
        "pattern": r"state mismatch.*CSRF|invalid_grant.*authorization code|redirect_uri mismatch",
        "fix_strategy": "llm_fix",
        "description": "OAuth2 flow error",
    },
    # ── WebSocket ─────────────────────────────────────────────────────────────
    {
        "error_type": "websocket_refused",
        "pattern": r"WebSocket.*handshake.*Unexpected response code|ERR_CONNECTION_REFUSED.*WebSocket",
        "fix_strategy": "llm_fix",
        "description": "WebSocket connection refused",
    },
    {
        "error_type": "websocket_cors_error",
        "pattern": r"WebSocket.*upgrade.*CORS|socket\.io.*CORS.*not allowed",
        "fix_strategy": "llm_fix",
        "description": "WebSocket CORS error",
    },
    {
        "error_type": "websocket_disconnect",
        "pattern": r"WebSocketDisconnect: code=100[0-9]|abnormal closure.*connection dropped",
        "fix_strategy": "llm_fix",
        "description": "WebSocket abnormal disconnect",
    },
    {
        "error_type": "websocket_memory_leak",
        "pattern": r"WebSocket.*listeners accumulate|socket\.on.*called multiple times.*leak",
        "fix_strategy": "llm_fix",
        "description": "WebSocket listener memory leak",
    },
    {
        "error_type": "websocket_event",
        "pattern": r"socket\.io.*event.*not received|WebSocket.*event name mismatch",
        "fix_strategy": "llm_fix",
        "description": "WebSocket event not received",
    },
    {
        "error_type": "websocket_auth",
        "pattern": r"WebSocket.*authentication failed.*token.*handshake",
        "fix_strategy": "llm_fix",
        "description": "WebSocket authentication failure",
    },
    # ── File upload ───────────────────────────────────────────────────────────
    {
        "error_type": "upload_too_large",
        "pattern": r"413 Request Entity Too Large|MultiPartException.*chunk too large|client_max_body_size exceeded",
        "fix_strategy": "llm_fix",
        "description": "Upload file too large",
    },
    {
        "error_type": "upload_content_type",
        "pattern": r"content type.*not allowed|expected multipart.*received application/octet-stream",
        "fix_strategy": "llm_fix",
        "description": "Upload wrong content type",
    },
    {
        "error_type": "upload_path_traversal",
        "pattern": r"path traversal.*detected|directory traversal.*filename|unsafe filename.*\.\./",
        "fix_strategy": "llm_fix",
        "description": "Path traversal in upload filename",
    },
    {
        "error_type": "upload_save_error",
        "pattern": r"Permission denied.*uploads|OSError.*upload.*directory",
        "fix_strategy": "llm_fix",
        "description": "Upload save permission error",
    },
    {
        "error_type": "upload_missing_dir",
        "pattern": r"upload directory.*does not exist|FileNotFoundError.*uploads",
        "fix_strategy": "create_upload_dir",
        "description": "Upload directory missing",
    },
    {
        "error_type": "upload_validation",
        "pattern": r"file extension.*not in allowed|file.*disguised as|NoCredentialsError.*S3",
        "fix_strategy": "llm_fix",
        "description": "Upload validation error",
    },
    # ── Performance ───────────────────────────────────────────────────────────
    {
        "error_type": "perf_n1_query",
        "pattern": r"N\+1 query|lazy loading triggered.*queries|SELECT.*executed.*times per request",
        "fix_strategy": "llm_fix",
        "description": "N+1 database query problem",
    },
    {
        "error_type": "perf_missing_index",
        "pattern": r"sequential scan.*table.*missing index|EXPLAIN.*Seq Scan|slow.*ORDER BY.*no index",
        "fix_strategy": "llm_fix",
        "description": "Missing database index",
    },
    {
        "error_type": "perf_bundle_size",
        "pattern": r"bundle size exceeds.*KiB|Large bundle.*lodash|moment\.js.*locale.*256KB",
        "fix_strategy": "llm_fix",
        "description": "JavaScript bundle too large",
    },
    {
        "error_type": "perf_memory_leak_react",
        "pattern": r"Memory leak.*component.*subscribe.*WebSocket|setInterval.*useEffect.*not cleared",
        "fix_strategy": "llm_fix",
        "description": "React component memory leak",
    },
    {
        "error_type": "perf_rerender",
        "pattern": r"re-renders per second.*context.*recreated|Excessive re-renders.*missing.*memo",
        "fix_strategy": "llm_fix",
        "description": "Excessive React re-renders",
    },
]

_KNOWN_VERSIONS: dict[str, str] = {
    "react-router-dom": "^6.8.0",
    "axios": "^1.3.0",
    "framer-motion": "^10.0.0",
    "lucide-react": "^0.263.0",
    "zustand": "^4.3.0",
    "recharts": "^2.5.0",
    "date-fns": "^2.29.0",
    "uuid": "^9.0.0",
    "react-hook-form": "^7.43.0",
    "zod": "^3.20.0",
    "clsx": "^1.2.0",
    "react-hot-toast": "^2.4.0",
    "react-toastify": "^9.1.0",
    "react-icons": "^4.7.0",
    "socket.io-client": "^4.6.0",
    "dayjs": "^1.11.0",
    "lodash": "^4.17.0",
    "@tanstack/react-query": "^4.0.0",
    "react-markdown": "^8.0.0",
    "@emotion/react": "^11.10.0",
    "@emotion/styled": "^11.10.0",
    "styled-components": "^5.3.0",
    "chart.js": "^4.2.0",
    "react-chartjs-2": "^5.2.0",
    "@headlessui/react": "^1.7.0",
    "immer": "^9.0.0",
    "swr": "^2.1.0",
    "react-select": "^5.7.0",
    "react-datepicker": "^4.10.0",
    "classnames": "^2.3.0",
    "react-dropzone": "^14.2.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "@types/node": "^18.0.0",
    "@types/uuid": "^9.0.0",
    "@types/lodash": "^4.14.0",
    "react-scripts": "5.0.1",
    "tailwindcss": "^3.3.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "socket.io-client": "^4.6.0",
    "flowbite-react": "^0.6.0",
    "@tabler/icons-react": "^2.30.0",
    "@mui/material": "^5.14.0",
    "@mui/icons-material": "^5.14.0",
    "@emotion/react": "^11.11.0",
    "react-dropzone": "^14.2.0",
    "react-intersection-observer": "^9.5.0",
    "@tanstack/react-table": "^8.9.0",
    "vite": "^4.4.0",
    "@vitejs/plugin-react": "^4.0.0",
}

_FAKE_PACKAGES = [
    "@react-bits/react", "@react-bits/ui", "@react-bits/core", "@react-bits/forms",
    "react-bits", "@shadcn/ui", "tailwindui", "react-awesome-components",
    "@ui/components", "react-styled-kit", "react-pro-kit", "@antfu/components",
    "react-animation-kit", "next-auth-react", "react-ui-components",
    "react-design-system", "@components/core", "pretty-react-hooks",
    "@flowbite/react",   # real package is flowbite-react
    "@tabler/icons",     # real package is @tabler/icons-react
]

_PYTHON_PACKAGE_MAP: dict[str, str] = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "sqlalchemy": "sqlalchemy[asyncio]",
    "asyncpg": "asyncpg",
    "dotenv": "python-dotenv",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "aiofiles": "aiofiles",
    "jose": "python-jose[cryptography]",
    "passlib": "passlib[bcrypt]",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "boto3": "boto3",
    "redis": "redis",
    "celery": "celery",
    "alembic": "alembic",
    "anthropic": "anthropic",
    "openai": "openai",
    "requests": "requests",
    "aiohttp": "aiohttp",
    "starlette": "starlette",
    "bcrypt": "bcrypt",
    "jwt": "pyjwt",
    "yaml": "pyyaml",
    "stripe": "stripe",
    "sendgrid": "sendgrid",
    "twilio": "twilio",
    "motor": "motor",
    "pymongo": "pymongo",
    "aiomysql": "aiomysql",
    "aiosqlite": "aiosqlite",
    "socketio": "python-socketio",
    "websockets": "websockets",
    "multipart": "python-multipart",
    "magic": "python-magic",
    "paramiko": "paramiko",
    "cryptography": "cryptography",
}


class TrainingCollector:
    """Collects and manages training data for NexusFlow improvement."""

    def __init__(self) -> None:
        self.agent_name = "TrainingCollector"
        logger.info("%s initialised.", self.agent_name)

    # ── Pattern detection ─────────────────────────────────────────────────────

    def detect_error_type(self, error_text: str) -> dict | None:
        """Match error text against known patterns. Returns the matched pattern or None."""
        for pattern in KNOWN_ERROR_PATTERNS:
            if re.search(pattern["pattern"], error_text, re.IGNORECASE):
                return pattern
        return None

    # ── Instant fixes ─────────────────────────────────────────────────────────

    def get_instant_fix(
        self, error_type: str, error_text: str, files: list[dict]
    ) -> list[dict] | None:
        """Apply an instant fix for a known error type without calling the LLM.

        Returns the updated file list on success, or None when no instant fix
        is available and the error must be forwarded to the LLM.
        """
        if error_type == "local_module_false_positive":
            return files  # No fix needed — this is a false positive, not a real error

        if error_type == "missing_css":
            has_css = any(f["path"].endswith(".css") for f in files)
            if not has_css:
                files.append({
                    "path": "frontend/src/index.css",
                    "content": "body { margin: 0; font-family: -apple-system, sans-serif; }",
                })
            return files

        if error_type == "wrong_import_path":
            for f in files:
                f["content"] = (
                    f["content"]
                    .replace("./styles/globals.css", "./index.css")
                    .replace("../styles/globals.css", "../index.css")
                    .replace("./styles/index.css", "./index.css")
                )
            return files

        if error_type == "fake_package":
            for f in files:
                if "package.json" in f["path"] and "frontend" in f["path"]:
                    try:
                        pkg = json.loads(f["content"])
                        for p in _FAKE_PACKAGES:
                            pkg.get("dependencies", {}).pop(p, None)
                            pkg.get("devDependencies", {}).pop(p, None)
                        f["content"] = json.dumps(pkg, indent=2)
                    except Exception:
                        pass
            return files

        if error_type == "missing_package":
            match = re.search(
                r"Cannot find module '([^']+)'|can't find packages?: ([^\n,]+)",
                error_text,
                re.IGNORECASE,
            )
            if match:
                raw_name = (match.group(1) or match.group(2) or "").strip()
                parts = raw_name.split("/")
                pkg_name = "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]
                version = _KNOWN_VERSIONS.get(pkg_name, "^latest")

                for f in files:
                    if "package.json" in f["path"] and "frontend" in f["path"]:
                        try:
                            pkg = json.loads(f["content"])
                            pkg.setdefault("dependencies", {})[pkg_name] = version
                            f["content"] = json.dumps(pkg, indent=2)
                        except Exception:
                            pass
            return files

        if error_type == "missing_index_html":
            has_html = any("public/index.html" in f["path"] for f in files)
            if not has_html:
                files.append({
                    "path": "frontend/public/index.html",
                    "content": (
                        '<!DOCTYPE html><html><head><meta charset="utf-8"/>'
                        "<title>App</title></head>"
                        '<body><div id="root"></div></body></html>'
                    ),
                })
            return files

        if error_type == "missing_env":
            has_env = any(f["path"] == "frontend/.env" for f in files)
            if not has_env:
                files.append({
                    "path": "frontend/.env",
                    "content": "REACT_APP_API_URL=http://localhost:8000",
                })
            return files

        if error_type == "missing_python_package":
            match = re.search(r"No module named '([^']+)'", error_text)
            if match:
                raw = match.group(1).split(".")[0]
                req_name = _PYTHON_PACKAGE_MAP.get(raw, raw)
                for f in files:
                    if f["path"] in ("backend/requirements.txt", "requirements.txt"):
                        if req_name not in f["content"]:
                            f["content"] = f["content"].rstrip() + f"\n{req_name}\n"
            return files

        if error_type == "typescript_missing_types":
            match = re.search(r"declaration file for module '([^']+)'", error_text)
            if match:
                module_name = match.group(1)
                for f in files:
                    if "frontend" in f["path"] and f["path"].endswith((".tsx", ".ts")):
                        lines = f["content"].split("\n")
                        new_lines: list[str] = []
                        for line in lines:
                            if f"from '{module_name}'" in line or f'from "{module_name}"' in line:
                                new_lines.append("// @ts-ignore")
                            new_lines.append(line)
                        f["content"] = "\n".join(new_lines)
            return files

        if error_type == "typescript_version":
            for f in files:
                if "tsconfig.json" in f["path"] and "frontend" in f["path"]:
                    try:
                        tsconfig = json.loads(f["content"])
                        tsconfig.setdefault("compilerOptions", {})["skipLibCheck"] = True
                        f["content"] = json.dumps(tsconfig, indent=2)
                    except Exception:
                        pass
            return files

        if error_type == "typescript_css":
            for f in files:
                if "frontend" in f["path"]:
                    f["content"] = (
                        f["content"]
                        .replace("insetX: 0", "left: 0, right: 0")
                        .replace("insetY: 0", "top: 0, bottom: 0")
                    )
            return files

        if error_type == "stackblitz_missing_css":
            return self.get_instant_fix("missing_css", error_text, files)

        if error_type == "stackblitz_missing_package":
            match = re.search(r"Can't find packages?:\s*([^\n,]+)", error_text, re.IGNORECASE)
            if match:
                pkg_name = match.group(1).strip()
                synthetic = f"Cannot find module '{pkg_name}'"
                return self.get_instant_fix("missing_package", synthetic, files)
            return files

        if error_type == "deploy_database_url":
            for f in files:
                if f["path"].endswith(".env") or "config" in f["path"]:
                    f["content"] = re.sub(
                        r"postgres://",
                        "postgresql+asyncpg://",
                        f["content"],
                    )
                    # Also fix plain postgresql:// (no driver) for asyncpg
                    f["content"] = re.sub(
                        r"postgresql://(?!.*\+)",
                        "postgresql+asyncpg://",
                        f["content"],
                    )
            return files

        if error_type == "deploy_port":
            import os as _os
            for f in files:
                path = f["path"]
                if path.endswith(("main.py", "app.py", "server.py", "run.py")):
                    content = f["content"]
                    # Fix uvicorn.run calls that hard-code host/port
                    content = re.sub(
                        r'uvicorn\.run\([^)]*host\s*=\s*["\']localhost["\'][^)]*\)',
                        lambda m: m.group(0).replace("localhost", "0.0.0.0"),
                        content,
                    )
                    content = re.sub(
                        r'uvicorn\.run\([^)]*port\s*=\s*\d+[^)]*\)',
                        lambda m: re.sub(
                            r'port\s*=\s*\d+',
                            'port=int(os.getenv("PORT", 8000))',
                            m.group(0),
                        ),
                        content,
                    )
                    if 'import os' not in content:
                        content = "import os\n" + content
                    f["content"] = content
            return files

        if error_type == "upload_missing_dir":
            upload_dir = "uploads"
            match = re.search(r"FileNotFoundError.*?'([^']*uploads[^']*)'", error_text)
            if match:
                upload_dir = match.group(1)
            # Add directory creation to an existing startup/main file
            startup_created = False
            for f in files:
                if f["path"].endswith(("main.py", "app.py")) and "backend" in f["path"]:
                    if "Path(" not in f["content"] and "mkdir" not in f["content"]:
                        from_line = "from pathlib import Path\n"
                        mkdir_line = f'Path("{upload_dir}").mkdir(parents=True, exist_ok=True)\n'
                        if "from pathlib" not in f["content"]:
                            f["content"] = from_line + f["content"]
                        f["content"] += "\n" + mkdir_line
                    startup_created = True
                    break
            if not startup_created:
                files.append({
                    "path": f"backend/{upload_dir}/.gitkeep",
                    "content": "",
                })
            return files

        if error_type == "add_legacy_peer_deps":
            npmrc_path = ".npmrc"
            for f in files:
                if f["path"].endswith(".npmrc"):
                    if "legacy-peer-deps" not in f["content"]:
                        f["content"] = f["content"].rstrip() + "\nlegacy-peer-deps=true\n"
                    return files
            # No existing .npmrc — create one
            files.append({"path": npmrc_path, "content": "legacy-peer-deps=true\n"})
            return files

        if error_type == "remove_package_from_requirements":
            pkg_to_remove: str | None = None
            # Pattern 1: "watchfiles requires Rust" — word directly before "requires Rust"
            m1 = re.search(r"(\w[\w\-]*)\s+requires\s+Rust", error_text, re.IGNORECASE)
            if m1:
                pkg_to_remove = m1.group(1).strip().lower()
            # Pattern 2: after em-dash separator "— watchfiles"
            if not pkg_to_remove:
                m2 = re.search(r"[—–-]{1,3}\s+(\w[\w\-]*)", error_text)
                if m2:
                    pkg_to_remove = m2.group(1).strip().lower()
            # Pattern 3: scan TRAINING_DATA for a matching remove hint
            if not pkg_to_remove:
                for ex in TRAINING_DATA:
                    if ex.get("fix_action") == "remove_package_from_requirements":
                        ex_err = ex.get("error", "").lower()
                        snippet = error_text.lower()[:120]
                        if snippet in ex_err or ex_err[:120] in snippet:
                            pkg_to_remove = ex.get("package_to_remove")
                            break
            if pkg_to_remove:
                for f in files:
                    if f["path"] in ("backend/requirements.txt", "requirements.txt"):
                        lines = [
                            ln for ln in f["content"].splitlines()
                            if not re.fullmatch(
                                rf"\s*{re.escape(pkg_to_remove)}\s*(?:[>=<!#\[].*)?",
                                ln,
                                re.IGNORECASE,
                            )
                        ]
                        f["content"] = "\n".join(lines) + "\n"
            return files

        if error_type == "replace_py_with_sys_executable":
            for f in files:
                if f["path"].endswith(".py"):
                    content = f["content"]
                    # Replace common Windows-only py invocations in subprocess calls
                    content = re.sub(r"['\"]py['\"],\s*['\"][-\d.]+['\"]", "sys.executable", content)
                    content = re.sub(r"\bsubprocess\.run\(\[?['\"]py['\"]\b", "subprocess.run([sys.executable", content)
                    if "sys.executable" in content and "import sys" not in content:
                        content = "import sys\n" + content
                    f["content"] = content
            return files

        if error_type == "scan_and_add_packages":
            # Collect all bare package names imported across TSX/TS files
            imported: set[str] = set()
            for f in files:
                if f["path"].endswith((".tsx", ".ts", ".js", ".jsx")):
                    for m in re.finditer(
                        r"""^import\s+(?:[^'"]+\s+from\s+)?['"]([^./][^'"]*)['"]\s*;?""",
                        f["content"],
                        re.MULTILINE,
                    ):
                        raw = m.group(1)
                        # Normalise scoped vs bare package name
                        parts = raw.split("/")
                        pkg = "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]
                        imported.add(pkg)

            # Apply to package.json
            for f in files:
                if "package.json" in f["path"] and "frontend" in f["path"]:
                    try:
                        pkg_json = json.loads(f["content"])
                        deps = pkg_json.setdefault("dependencies", {})
                        dev_deps = pkg_json.get("devDependencies", {})
                        builtin = {"react", "react-dom", "react-scripts"}
                        for pkg in imported:
                            if pkg not in deps and pkg not in dev_deps and pkg not in builtin:
                                deps[pkg] = _KNOWN_VERSIONS.get(pkg, "^latest")
                        f["content"] = json.dumps(pkg_json, indent=2)
                    except Exception:
                        pass
            return files

        # Fallback: scan TRAINING_DATA for a matching file-creation or package instant fix
        for example in TRAINING_DATA:
            if not example.get("instant_fix"):
                continue
            ex_error = example.get("error", "").lower()
            if not ex_error:
                continue
            error_lower = error_text.lower()
            if ex_error not in error_lower and error_lower not in ex_error:
                continue
            file_to_fix = example.get("file_to_fix")
            fix_content = example.get("fix_content")
            if file_to_fix and fix_content:
                if not any(f["path"] == file_to_fix for f in files):
                    files.append({"path": file_to_fix, "content": fix_content})
                return files
            pkg_name = example.get("package_to_add")
            version = example.get("version", "^latest")
            if pkg_name:
                for f in files:
                    if "package.json" in f["path"] and "frontend" in f["path"]:
                        try:
                            pkg = json.loads(f["content"])
                            pkg.setdefault("dependencies", {})[pkg_name] = version
                            f["content"] = json.dumps(pkg, indent=2)
                        except Exception:
                            pass
                return files

        return None  # No instant fix — needs LLM

    # ── DB recording ──────────────────────────────────────────────────────────

    async def record_build_attempt(
        self,
        db: AsyncSession,
        project_id: int,
        problem_statement: str,
        errors: list[str],
        fixes: list[str],
        status: str,
        build_time: float,
    ) -> None:
        """Record a build attempt for training data collection."""
        try:
            attempt = BuildAttempt(
                project_id=project_id,
                problem_statement=problem_statement,
                generated_files="[]",
                errors_found=json.dumps(errors),
                fixes_applied=json.dumps(fixes),
                final_status=status,
                build_time_seconds=build_time,
            )
            db.add(attempt)
            await db.commit()
            logger.info(
                "%s recorded build attempt for project %d", self.agent_name, project_id
            )
        except Exception as e:
            logger.warning("%s failed to record build attempt: %s", self.agent_name, e)

    async def record_training_example(
        self,
        db: AsyncSession,
        input_prompt: str,
        error_context: str,
        correct_output: str,
        example_type: str,
        quality_score: float = 1.0,
    ) -> None:
        """Record a high-quality training example."""
        try:
            example = TrainingExample(
                input_prompt=input_prompt,
                error_context=error_context,
                correct_output=correct_output,
                example_type=example_type,
                quality_score=quality_score,
            )
            db.add(example)
            await db.commit()
        except Exception as e:
            logger.warning(
                "%s failed to record training example: %s", self.agent_name, e
            )

    async def update_error_pattern_stats(
        self, db: AsyncSession, error_type: str, success: bool
    ) -> None:
        """Increment success or failure counter for a known error pattern."""
        try:
            result = await db.execute(
                select(ErrorPattern).where(ErrorPattern.error_type == error_type)
            )
            pattern = result.scalar_one_or_none()

            if pattern:
                if success:
                    pattern.success_count += 1
                else:
                    pattern.failure_count += 1
                pattern.last_used = datetime.utcnow()
            else:
                pattern = ErrorPattern(
                    error_type=error_type,
                    error_pattern="",
                    fix_strategy="unknown",
                    success_count=1 if success else 0,
                    failure_count=0 if success else 1,
                )
                db.add(pattern)

            await db.commit()
        except Exception as e:
            logger.warning(
                "%s failed to update pattern stats: %s", self.agent_name, e
            )

    async def get_training_stats(self, db: AsyncSession) -> dict:
        """Return aggregated training data statistics."""
        try:
            all_attempts = (await db.execute(select(BuildAttempt))).scalars().all()
            all_examples = (await db.execute(select(TrainingExample))).scalars().all()
            all_patterns = (await db.execute(select(ErrorPattern))).scalars().all()

            success_count = sum(1 for a in all_attempts if a.final_status == "success")

            return {
                "total_builds": len(all_attempts),
                "successful_builds": success_count,
                "success_rate": (
                    success_count / len(all_attempts) if all_attempts else 0.0
                ),
                "training_examples": len(all_examples),
                "known_patterns": len(all_patterns),
                "top_errors": [
                    {
                        "type": p.error_type,
                        "successes": p.success_count,
                        "failures": p.failure_count,
                    }
                    for p in sorted(
                        all_patterns,
                        key=lambda x: x.success_count + x.failure_count,
                        reverse=True,
                    )[:5]
                ],
            }
        except Exception as e:
            logger.warning("%s failed to get stats: %s", self.agent_name, e)
            return {}


training_collector = TrainingCollector()
