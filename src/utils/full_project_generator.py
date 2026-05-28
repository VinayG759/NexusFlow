"""Full project generator utility for NexusFlow.

Generates an entire project in a SINGLE LLM call — all files are returned at
once in one JSON response, guaranteeing that every import path, type, and
interface is consistent across the whole codebase before any file is written
to disk.

Usage::

    from src.utils.full_project_generator import full_project_generator

    result = await full_project_generator.generate(
        "Build a weather dashboard with a FastAPI backend and React+TypeScript frontend",
        options={"threejs": True, "gsap": False, "reactbits": False},
    )
"""

import json
import os
import time
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.debugging_agent import debugging_agent
from src.agents.file_agent import file_agent
from src.agents.ui_design_agent import ui_design_agent
from src.config.settings import settings
from src.database.models import Project, ProjectFile
from src.rag.rag_retriever import rag_retriever
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger
from src.utils.training_collector import training_collector

logger = get_logger(__name__)

TEMPLATE_FILES = {
    "backend/main.py": """import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="{project_name} API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {{"message": "{project_name} API is running"}}

@app.get("/health")
async def health():
    return {{"status": "ok"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
""",

    "backend/database.py": """import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:vinay2004@localhost:5432/{project_name}"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
""",

    "backend/requirements.txt": """fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
python-dotenv
pydantic
httpx
passlib[bcrypt]
python-jose[cryptography]
python-multipart
""",

    "backend/.env.example": """DATABASE_URL=postgresql+asyncpg://postgres:vinay2004@localhost:5432/{project_name}
""",

    "frontend/src/index.tsx": """import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""",

    "frontend/src/index.css": """@tailwind base;
@tailwind components;
@tailwind utilities;

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
  background-color: #ffffff;
  color: #1a1a1a;
}}

a {{ text-decoration: none; color: inherit; }}
button {{ cursor: pointer; border: none; outline: none; }}
input, textarea, select {{ outline: none; font-family: inherit; }}
""",

    "frontend/tsconfig.json": """{{
  "compilerOptions": {{
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": false,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true
  }},
  "include": ["src"]
}}""",

    "frontend/index.html": """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/index.tsx"></script>
  </body>
</html>""",

    "frontend/vite.config.ts": """import {{ defineConfig }} from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({{
  plugins: [react()],
  server: {{
    port: 5173,
    proxy: {{
      '/api': {{
        target: 'http://localhost:8001',
        changeOrigin: true,
      }}
    }}
  }},
}})
""",

    "frontend/tailwind.config.js": """/** @type {{import('tailwindcss').Config}} */
export default {{
  content: ['./index.html', './src/**/*.{{js,ts,jsx,tsx}}'],
  theme: {{ extend: {{}} }},
  plugins: [],
}}
""",

    "frontend/postcss.config.js": """export default {{
  plugins: {{
    tailwindcss: {{}},
    autoprefixer: {{}},
  }},
}}
""",

    "frontend/src/declarations.d.ts": """declare module '*.css';
declare module '*.svg';
declare module '*.png';
declare module '*.jpg';
""",

    "frontend/src/components/ui.tsx": """import React from 'react';

export function Button({{ children, onClick, variant = 'primary', disabled = false, fullWidth = false }}: {{
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  disabled?: boolean;
  fullWidth?: boolean;
}}) {{
  const variants = {{
    primary: 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg',
    secondary: 'bg-gray-100 hover:bg-gray-200 text-gray-700',
    danger: 'bg-red-500 hover:bg-red-600 text-white',
    ghost: 'bg-transparent hover:bg-gray-100 text-gray-600',
  }};
  return (
    <button onClick={{onClick}} disabled={{disabled}}
      className={{`${{variants[variant]}} ${{fullWidth ? 'w-full' : ''}} font-semibold py-2.5 px-5 rounded-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2`}}>
      {{children}}
    </button>
  );
}}

export function Input({{ label, value, onChange, type = 'text', placeholder = '', error = '' }}: {{
  label: string; value: string; onChange: (v: string) => void;
  type?: string; placeholder?: string; error?: string;
}}) {{
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-gray-700">{{label}}</label>
      <input type={{type}} value={{value}} onChange={{e => onChange(e.target.value)}} placeholder={{placeholder}}
        className={{`border-2 ${{error ? 'border-red-400' : 'border-gray-200'}} rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors bg-white text-gray-900`}} />
      {{error && <span className="text-xs text-red-500">{{error}}</span>}}
    </div>
  );
}}

export function Card({{ children, className = '' }}: {{ children: React.ReactNode; className?: string }}) {{
  return (
    <div className={{`bg-white rounded-2xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow ${{className}}`}}>
      {{children}}
    </div>
  );
}}

export function Badge({{ text, color = 'blue' }}: {{ text: string; color?: string }}) {{
  const colors: Record<string, string> = {{
    blue: 'bg-blue-100 text-blue-700',
    green: 'bg-green-100 text-green-700',
    red: 'bg-red-100 text-red-700',
    yellow: 'bg-yellow-100 text-yellow-700',
    gray: 'bg-gray-100 text-gray-700',
  }};
  return <span className={{`${{colors[color] || colors.blue}} text-xs font-semibold px-2.5 py-1 rounded-full`}}>{{text}}</span>;
}}

export function Spinner() {{
  return <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />;
}}

export function EmptyState({{ title, description, action }}: {{ title: string; description: string; action?: React.ReactNode }}) {{
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="text-5xl mb-4">📭</div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{{title}}</h3>
      <p className="text-gray-500 mb-6 max-w-sm">{{description}}</p>
      {{action}}
    </div>
  );
}}

export function Alert({{ message, type = 'error' }}: {{ message: string; type?: 'error' | 'success' | 'warning' | 'info' }}) {{
  const styles = {{
    error: 'bg-red-50 border-red-200 text-red-700',
    success: 'bg-green-50 border-green-200 text-green-700',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-700',
    info: 'bg-blue-50 border-blue-200 text-blue-700',
  }};
  return <div className={{`${{styles[type]}} border rounded-xl px-4 py-3 text-sm font-medium`}}>{{message}}</div>;
}}
""",

    "frontend/.env": """VITE_API_URL=http://localhost:8001
""",
}

_SYSTEM_PROMPT = """\
You are a senior full-stack software engineer.

Your task is to generate a COMPLETE, production-ready project in a single response.

Return ONLY a valid JSON object — no explanation, no markdown, no text outside the JSON.

The JSON must follow this exact structure:
{
  "project_name": "<lowercase-slug>",
  "description": "<one sentence describing the project>",
  "files": [
    {
      "path": "<relative path from project root>",
      "content": "<complete file content — no markdown fences, no placeholders>"
    }
  ],
  "setup_instructions": "<step-by-step plain text setup guide>",
  "env_variables": [
    {
      "name": "<ENV_VAR_NAME>",
      "description": "<what this variable is for>",
      "example": "<example value>"
    }
  ]
}

Rules you must follow without exception:
1. Every file must have correct imports that reference other files in THIS project by their exact paths.
2. Backend stack: FastAPI + SQLAlchemy (async) + asyncpg + PostgreSQL. Never use Flask, Django, or MongoDB.
3. Frontend stack: React 18 + TypeScript + TailwindCSS. Never use class components or JavaScript.
4. Use async/await throughout all FastAPI route handlers and SQLAlchemy queries.
5. Use SQLAlchemy 2.0 ORM style with Mapped and mapped_column.
6. Use FastAPI lifespan (asynccontextmanager) — never @app.on_event.
7. Always include these files at minimum:
   - backend/main.py
   - backend/database.py
   - backend/models.py
   - backend/routes.py
   - backend/schemas.py
   - backend/requirements.txt
   - frontend/package.json
   - frontend/tsconfig.json
   - frontend/src/App.tsx
   - frontend/src/index.tsx
    - frontend/src/index.css
    - frontend/src/components/ui.tsx
    - frontend/index.html
    - frontend/vite.config.ts
    - frontend/tailwind.config.js
    - frontend/postcss.config.js
    - backend/.env.example
    - frontend/.env
   - docker-compose.yml
   - README.md
8. File content must be complete — no "# TODO", no "..." ellipsis, no placeholder logic.
9. No markdown fences (no ```) anywhere inside file content strings.
10. No hardcoded credentials — use environment variables loaded via python-dotenv or process.env.
11. All TypeScript files must have proper type annotations — no implicit any.
12. The frontend must call the backend API at the URL from VITE_API_URL (Vite env var).
13. Include proper CORS configuration in the FastAPI backend allowing the frontend origin.
14. docker-compose.yml must wire backend, frontend, and PostgreSQL together.
15. For PostgreSQL always use these exact credentials in .env and database.py:
     - username: postgres
     - password: vinay2004
     - host: localhost
     - port: 5432
     - database name: use the project_name slug (e.g. todo-app)
     - DATABASE_URL format: postgresql+asyncpg://postgres:vinay2004@localhost:5432/{project_name}
16. Never use placeholder credentials — always use the exact credentials above.
17. NEVER use fake, invented, or non-existent npm packages. Only use real, published packages from npmjs.com. Verified safe packages include: react, react-dom, react-router-dom, axios, typescript, tailwindcss, framer-motion, lucide-react, three, gsap, @types/react, @types/react-dom, @types/node, vite, @vitejs/plugin-react, zustand, react-query, date-fns, uuid, dotenv, cors, bcryptjs, jsonwebtoken. NEVER use react-scripts — it is EOL and incompatible with Node 22+.
18. Every generated frontend project must have 'skipLibCheck': true in tsconfig.json compilerOptions. This prevents TypeScript errors from third-party type definitions.
19. Every generated frontend tsconfig.json must use Vite-compatible settings:
    - 'target': 'ES2020'
    - 'lib': ['ES2020', 'DOM', 'DOM.Iterable']
    - 'module': 'ESNext'
    - 'moduleResolution': 'bundler'
    - 'allowImportingTsExtensions': true
    - 'skipLibCheck': true
    - 'noEmit': true
    - 'jsx': 'react-jsx'
20. For Three.js imports always add // @ts-ignore comment before the import line:
    // @ts-ignore
    import * as THREE from 'three';

21. CORS Configuration — ALWAYS use this exact pattern in FastAPI:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    NEVER use allow_methods='*' or allow_headers='*' (must be lists, not strings).
    NEVER use allow_credentials=True with allow_origins=['*'].

22. React 18 — ALWAYS use createRoot, NEVER use ReactDOM.render:
    CORRECT:
    import ReactDOM from 'react-dom/client';
    const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
    root.render(<React.StrictMode><App /></React.StrictMode>);

    WRONG (never use):
    ReactDOM.render(<App />, document.getElementById('root'));

23. requirements.txt — ALWAYS include ALL of these:
    fastapi
    uvicorn[standard]
    sqlalchemy[asyncio]
    asyncpg
    python-dotenv
    pydantic
    httpx

24. TypeScript catch blocks — ALWAYS type the catch variable:
    CORRECT: catch (error: unknown) { ... }
    Or: catch (error) { const err = error as Error; ... }
    NEVER: catch (error) { error.message } without typing.

25. tsconfig.json — ALWAYS use this exact Vite-compatible configuration:
    {
      "compilerOptions": {
        "target": "ES2020",
        "useDefineForClassFields": true,
        "lib": ["ES2020", "DOM", "DOM.Iterable"],
        "module": "ESNext",
        "skipLibCheck": true,
        "moduleResolution": "bundler",
        "allowImportingTsExtensions": true,
        "resolveJsonModule": true,
        "isolatedModules": true,
        "noEmit": true,
        "jsx": "react-jsx",
        "strict": false,
        "esModuleInterop": true,
        "allowSyntheticDefaultImports": true
      },
      "include": ["src"]
    }

26. Pydantic v2 — Optional fields MUST have defaults:
    CORRECT: field: Optional[int] = None
    WRONG: field: Optional[int]

27. CSS imports — index.tsx MUST import index.css, not globals.css:
    CORRECT: import './index.css';
    WRONG: import './styles/globals.css';

28. Always create frontend/src/declarations.d.ts with:
    declare module '*.css';
    declare module '*.svg';
    declare module '*.png';
    declare module '*.jpg';

29. Always create frontend/index.html (Vite entry point) at the ROOT of the frontend directory (NOT inside public/). It must reference src/index.tsx with: <script type="module" src="/src/index.tsx"></script>

30. SQLAlchemy async — ALWAYS use this pattern for database.py:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import DeclarativeBase

    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+asyncpg://postgres:vinay2004@localhost:5432/dbname')
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    class Base(DeclarativeBase):
        pass

    async def get_db():
        async with AsyncSessionLocal() as session:
            yield session

    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

31. FastAPI lifespan — ALWAYS use this pattern:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db()
        yield

    app = FastAPI(lifespan=lifespan)

    NEVER use @app.on_event('startup') — it is deprecated.

32. ALWAYS include frontend/src/index.css with basic reset styles.
    ALWAYS import it in index.tsx as: import './index.css'

33. Frontend .env — ALWAYS create with:
    VITE_API_URL=http://localhost:8001

34. Backend .env.example — ALWAYS create with exact values:
    DATABASE_URL=postgresql+asyncpg://postgres:vinay2004@localhost:5432/{project_name}

35. package.json — ALWAYS include react-router-dom and axios in dependencies. ALWAYS include vite, @vitejs/plugin-react, typescript, @types/react, @types/react-dom in devDependencies. NEVER include react-scripts.

36. API URL in frontend — ALWAYS use this exact pattern (Vite uses import.meta.env, NOT process.env):
    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
    NEVER use: process.env.REACT_APP_API_URL — Vite does NOT support process.env.
    NEVER use: axios.post('/calculate') without full URL.
    ALWAYS use: axios.post(`${API_URL}/calculate`)

37. React state initialization — ALWAYS initialize with correct default values:
    CORRECT:
    const [num1, setNum1] = useState<number>(0);
    const [num2, setNum2] = useState<number>(0);
    const [result, setResult] = useState<number | null>(null);
    const [text, setText] = useState<string>('');
    const [items, setItems] = useState<Item[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    NEVER initialize number state as undefined or empty:
    WRONG: useState() or useState(undefined) for numbers.

38. Input onChange handlers — ALWAYS convert to the correct type:
    For number inputs:
    onChange={(e) => setValue(Number(e.target.value) || 0)}
    For text inputs:
    onChange={(e) => setValue(e.target.value)}

39. API calls — ALWAYS handle errors properly:
    try {
        const response = await axios.post(`${API_URL}/endpoint`, payload);
        setResult(response.data);
    } catch (error: unknown) {
        const err = error as any;
        setError(err?.response?.data?.detail || err?.message || 'An error occurred');
    }

40. Backend API routes — ALWAYS match frontend calls exactly:
    If frontend calls: axios.post(`${API_URL}/calculate`)
    Backend must have: @router.post('/calculate')
    If frontend calls: axios.get(`${API_URL}/items`)
    Backend must have: @router.get('/items')

41. FastAPI router — ALWAYS include router in main.py:
    from routes import router
    app.include_router(router)
    NEVER forget to include the router or all endpoints will return 404.

42. Frontend runs on port 5173, backend runs on port 8001.
    VITE_API_URL=http://localhost:8001
    Backend CORS must allow_origins=['*'] to permit frontend access.

43. Database initialization — ALWAYS call init_db() in lifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db()  # Creates all tables
        yield
    NEVER skip init_db() or tables won't exist and queries will fail.

44. Pydantic models — ALWAYS match what the frontend sends:
    If frontend sends: {num1: 5, num2: 3, operation: 'add'}
    Backend must have:
    class CalculateRequest(BaseModel):
        num1: float
        num2: float
        operation: str

    ALWAYS use float not int for numbers to handle decimal inputs.

45. Response format — ALWAYS return consistent JSON:
    CORRECT:
    return {'result': result, 'status': 'success'}
    Frontend expects: response.data.result

    NEVER return plain values:
    WRONG: return result  # Frontend cannot access this properly

46. Pydantic v2 schemas — ALWAYS use model_config, NEVER use class Config:
    CORRECT:
    from pydantic import BaseModel, ConfigDict

    class ItemResponse(BaseModel):
        id: int
        name: str
        model_config = ConfigDict(from_attributes=True)

    WRONG (Pydantic v1 syntax — never use):
    class Config:
        orm_mode = True

47. NEVER mix SQLAlchemy models and Pydantic schemas in the same file.

48. SQLAlchemy models — NEVER use table reflection. NEVER write:
    __table__ = Base.metadata.tables['tablename']
    This crashes at import time because the table doesn't exist in metadata yet.
    ALWAYS define models with __tablename__ and mapped_column:
    class Item(Base):
        __tablename__ = 'items'
        id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
        name: Mapped[str] = mapped_column(String(255))

48b. FastAPI dependency injection — ALWAYS use Depends(get_db) for database sessions in route parameters:
    CORRECT: async def create_item(payload: ItemRequest, db: AsyncSession = Depends(get_db)):
    WRONG: async def create_item(payload: ItemRequest, db: AsyncSession = next(get_db())):
    get_db() is an async generator — next() cannot iterate async generators and will crash at import time.
    ALWAYS import Depends: from fastapi import Depends

49. We use Vite not Create React App:
- Environment variables MUST use VITE_ prefix: VITE_API_URL
- Access with: import.meta.env.VITE_API_URL
- NEVER use process.env.REACT_APP_* with Vite
- const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'
- models.py: ONLY SQLAlchemy ORM classes that inherit from Base
- schemas.py: ONLY Pydantic BaseModel classes for request/response validation
- schemas.py must NEVER import Base or use SQLAlchemy
- models.py must NEVER import BaseModel or use Pydantic
- Always import Base in models.py: from database import Base
- Always import models in routes.py: from models import Todo (or whatever model)

50. MANDATORY UI DESIGN RULES — Every app MUST look professional:

IMPORT UI COMPONENTS in every page:
import { Button, Input, Card, Badge, Spinner, EmptyState, Alert } from '../components/ui';

MANDATORY APP STRUCTURE:
- Full page: min-h-screen bg-gray-50
- Header: sticky top-0 z-10 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm
- Main content: max-w-5xl mx-auto px-4 py-8
- Page title: text-3xl font-bold text-gray-900 mb-2
- Subtitle: text-gray-500 mb-8

MANDATORY STATES:
- Loading: <div className='flex justify-center py-16'><Spinner /></div>
- Error: <Alert message={error} type='error' />
- Empty: <EmptyState title='No items yet' description='Create your first item to get started' action={<Button onClick={handleCreate}>Create New</Button>} />

MANDATORY LIST ITEMS - each item in a styled row:
<div className='bg-white rounded-xl border border-gray-100 p-4 flex items-center justify-between hover:shadow-sm transition-shadow mb-3'>
  <div>
    <h3 className='font-semibold text-gray-900'>{item.title}</h3>
    <p className='text-sm text-gray-500'>{item.description}</p>
  </div>
  <div className='flex gap-2'>
    <Button variant='ghost' onClick={() => handleEdit(item.id)}>Edit</Button>
    <Button variant='danger' onClick={() => handleDelete(item.id)}>Delete</Button>
  </div>
</div>

MANDATORY FORM LAYOUT:
<Card className='max-w-lg'>
  <h2 className='text-xl font-bold text-gray-900 mb-6'>Form Title</h2>
  <div className='space-y-4'>
    <Input label='Field Name' value={value} onChange={setValue} placeholder='Enter value' />
    <Button fullWidth onClick={handleSubmit}>Submit</Button>
  </div>
</Card>

COLOR SYSTEM — use consistently:
- Primary: blue-600
- Success: green-500
- Danger: red-500
- Text: gray-900 (headings), gray-600 (body), gray-400 (muted)
- Backgrounds: white (cards), gray-50 (page), gray-100 (hover)
- Borders: gray-200

NEVER use inline styles.
NEVER use unstyled HTML.
ALWAYS use TailwindCSS classes. TailwindCSS is installed via npm (tailwindcss + postcss + autoprefixer) — do NOT use the CDN.

CRITICAL: Every JSX element MUST have className with TailwindCSS classes.
NEVER generate plain HTML without className.
WRONG:   <div><h1>Title</h1><p>Body</p></div>
CORRECT: <div className="min-h-screen bg-gray-50"><h1 className="text-3xl font-bold text-gray-900">Title</h1><p className="text-gray-600">Body</p></div>

51. App.tsx MUST import and use React Router for navigation:
import { BrowserRouter, Routes, Route, Link, useNavigate } from 'react-router-dom';

ALWAYS wrap app in BrowserRouter.
ALWAYS have at minimum these routes:
- / (home/landing or main feature)
- /about or second feature page

Navigation bar MUST use Link components not <a> tags.

52. For SaaS applications ALWAYS generate these files:
Backend: main.py, database.py, models.py, routes.py, schemas.py, auth.py, requirements.txt, .env.example
Frontend: src/App.tsx, src/index.tsx, src/index.css, src/pages/Login.tsx, src/pages/Register.tsx, src/pages/Dashboard.tsx, src/layouts/DashboardLayout.tsx, src/contexts/AuthContext.tsx, vite.config.ts, package.json, tsconfig.json, public/index.html

Auth file checklist (ALL required when auth is included):
- frontend/src/pages/Login.tsx (REQUIRED)
- frontend/src/pages/Register.tsx (REQUIRED when auth is included)
- frontend/src/contexts/AuthContext.tsx (REQUIRED)
- backend/auth.py (REQUIRED)

When auth feature is detected, requirements.txt MUST include:
fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
python-dotenv
pydantic
httpx
passlib[bcrypt]
python-jose[cryptography]
python-multipart

For simple apps ALWAYS generate:
Backend: main.py, database.py, models.py, routes.py, schemas.py, requirements.txt, .env.example
Frontend: src/App.tsx, src/index.tsx, src/index.css, vite.config.ts, package.json, tsconfig.json, public/index.html

53. Multi-page React apps MUST use React Router:
- App.tsx wraps everything in <BrowserRouter>
- Each feature gets its own page component in src/pages/
- Navigation uses <Link> not <a>
- Protected routes redirect to /login if not authenticated
- Always import: import { BrowserRouter, Routes, Route, Link, useNavigate } from 'react-router-dom'

The / route MUST always show content. NEVER leave / route empty or blank.
If app has auth: / should redirect to /dashboard if logged in, else redirect to /login.
If app has no auth: / is the main feature page (list, dashboard, home screen).
A blank / route will cause a blank Vercel deployment — this is a critical error.

NEVER generate empty components like:
const Home = () => <div />;
const Home = () => null;
const Home = () => <></>;

Every component MUST have meaningful content with TailwindCSS styling.

54. NEVER add # comments to JSON, TypeScript, TSX, HTML, or CSS files.
JSON, HTML, TypeScript, and CSS do not support # comments.
WRONG: # Complete package.json\n{{ "name": "app" }}
WRONG: # Main App component\nimport React from 'react';
CORRECT: Start JSON files with {{ or [ directly.
CORRECT: Start TypeScript/TSX files with import or const directly.
CORRECT: Start HTML files with <!DOCTYPE html> directly.

Pydantic import rule: NEVER import lowercase 'field' from pydantic.
It does not exist. Use 'Field' (capital F) if you need field defaults.
WRONG: from pydantic import BaseModel, field, ConfigDict
CORRECT: from pydantic import BaseModel, Field, ConfigDict

55. Frontend completeness — The initial frontend MUST implement all key screens and flows described in the problem statement.
    Use pages and layouts where appropriate, include loading/error/empty states, and avoid placeholder UIs.
"""


class FullProjectGenerator:
    """Generates a complete full-stack project in a single LLM call.

    Unlike :class:`~src.utils.project_generator.ProjectGenerator` which makes
    one LLM call per file, :class:`FullProjectGenerator` generates every file
    in a single call. This guarantees that all imports, type definitions, API
    contracts, and environment variables are internally consistent before
    anything is written to disk.

    Attributes:
        agent_name: Human-readable identifier used in logs and result dicts.
    """

    def __init__(self, agent_name: str = "FullProjectGenerator") -> None:
        """Initialise the FullProjectGenerator.

        Args:
            agent_name: Name used in log messages and result dicts.
        """
        self.agent_name = agent_name
        logger.info("FullProjectGenerator '%s' initialised.", agent_name)

    # ── Public methods ────────────────────────────────────────────────────────

    async def generate(
        self,
        problem_statement: str,
        options: dict | None = None,
        db: AsyncSession | None = None,
        clarifying_answers: dict[str, str] | None = None,
    ) -> dict:
        """Generate a complete project from a problem statement in one LLM call.

        Builds a user prompt from *problem_statement* and any enabled feature
        flags in *options*, then sends a single request to the Groq LLM with a
        strict system prompt that demands a complete JSON project structure.

        The returned JSON is parsed and every ``files`` entry is saved to disk
        via :func:`~src.agents.file_agent.FileAgent.create_project_file`.

        Args:
            problem_statement: Plain-language description of the project to
                build (e.g. ``"Weather dashboard with live OpenWeatherMap data"``).
            options: Optional dict of feature flags and settings:
                - ``threejs`` (bool): append a Three.js 3D visuals requirement.
                - ``gsap`` (bool): append a GSAP animation requirement.
                - ``reactbits`` (bool): append a ReactBits UI component requirement.
                - ``output_directory`` (str): base directory to save the project
                  under. Files are written to ``output_directory/project_name/path``.
                  When omitted, files are saved to ``project_name/path`` relative to
                  the NexusFlow working directory.

        Returns:
            On success or partial success::

                {
                    "status":             "success" | "partial" | "error",
                    "project_name":       str,
                    "output_path":        str,   # actual base path used for saving
                    "files_saved":        list[str],   # paths written to disk
                    "files_failed":       list[str],   # paths that failed to save
                    "setup_instructions": str,
                    "env_variables":      list[dict],  # [{name, description, example}]
                }

            On total failure (LLM error or JSON parse failure)::

                {
                    "status":       "error",
                    "project_name": "",
                    "files_saved":  [],
                    "files_failed": [],
                    "setup_instructions": "",
                    "env_variables": [],
                    "error":        str,
                }
        """
        opts = options or {}
        build_start_time = time.time()
        logger.info(
            "[%s] Starting full project generation for: %r options=%s",
            self.agent_name, problem_statement[:80], opts,
        )

        # Retrieve RAG context — verified templates injected before the task
        try:
            rag_context = rag_retriever.get_context_for_project(problem_statement)
            logger.info("[%s] RAG context retrieved (%d chars)", self.agent_name, len(rag_context))
            # Cap to avoid Groq 413 "request too large" — ChromaDB returns more context
            # as the knowledge base grows; keep only the most relevant portion.
            _RAG_MAX_CHARS = 4000
            if len(rag_context) > _RAG_MAX_CHARS:
                rag_context = rag_context[:_RAG_MAX_CHARS]
                logger.warning("[%s] RAG context capped at %d chars to prevent 413", self.agent_name, _RAG_MAX_CHARS)
        except Exception as rag_exc:
            logger.warning("[%s] RAG retrieval failed (non-critical): %s", self.agent_name, rag_exc)
            rag_context = ""

        user_prompt = self._build_user_prompt(problem_statement, opts, rag_context, clarifying_answers or {})

        # ── Single LLM call ───────────────────────────────────────────────────
        logger.info("[%s] Sending single LLM call for complete project.", self.agent_name)
        llm_result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=6200,
        )

        _model_used = llm_result.get("model_used") or llm_result.get("model", "llama-3.3-70b-versatile")
        _is_fallback = _model_used != "llama-3.3-70b-versatile"
        logger.info("[%s] LLM generation used model: %r%s", self.agent_name, _model_used,
                    " (fallback)" if _is_fallback else "")

        if llm_result["status"] != "success":
            error_msg = llm_result.get("error", "LLM call returned non-success status.")
            logger.error("[%s] LLM call failed: %s", self.agent_name, error_msg)
            return {
                "status": "error",
                "project_name": "",
                "files_saved": [],
                "files_failed": [],
                "setup_instructions": "",
                "env_variables": [],
                "error": error_msg,
            }

        # ── Parse JSON response ───────────────────────────────────────────────
        try:
            clean_raw = self._clean_json(llm_result["content"])
            project = json.loads(clean_raw)
        except json.JSONDecodeError as exc:
            logger.error("[%s] Failed to parse LLM JSON response: %s", self.agent_name, exc)
            return {
                "status": "error",
                "project_name": "",
                "files_saved": [],
                "files_failed": [],
                "setup_instructions": "",
                "env_variables": [],
                "error": f"JSON parse error: {exc}",
            }

        project_name = project.get("project_name", "generated-project")
        description: str = project.get("description", "")
        files: list[dict] = project.get("files", [])
        setup_instructions: str = project.get("setup_instructions", "")
        env_variables: list[dict] = project.get("env_variables", [])

        # ── Hybrid generation: apply templates for boilerplate files ──────────
        pname_slug = project_name.replace("-", "_")
        template_applied: list[str] = []

        for i, f in enumerate(files):
            path = f.get("path", "")
            if path in TEMPLATE_FILES:
                files[i] = {
                    "path": path,
                    "content": TEMPLATE_FILES[path].format(project_name=pname_slug),
                }
                template_applied.append(path)
                logger.info("[%s] Applied template for: %s", self.agent_name, path)

        existing_paths = {f.get("path", "") for f in files}
        for template_path, template_content in TEMPLATE_FILES.items():
            if template_path not in existing_paths:
                files.append({
                    "path": template_path,
                    "content": template_content.format(project_name=pname_slug),
                })
                logger.info("[%s] Added missing template: %s", self.agent_name, template_path)

        for i, f in enumerate(files):
            if f.get("path") == "frontend/package.json":
                try:
                    pkg = json.loads(f["content"])
                    pkg.setdefault("dependencies", {})
                    pkg.setdefault("devDependencies", {})
                    # Remove CRA/react-scripts (incompatible with Node 22+)
                    pkg["dependencies"].pop("react-scripts", None)
                    for fake_pkg in ["@react-bits/react", "@react-bits/ui", "react-bits"]:
                        pkg["dependencies"].pop(fake_pkg, None)
                    runtime_deps = {
                        "react": "^18.0.0",
                        "react-dom": "^18.0.0",
                        "react-router-dom": "^6.8.0",
                        "axios": "^1.3.0",
                    }
                    dev_deps = {
                        "vite": "^5.0.0",
                        "@vitejs/plugin-react": "^4.0.0",
                        "typescript": "^5.0.0",
                        "@types/react": "^18.0.0",
                        "@types/react-dom": "^18.0.0",
                        "@types/node": "^18.0.0",
                        "tailwindcss": "^3.4.0",
                        "autoprefixer": "^10.4.0",
                        "postcss": "^8.4.0",
                    }
                    for dep, ver in runtime_deps.items():
                        pkg["dependencies"].setdefault(dep, ver)
                    for dep, ver in dev_deps.items():
                        pkg["devDependencies"].setdefault(dep, ver)
                        pkg["dependencies"].pop(dep, None)  # move build tools out of deps
                    pkg["scripts"] = {
                        "dev": "vite",
                        "start": "vite",
                        "build": "vite build",
                        "preview": "vite preview",
                    }
                    files[i]["content"] = json.dumps(pkg, indent=2)
                except Exception as e:
                    logger.warning("[%s] package.json fix failed: %s", self.agent_name, e)
                break

        # Ensure App.tsx exists — index.tsx always imports './App'
        if not any(f.get("path") in ("frontend/src/App.tsx", "frontend/src/app.tsx") for f in files):
            files.append({
                "path": "frontend/src/App.tsx",
                "content": (
                    "import React from 'react';\n\n"
                    "function App() {\n"
                    "  return (\n"
                    "    <div className=\"min-h-screen bg-gray-50 flex items-center justify-center\">\n"
                    "      <div className=\"bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center\">\n"
                    "        <h1 className=\"text-3xl font-bold text-gray-900 mb-2\">App is running!</h1>\n"
                    "        <p className=\"text-gray-500\">Your application is ready.</p>\n"
                    "      </div>\n"
                    "    </div>\n"
                    "  );\n"
                    "}\n\n"
                    "export default App;\n"
                ),
            })
            logger.info("[%s] Added default App.tsx (was missing from LLM output)", self.agent_name)

        logger.info(
            "[%s] Hybrid generation: %d template(s) applied, %d LLM-generated file(s).",
            self.agent_name, len(template_applied), len(files) - len(template_applied),
        )

        # ── Build processed file list (in-memory tsconfig patch + README) ─────
        processed: list[dict] = []

        for file_entry in files:
            path: str = file_entry.get("path", "").strip()
            content: str = self._clean_content(file_entry.get("content", ""))
            if not path:
                logger.warning("[%s] Skipping file entry with empty path.", self.agent_name)
                continue
            # Patch tsconfig.json in-memory to ensure skipLibCheck
            if path.endswith("tsconfig.json") and "node_modules" not in path:
                try:
                    tsconfig_data = json.loads(content)
                    if "compilerOptions" not in tsconfig_data:
                        tsconfig_data["compilerOptions"] = {}
                    tsconfig_data["compilerOptions"]["skipLibCheck"] = True
                    content = json.dumps(tsconfig_data, indent=2)
                except Exception:
                    pass
            processed.append({"path": path, "content": content})

        # ── UI Design Agent — enhance frontend if ANTHROPIC_API_KEY is set ──────
        frontend_files = [f for f in processed if ui_design_agent.is_frontend_file(f.get("path", ""))]
        backend_files  = [f for f in processed if not ui_design_agent.is_frontend_file(f.get("path", ""))]

        if os.getenv("ANTHROPIC_API_KEY") and frontend_files:
            logger.info("[%s] Running UIDesignAgent to enhance frontend...", self.agent_name)
            design_result = await ui_design_agent.enhance_frontend(
                project_name=project_name,
                problem_statement=problem_statement,
                existing_frontend_files=frontend_files,
                reference_context=opts.get("reference_context", ""),
            )
            if design_result["status"] == "success":
                frontend_files = design_result["files"]
                logger.info("[%s] Frontend enhanced by UIDesignAgent.", self.agent_name)
            else:
                logger.warning("[%s] UIDesignAgent failed, using Groq frontend.", self.agent_name)

        processed = backend_files + frontend_files

        # Always generate a detailed README via a separate LLM call, replacing any generic one.
        file_paths = [f["path"] for f in processed if f["path"].lower() != "readme.md"]
        readme_content = await self._generate_readme(
            project_name=project_name,
            file_list=file_paths,
            env_variables=env_variables,
        )
        processed = [f for f in processed if f["path"].lower() != "readme.md"]
        processed.append({"path": "README.md", "content": readme_content})

        logger.info(
            "[%s] LLM returned project=%r with %d file(s); LLM-generated README added.",
            self.agent_name, project_name, len(files),
        )

        # ── Debugging Agent — autonomous 5-phase fix pipeline ────────────────
        debug_fixes_applied = 0
        debug_remaining_errors: list[str] = []
        debug_attempts = 0
        debug_fix_summary = ""
        backend_verified = False
        frontend_verified = False

        logger.info("[%s] Running DebuggingAgent (5-phase)...", self.agent_name)
        debug_result = await debugging_agent.debug_project(
            project_files=processed,
            project_name=project_name,
            db_session=db,
        )
        if debug_result["status"] in ("success", "partial"):
            processed = debug_result["fixed_files"]
            debug_fixes_applied = debug_result.get("fixes_applied_count", 0)
            debug_remaining_errors = debug_result.get("remaining_errors", [])
            debug_attempts = debug_result.get("attempts", 0)
            debug_fix_summary = debug_result.get("fix_summary", "")
            backend_verified = debug_result.get("backend_status") == "running"
            frontend_verified = debug_result.get("frontend_status") == "built"
            logger.info(
                "[%s] DebuggingAgent: %d fix(es), %d remaining — backend=%s frontend=%s",
                self.agent_name, debug_fixes_applied, len(debug_remaining_errors),
                debug_result.get("backend_status"), debug_result.get("frontend_status"),
            )

        # ── Record successful build to RAG knowledge base ─────────────────────
        try:
            rag_retriever.record_successful_build(project_name, processed)
        except Exception as rag_exc:
            logger.warning("[%s] RAG record failed (non-critical): %s", self.agent_name, rag_exc)

        # ── Save to database (preferred) ──────────────────────────────────────
        if db is not None:
            db_result = await self._save_to_db(
                db=db,
                project_name=project_name,
                description=description,
                problem_statement=problem_statement,
                setup_instructions=setup_instructions,
                env_variables=env_variables,
                processed=processed,
            )
            if _is_fallback:
                db_result["generated_with"] = "fallback_model"
                db_result["model_used"] = _model_used
            db_result["debug_fixes_applied"] = debug_fixes_applied
            db_result["debug_remaining_errors"] = debug_remaining_errors
            db_result["debug_attempts"] = debug_attempts
            db_result["debug_fix_summary"] = debug_fix_summary
            db_result["backend_verified"] = backend_verified
            db_result["frontend_verified"] = frontend_verified

            # ── Record training data ──────────────────────────────────────────
            try:
                logger.info(
                    "[%s] debug_result: backend_status=%r frontend_status=%r",
                    self.agent_name,
                    debug_result.get("backend_status"),
                    debug_result.get("frontend_status"),
                )
                backend_ok = backend_verified
                frontend_ok = frontend_verified
                build_status = "success" if backend_ok and frontend_ok else "partial" if backend_ok else "failed"
                await training_collector.record_build_attempt(
                    db=db,
                    project_id=db_result.get("project_id"),
                    problem_statement=problem_statement,
                    errors=debug_result.get("remaining_errors", []),
                    fixes=debug_result.get("fixes_applied_list", []),
                    status=build_status,
                    build_time=time.time() - build_start_time,
                )
                if backend_ok and frontend_ok:
                    files_summary = "\n".join([f["path"] for f in processed[:10]])
                    await training_collector.record_training_example(
                        db=db,
                        input_prompt=problem_statement,
                        error_context="successful_build",
                        correct_output=files_summary,
                        example_type="successful_build",
                        quality_score=1.0,
                    )
            except Exception as e:
                logger.warning("Failed to record training data: %s", e)
            return db_result

        # ── Fallback: save to filesystem ──────────────────────────────────────
        output_base: str = opts.get("output_directory", "").strip() or settings.OUTPUT_DIRECTORY
        base_path = str(Path(output_base) / project_name)
        Path(base_path).mkdir(parents=True, exist_ok=True)

        files_saved: list[str] = []
        files_failed: list[str] = []

        for file_entry in processed:
            save_path = f"{base_path}/{file_entry['path']}"
            save_result = file_agent.create_project_file(save_path, file_entry["content"])
            if save_result.get("status") == "success":
                files_saved.append(save_path)
            else:
                files_failed.append(save_path)
                logger.warning("[%s] Failed to save %r: %s", self.agent_name, save_path, save_result.get("error"))

        overall_status = "error" if (files_failed and not files_saved) else ("partial" if files_failed else "success")
        logger.info(
            "[%s] Generation complete (filesystem) — status=%s saved=%d failed=%d",
            self.agent_name, overall_status, len(files_saved), len(files_failed),
        )
        return {
            "status": overall_status,
            "project_name": project_name,
            "actual_output_path": base_path,
            "files_saved": files_saved,
            "files_failed": files_failed,
            "setup_instructions": setup_instructions,
            "env_variables": env_variables,
            "debug_fixes_applied": debug_fixes_applied,
            "debug_remaining_errors": debug_remaining_errors,
            "debug_attempts": debug_attempts,
            "debug_fix_summary": debug_fix_summary,
            "backend_verified": backend_verified,
            "frontend_verified": frontend_verified,
        }

    async def _save_to_db(
        self,
        db: AsyncSession,
        project_name: str,
        description: str,
        problem_statement: str,
        setup_instructions: str,
        env_variables: list[dict],
        processed: list[dict],
    ) -> dict:
        """Persist the generated project to the database and return the project ID."""
        try:
            project_record = Project(
                name=project_name,
                description=description,
                problem_statement=problem_statement,
                status="ready",
                setup_instructions=setup_instructions,
                tech_stack="FastAPI · React · PostgreSQL",
            )
            db.add(project_record)
            await db.flush()  # assigns project_record.id

            for file_entry in processed:
                ext = Path(file_entry["path"]).suffix.lower()
                file_type = _ext_to_type(ext)
                db.add(ProjectFile(
                    project_id=project_record.id,
                    file_path=file_entry["path"],
                    content=file_entry["content"],
                    file_type=file_type,
                ))

            await db.commit()
            await db.refresh(project_record)

            logger.info(
                "[%s] Saved project id=%d name=%r with %d file(s) to DB.",
                self.agent_name, project_record.id, project_name, len(processed),
            )
            return {
                "status": "success",
                "project_id": project_record.id,
                "project_name": project_name,
                "total_files": len(processed),
                "setup_instructions": setup_instructions,
                "env_variables": env_variables,
            }
        except Exception as exc:
            await db.rollback()
            logger.exception("[%s] DB save failed: %s", self.agent_name, exc)
            return {
                "status": "error",
                "project_name": project_name,
                "error": str(exc),
                "total_files": 0,
                "setup_instructions": setup_instructions,
                "env_variables": env_variables,
            }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_project_type(self, problem_statement: str) -> dict:
        """Analyse the problem statement and return project type + required features."""
        problem_lower = problem_statement.lower()

        project_type = "simple"
        features = []

        if any(w in problem_lower for w in ["saas", "subscription", "billing", "multi-tenant", "team"]):
            project_type = "saas"
            features += ["auth", "dashboard", "payments", "settings"]

        if any(w in problem_lower for w in ["auth", "login", "register", "user", "account", "profile", "password"]):
            features.append("auth")
            if project_type == "simple":
                project_type = "auth_app"

        if any(w in problem_lower for w in ["dashboard", "admin", "analytics", "stats", "chart", "report"]):
            features.append("dashboard")

        if any(w in problem_lower for w in ["payment", "stripe", "subscription", "billing", "price", "plan"]):
            features.append("payments")

        if any(w in problem_lower for w in ["real-time", "realtime", "chat", "live", "websocket", "notification"]):
            features.append("realtime")

        if any(w in problem_lower for w in ["upload", "file", "image", "photo", "attachment", "document"]):
            features.append("file_upload")

        features = list(set(features))

        return {"type": project_type, "features": features}

    def _build_user_prompt(
        self,
        problem_statement: str,
        options: dict,
        rag_context: str = "",
        clarifying_answers: dict[str, str] | None = None,
    ) -> str:
        """Compose the user-facing prompt from the problem statement and options.

        Args:
            problem_statement: Core project description.
            options: Feature flags — ``threejs``, ``gsap``, ``reactbits``.
            rag_context: Pre-built RAG context string with verified templates.
            clarifying_answers: Answers from the /clarify endpoint keyed by
                question id (e.g. {"auth_type": "JWT tokens", "db": "PostgreSQL"}).

        Returns:
            Formatted prompt string ready to send to the LLM.
        """
        lines: list[str] = []

        if rag_context:
            lines.append(rag_context)
            lines.append("\n=== YOUR TASK ===\n")

        lines.append(f"Project requirement: {problem_statement}")

        if clarifying_answers:
            lines.append("\n=== USER REQUIREMENTS (from clarifying questions) ===")
            for q_id, answer in clarifying_answers.items():
                lines.append(f"  - {q_id}: {answer}")
            lines.append("")

        extras: list[str] = []
        if options.get("threejs"):
            extras.append("Use Three.js for 3D visuals in the frontend.")
        if options.get("gsap"):
            extras.append("Use GSAP for animations in the frontend.")
        if options.get("reactbits"):
            extras.append("Use ReactBits for UI components in the frontend.")
        if extras:
            lines.append("Additional requirements:")
            lines.extend(f"  - {e}" for e in extras)

        project_info = self._detect_project_type(problem_statement)
        if project_info["features"]:
            feature_instructions = (
                f"\n=== PROJECT TYPE: {project_info['type'].upper()} ===\n"
                f"Required features to implement: {', '.join(project_info['features'])}\n"
                + ("\nINCLUDE FULL JWT AUTH SYSTEM: Register, Login, Protected routes, Token storage" if "auth" in project_info["features"] else "")
                + ("\nINCLUDE DASHBOARD LAYOUT: Sidebar navigation, Stats cards, Data tables" if "dashboard" in project_info["features"] else "")
                + ("\nINCLUDE STRIPE PAYMENTS: Checkout session, Webhook handler, Subscription status" if "payments" in project_info["features"] else "")
                + ("\nINCLUDE WEBSOCKET: Real-time updates, Connection manager, Room-based messaging" if "realtime" in project_info["features"] else "")
                + ("\nINCLUDE FILE UPLOAD: Multipart form, File validation, Storage handling" if "file_upload" in project_info["features"] else "")
            )
            lines.append(feature_instructions)

        lines.append(
            "\nGenerate the complete project now. Return only the JSON object — nothing else."
        )
        return "\n".join(lines)

    def _clean_json(self, raw: str) -> str:
        """Strip markdown fences and sanitize control characters in an LLM JSON response."""
        raw = raw.strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        return self._sanitize_json_control_chars(raw)

    @staticmethod
    def _sanitize_json_control_chars(raw: str) -> str:
        """Escape literal control characters inside JSON string values.

        Groq and other LLMs sometimes embed raw newlines/tabs inside JSON string
        values instead of the required \\n / \\t escape sequences, causing
        json.loads to raise 'Invalid control character'.  This method walks the
        text character-by-character, tracks whether the cursor is inside a JSON
        string, and replaces any bare control character (U+0000–U+001F) with its
        proper JSON escape — leaving structural whitespace outside strings alone.
        """
        _ESCAPE: dict[str, str] = {
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            "\b": "\\b",
            "\f": "\\f",
        }
        result: list[str] = []
        in_string = False
        escape_next = False

        for ch in raw:
            if escape_next:
                result.append(ch)
                escape_next = False
            elif ch == "\\" and in_string:
                result.append(ch)
                escape_next = True
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
            elif in_string and ord(ch) < 0x20:
                result.append(_ESCAPE.get(ch, f"\\u{ord(ch):04x}"))
            else:
                result.append(ch)

        return "".join(result)

    async def _generate_readme(
        self,
        project_name: str,
        file_list: list[str],
        env_variables: list[dict],
    ) -> str:
        """Generate a detailed README via a separate LLM call.

        Falls back to :meth:`_build_readme` if the LLM call fails.
        """
        slug = project_name.lower().replace(" ", "-").replace("_", "-")
        file_tree = "\n".join(f"  {p}" for p in file_list)
        env_lines = "\n".join(
            f"  {ev.get('name', '')}: {ev.get('description', '')} (e.g. {ev.get('example', '')})"
            for ev in env_variables
        ) or "  (none specified)"

        system_prompt = (
            "You are a technical writer. Generate a detailed, accurate README.md for this project. "
            "Include EXACT commands, EXACT file paths, EXACT environment variable names. "
            "No generic placeholders. Every command must be copy-pasteable and work."
        )

        user_prompt = f"""Generate a detailed README.md for this project:
Project name: {project_name}
Tech stack: FastAPI backend, React TypeScript frontend, PostgreSQL database
Files generated:
{file_tree}
Environment variables needed:
{env_lines}

The README must include these exact sections:

# {project_name}

## Prerequisites
List exact versions needed based on requirements.txt and package.json

## Project Structure
Show the actual file tree

## Backend Setup
### 1. Create virtual environment
python -m venv venv

### 2. Activate virtual environment
Windows: venv\\Scripts\\activate
Mac/Linux: source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Create PostgreSQL database
createdb {slug}
OR using psql:
psql -U postgres -c 'CREATE DATABASE {slug};'

### 5. Configure environment variables
Create backend/.env file:
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/{slug}
(list ALL env variables with descriptions)

### 6. Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Backend API available at: http://localhost:8001
API Documentation: http://localhost:8001/docs

## Quick Start
Run setup.bat (Windows) or bash setup.sh (Mac/Linux) to install everything and launch both servers automatically.

## Frontend Setup
### 1. Install dependencies
cd frontend
npm install

### 2. Configure environment
Create frontend/.env:
VITE_API_URL=http://localhost:8001

### 3. Start frontend
npm run dev

Frontend available at: http://localhost:5173

## Running Both Together
Open two terminals:
Terminal 1 (Backend): cd backend && uvicorn main:app --port 8001 --reload
Terminal 2 (Frontend): cd frontend && npm run dev

## API Endpoints
List all FastAPI routes found in routes.py

## Common Issues & Fixes
- Database connection error: Verify PostgreSQL is running with: pg_isready
- Port in use: Change port with --port flag
- Module not found: Make sure virtual environment is activated
- npm install fails: Try npm install --legacy-peer-deps

## Tech Stack
- Backend: FastAPI, SQLAlchemy 2.0, asyncpg, PostgreSQL
- Frontend: React 18, TypeScript, TailwindCSS (Vite)
- Database: PostgreSQL 14+"""

        logger.info("[%s] Generating README via LLM for project=%r", self.agent_name, project_name)
        result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2000,
        )

        if result["status"] == "success" and result.get("content"):
            content = result["content"].strip()
            # Strip any wrapping markdown fences some LLMs add despite instructions
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            logger.info("[%s] README generated by LLM (%d chars).", self.agent_name, len(content))
            return content.strip()

        logger.warning("[%s] README LLM call failed, using fallback template.", self.agent_name)
        return self._build_readme(project_name)

    def _build_readme(self, project_name: str) -> str:
        """Fallback README template used when the LLM call for README generation fails."""
        slug = project_name.lower().replace(" ", "-").replace("_", "-")
        return f"""\
# {project_name}

## Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+

## Backend Setup
```bash
cd backend
python -m venv venv
# Windows: venv\\Scripts\\activate  |  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create the database:
```bash
psql -U postgres -c 'CREATE DATABASE {slug};'
```

Create `backend/.env`:
```
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/{slug}
```

Start backend:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
API docs: http://localhost:8001/docs

## Frontend Setup
```bash
cd frontend
npm install
```

Create `frontend/.env`:
```
VITE_API_URL=http://localhost:8001
```

Start frontend:
```bash
npm run dev
```
App: http://localhost:5173

## Common Issues
- **DB connection error**: Check PostgreSQL is running and DATABASE_URL is correct
- **Port in use**: Add `--port 8001` to the uvicorn command
- **npm install fails**: Run `npm install --legacy-peer-deps`
"""

    def _clean_content(self, content: str) -> str:
        """Strip markdown fences from generated file content.

        Args:
            content: File content string from the LLM JSON response.

        Returns:
            Clean file content with no surrounding markdown fences.
        """
        content = content.strip()
        for fence in ("```python", "```typescript", "```tsx", "```jsx", "```json", "```yaml", "```"):
            if content.startswith(fence):
                content = content[len(fence):]
                break
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()


def _ext_to_type(ext: str) -> str:
    """Map a file extension to a simple file-type label."""
    mapping = {
        ".py": "python", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript", ".json": "json",
        ".css": "css", ".html": "html", ".md": "markdown",
        ".yml": "yaml", ".yaml": "yaml", ".sh": "bash", ".bat": "batch",
        ".env": "env", ".txt": "text", ".sql": "sql", ".toml": "toml",
    }
    return mapping.get(ext, "text")


# Module-level singleton — import this directly instead of instantiating FullProjectGenerator yourself.
full_project_generator = FullProjectGenerator()
