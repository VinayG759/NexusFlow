"""
NexusFlow RAG Knowledge Base
Contains perfect, tested code examples that are embedded and retrieved
during project generation to improve code quality.
"""

PERFECT_EXAMPLES = [

# ═══ FASTAPI MAIN.PY TEMPLATE ═══
{
    "id": "fastapi_main_template",
    "category": "backend",
    "subcategory": "main",
    "description": "Perfect FastAPI main.py with correct CORS, lifespan, router include",
    "tags": ["fastapi", "cors", "lifespan", "router"],
    "code": """
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="App API", lifespan=lifespan)

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
    return {"message": "API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}
""",
    "file_path": "backend/main.py"
},

# ═══ DATABASE.PY TEMPLATE ═══
{
    "id": "database_template",
    "category": "backend",
    "subcategory": "database",
    "description": "Perfect SQLAlchemy async database.py",
    "tags": ["sqlalchemy", "async", "postgresql", "database"],
    "code": """
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:vinay2004@localhost:5432/myapp"
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
    "file_path": "backend/database.py"
},

# ═══ REQUIREMENTS.TXT TEMPLATE ═══
{
    "id": "requirements_template",
    "category": "backend",
    "subcategory": "requirements",
    "description": "Complete requirements.txt for FastAPI + PostgreSQL app",
    "tags": ["requirements", "fastapi", "sqlalchemy", "asyncpg"],
    "code": """fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
python-dotenv
pydantic
httpx
""",
    "file_path": "backend/requirements.txt",
    "important_note": "NEVER add database, routes, models, schemas to requirements.txt - these are local modules not pip packages"
},

# ═══ REACT INDEX.TSX TEMPLATE ═══
{
    "id": "react_index_template",
    "category": "frontend",
    "subcategory": "entry",
    "description": "Perfect React 18 index.tsx with createRoot",
    "tags": ["react18", "createroot", "typescript"],
    "code": """
import React from 'react';
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
    "file_path": "frontend/src/index.tsx"
},

# ═══ INDEX.CSS TEMPLATE ═══
{
    "id": "index_css_template",
    "category": "frontend",
    "subcategory": "styles",
    "description": "Base CSS reset that must always be generated",
    "tags": ["css", "reset", "styles"],
    "code": """
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
  background-color: #ffffff;
  color: #1a1a1a;
}

a {
  text-decoration: none;
  color: inherit;
}

button {
  cursor: pointer;
  border: none;
  outline: none;
}

input, textarea, select {
  outline: none;
  font-family: inherit;
}
""",
    "file_path": "frontend/src/index.css"
},

# ═══ TSCONFIG.JSON TEMPLATE ═══
{
    "id": "tsconfig_template",
    "category": "frontend",
    "subcategory": "config",
    "description": "Perfect tsconfig.json for React TypeScript app",
    "tags": ["typescript", "tsconfig", "react"],
    "code": """{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}""",
    "file_path": "frontend/tsconfig.json"
},

# ═══ PACKAGE.JSON TEMPLATE (VITE) ═══
{
    "id": "package_json_template",
    "category": "frontend",
    "subcategory": "config",
    "description": "Vite-based package.json for React TypeScript app — NEVER use react-scripts",
    "tags": ["package.json", "npm", "react", "vite", "dependencies"],
    "code": """{
  "name": "app-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "axios": "^1.7.7"
  },
  "devDependencies": {
    "vite": "^5.4.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.6.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@types/node": "^22.0.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }
}""",
    "file_path": "frontend/package.json"
},

# ═══ PUBLIC INDEX.HTML TEMPLATE ═══
{
    "id": "index_html_template",
    "category": "frontend",
    "subcategory": "public",
    "description": "Required public/index.html for React app",
    "tags": ["html", "public", "react"],
    "code": """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>""",
    "file_path": "frontend/public/index.html"
},

# ═══ BACKEND .ENV.EXAMPLE TEMPLATE ═══
{
    "id": "backend_env_template",
    "category": "backend",
    "subcategory": "config",
    "description": "Backend .env.example with all required variables",
    "tags": ["env", "environment", "config"],
    "code": """DATABASE_URL=postgresql+asyncpg://postgres:vinay2004@localhost:5432/myapp
""",
    "file_path": "backend/.env.example"
},

# ═══ FRONTEND .ENV TEMPLATE ═══
{
    "id": "frontend_env_template",
    "category": "frontend",
    "subcategory": "config",
    "description": "Frontend .env with API URL",
    "tags": ["env", "environment", "api"],
    "code": """REACT_APP_API_URL=http://localhost:8000
""",
    "file_path": "frontend/.env"
},

# ═══ DECLARATIONS.D.TS TEMPLATE ═══
{
    "id": "declarations_template",
    "category": "frontend",
    "subcategory": "types",
    "description": "TypeScript declarations for CSS and asset imports",
    "tags": ["typescript", "declarations", "css"],
    "code": """declare module '*.css';
declare module '*.svg';
declare module '*.png';
declare module '*.jpg';
declare module '*.jpeg';
declare module '*.gif';
""",
    "file_path": "frontend/src/declarations.d.ts"
},

# ═══ SQLALCHEMY MODEL TEMPLATE ═══
{
    "id": "sqlalchemy_model_template",
    "category": "backend",
    "subcategory": "models",
    "description": "Perfect SQLAlchemy 2.0 model with Mapped columns",
    "tags": ["sqlalchemy", "model", "mapped", "orm"],
    "code": """
from datetime import datetime
from sqlalchemy import String, Text, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

class Todo(Base):
    __tablename__ = "todos"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
""",
    "file_path": "backend/models.py"
},

# ═══ FASTAPI ROUTES TEMPLATE ═══
{
    "id": "fastapi_routes_template",
    "category": "backend",
    "subcategory": "routes",
    "description": "Perfect FastAPI routes with async SQLAlchemy",
    "tags": ["fastapi", "routes", "sqlalchemy", "crud"],
    "code": """
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import Todo
from schemas import TodoCreate, TodoResponse

router = APIRouter()

@router.get("/todos", response_model=list[TodoResponse])
async def get_todos(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Todo))
    return result.scalars().all()

@router.post("/todos", response_model=TodoResponse)
async def create_todo(todo: TodoCreate, db: AsyncSession = Depends(get_db)):
    db_todo = Todo(**todo.model_dump())
    db.add(db_todo)
    await db.commit()
    await db.refresh(db_todo)
    return db_todo

@router.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Todo).where(Todo.id == todo_id))
    todo = result.scalar_one_or_none()
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    await db.delete(todo)
    await db.commit()
    return {"message": "Deleted successfully"}
""",
    "file_path": "backend/routes.py"
},

# ═══ PYDANTIC SCHEMAS TEMPLATE ═══
{
    "id": "pydantic_schemas_template",
    "category": "backend",
    "subcategory": "schemas",
    "description": "Perfect Pydantic v2 schemas",
    "tags": ["pydantic", "schemas", "validation"],
    "code": """
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class TodoCreate(BaseModel):
    title: str
    description: str = ""

class TodoResponse(BaseModel):
    id: int
    title: str
    description: str
    completed: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
""",
    "file_path": "backend/schemas.py"
},

# ═══ REACT APP.TSX TEMPLATE ═══
{
    "id": "react_app_template",
    "category": "frontend",
    "subcategory": "app",
    "description": "Perfect React App.tsx with API calls",
    "tags": ["react", "typescript", "axios", "api"],
    "code": """
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface Todo {
  id: number;
  title: string;
  completed: boolean;
}

function App() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [newTodo, setNewTodo] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTodos();
  }, []);

  const fetchTodos = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/todos`);
      setTodos(response.data);
    } catch (err: unknown) {
      const e = err as any;
      setError(e?.response?.data?.detail || 'Failed to fetch todos');
    } finally {
      setLoading(false);
    }
  };

  const addTodo = async () => {
    if (!newTodo.trim()) return;
    try {
      const response = await axios.post(`${API_URL}/api/todos`, { title: newTodo });
      setTodos([...todos, response.data]);
      setNewTodo('');
    } catch (err: unknown) {
      const e = err as any;
      setError(e?.response?.data?.detail || 'Failed to add todo');
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: '0 20px' }}>
      <h1>Todo App</h1>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <input
          value={newTodo}
          onChange={(e) => setNewTodo(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addTodo()}
          placeholder="Add a todo..."
          style={{ flex: 1, padding: '8px 12px', borderRadius: 4, border: '1px solid #ddd' }}
        />
        <button onClick={addTodo} style={{ padding: '8px 16px', background: '#007bff', color: '#fff', borderRadius: 4 }}>
          Add
        </button>
      </div>
      {loading ? <p>Loading...</p> : (
        <ul style={{ listStyle: 'none', padding: 0 }}>
          {todos.map(todo => (
            <li key={todo.id} style={{ padding: '12px', borderBottom: '1px solid #eee' }}>
              {todo.title}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;
""",
    "file_path": "frontend/src/App.tsx"
},

# ═══ JWT AUTHENTICATION BACKEND ═══
{
    "id": "jwt_auth_backend",
    "category": "backend",
    "subcategory": "auth",
    "description": "Complete JWT authentication system with login, register, token refresh",
    "tags": ["jwt", "auth", "login", "register", "security", "fastapi"],
    "code": """
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext
from database import get_db
from models import User
from schemas import UserCreate, UserResponse, Token

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router = APIRouter()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user

@router.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user = User(email=user.email, hashed_password=hash_password(user.password), name=user.name)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@router.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
""",
    "file_path": "backend/auth.py"
},

# ═══ USER MODEL FOR AUTH ═══
{
    "id": "user_model_auth",
    "category": "backend",
    "subcategory": "models",
    "description": "SQLAlchemy User model for authentication",
    "tags": ["user", "model", "auth", "sqlalchemy"],
    "code": """
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
""",
    "file_path": "backend/models.py"
},

# ═══ AUTH SCHEMAS ═══
{
    "id": "auth_schemas",
    "category": "backend",
    "subcategory": "schemas",
    "description": "Pydantic schemas for authentication",
    "tags": ["auth", "schemas", "pydantic", "jwt"],
    "code": """
from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    email: str
    name: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    is_active: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
""",
    "file_path": "backend/schemas.py"
},

# ═══ REACT AUTH CONTEXT ═══
{
    "id": "react_auth_context",
    "category": "frontend",
    "subcategory": "auth",
    "description": "React authentication context with login, logout, token management",
    "tags": ["react", "auth", "context", "jwt", "typescript"],
    "code": """
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface User {
  id: number;
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUser();
    } else {
      setIsLoading(false);
    }
  }, [token]);

  const fetchUser = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/auth/me`);
      setUser(response.data);
    } catch (err: unknown) {
      logout();
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    const response = await axios.post(`${API_URL}/api/auth/login`, formData);
    const { access_token } = response.data;
    localStorage.setItem('token', access_token);
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
    setToken(access_token);
  };

  const logout = () => {
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}
""",
    "file_path": "frontend/src/contexts/AuthContext.tsx"
},

# ═══ LOGIN PAGE ═══
{
    "id": "login_page",
    "category": "frontend",
    "subcategory": "auth",
    "description": "Beautiful login page with TailwindCSS",
    "tags": ["login", "auth", "react", "tailwind", "form"],
    "code": """
import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err: unknown) {
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome back</h1>
        <p className="text-gray-500 mb-8">Sign in to your account</p>
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
              placeholder="you@example.com" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
              className="w-full border-2 border-gray-200 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
              placeholder="••••••••" />
          </div>
          <button type="submit" disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200 shadow-md">
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
        <p className="mt-6 text-center text-gray-500">
          Don't have an account? <Link to="/register" className="text-blue-600 font-semibold hover:underline">Sign up</Link>
        </p>
      </div>
    </div>
  );
}
""",
    "file_path": "frontend/src/pages/Login.tsx"
},

# ═══ DASHBOARD LAYOUT ═══
{
    "id": "dashboard_layout",
    "category": "frontend",
    "subcategory": "layout",
    "description": "Professional SaaS dashboard layout with sidebar",
    "tags": ["dashboard", "layout", "sidebar", "saas", "tailwind"],
    "code": """
import React, { ReactNode, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

interface NavItem {
  label: string;
  path: string;
  icon: string;
}

const navItems: NavItem[] = [
  { label: 'Dashboard', path: '/dashboard', icon: '📊' },
  { label: 'Projects', path: '/projects', icon: '📁' },
  { label: 'Settings', path: '/settings', icon: '⚙️' },
];

export default function DashboardLayout({ children }: { children: ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleLogout = () => { logout(); navigate('/login'); };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-white border-r border-gray-200 flex flex-col transition-all duration-300`}>
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          {sidebarOpen && <span className="font-bold text-xl text-blue-600">AppName</span>}
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 rounded-lg hover:bg-gray-100">☰</button>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map(item => (
            <Link key={item.path} to={item.path}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${location.pathname === item.path ? 'bg-blue-50 text-blue-600 font-semibold' : 'text-gray-600 hover:bg-gray-100'}`}>
              <span>{item.icon}</span>
              {sidebarOpen && <span>{item.label}</span>}
            </Link>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-200">
          {sidebarOpen && <p className="text-sm text-gray-500 mb-2 truncate">{user?.email}</p>}
          <button onClick={handleLogout} className="flex items-center gap-2 text-red-500 hover:text-red-700 text-sm font-medium">
            <span>🚪</span>{sidebarOpen && 'Logout'}
          </button>
        </div>
      </aside>
      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-800">{navItems.find(i => i.path === location.pathname)?.label || 'Dashboard'}</h1>
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-500">Welcome, {user?.name}</span>
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
              {user?.name?.[0]?.toUpperCase()}
            </div>
          </div>
        </header>
        <div className="p-6">{children}</div>
      </main>
    </div>
  );
}
""",
    "file_path": "frontend/src/layouts/DashboardLayout.tsx"
},

# ═══ STRIPE PAYMENT BACKEND ═══
{
    "id": "stripe_payment_backend",
    "category": "backend",
    "subcategory": "payments",
    "description": "Stripe payment integration with checkout session and webhook",
    "tags": ["stripe", "payments", "subscription", "webhook", "saas"],
    "code": """
import stripe
import os
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
router = APIRouter()

@router.post("/payments/create-checkout")
async def create_checkout_session(price_id: str, user_id: int, db: AsyncSession = Depends(get_db)):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url="http://localhost:5173/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:5173/pricing",
            metadata={"user_id": str(user_id)},
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/payments/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"]["user_id"]
        # Update user subscription status in DB here

    return {"status": "ok"}
""",
    "file_path": "backend/payments.py"
},

# ═══ PAGINATION CRUD ═══
{
    "id": "pagination_crud",
    "category": "backend",
    "subcategory": "routes",
    "description": "FastAPI CRUD with pagination, search and filtering",
    "tags": ["crud", "pagination", "search", "filter", "fastapi"],
    "code": """
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from database import get_db
from models import Item
from schemas import ItemCreate, ItemResponse, PaginatedResponse

router = APIRouter()

@router.get("/items", response_model=PaginatedResponse)
async def get_items(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: str = Query(""),
    db: AsyncSession = Depends(get_db)
):
    offset = (page - 1) * limit
    query = select(Item)
    if search:
        query = query.where(or_(Item.title.ilike(f"%{search}%"), Item.description.ilike(f"%{search}%")))

    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar()

    result = await db.execute(query.offset(offset).limit(limit))
    items = result.scalars().all()

    return {"items": items, "total": total, "page": page, "pages": (total + limit - 1) // limit}

@router.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    db_item = Item(**item.model_dump())
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@router.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: ItemCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Item).where(Item.id == item_id))
    db_item = result.scalar_one_or_none()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    for key, value in item.model_dump().items():
        setattr(db_item, key, value)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@router.delete("/items/{item_id}")
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Item).where(Item.id == item_id))
    db_item = result.scalar_one_or_none()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    await db.delete(db_item)
    await db.commit()
    return {"message": "Deleted successfully"}
""",
    "file_path": "backend/routes.py"
},

# ═══ DATA TABLE COMPONENT ═══
{
    "id": "data_table_component",
    "category": "frontend",
    "subcategory": "components",
    "description": "Reusable data table with pagination, search and sorting",
    "tags": ["table", "pagination", "search", "react", "tailwind", "saas"],
    "code": """
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface Column<T> {
  key: keyof T;
  label: string;
  render?: (value: T[keyof T], row: T) => React.ReactNode;
}

interface DataTableProps<T> {
  endpoint: string;
  columns: Column<T>[];
  title: string;
}

export default function DataTable<T extends { id: number }>({ endpoint, columns, title }: DataTableProps<T>) {
  const [data, setData] = useState<T[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const limit = 10;

  useEffect(() => { fetchData(); }, [page, search]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/api/${endpoint}`, { params: { page, limit, search } });
      setData(res.data.items);
      setTotal(res.data.total);
    } catch (err: unknown) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const pages = Math.ceil(total / limit);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        <input value={search} onChange={e => { setSearch(e.target.value); setPage(1); }}
          placeholder="Search..." className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-64" />
      </div>
      {loading ? (
        <div className="flex justify-center py-12"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" /></div>
      ) : (
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>{columns.map(col => <th key={String(col.key)} className="text-left px-6 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">{col.label}</th>)}</tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {data.map(row => (
              <tr key={row.id} className="hover:bg-gray-50 transition-colors">
                {columns.map(col => <td key={String(col.key)} className="px-6 py-4 text-sm text-gray-700">{col.render ? col.render(row[col.key], row) : String(row[col.key])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
        <span className="text-sm text-gray-500">{total} total results</span>
        <div className="flex gap-2">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
            className="px-3 py-1 rounded border border-gray-300 text-sm disabled:opacity-50 hover:bg-gray-50">Previous</button>
          <span className="px-3 py-1 text-sm text-gray-700">Page {page} of {pages}</span>
          <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages}
            className="px-3 py-1 rounded border border-gray-300 text-sm disabled:opacity-50 hover:bg-gray-50">Next</button>
        </div>
      </div>
    </div>
  );
}
""",
    "file_path": "frontend/src/components/DataTable.tsx"
},

# ═══ STATS CARDS COMPONENT ═══
{
    "id": "stats_cards_component",
    "category": "frontend",
    "subcategory": "components",
    "description": "SaaS dashboard stats cards with icons and trends",
    "tags": ["stats", "dashboard", "cards", "saas", "tailwind", "react"],
    "code": """
import React from 'react';

interface StatCard {
  label: string;
  value: string | number;
  icon: string;
  trend?: string;
  trendUp?: boolean;
  color: string;
}

interface StatsCardsProps {
  stats: StatCard[];
}

export default function StatsCards({ stats }: StatsCardsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {stats.map((stat, i) => (
        <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <span className={`text-2xl p-2 rounded-lg ${stat.color}`}>{stat.icon}</span>
            {stat.trend && (
              <span className={`text-sm font-medium ${stat.trendUp ? 'text-green-600' : 'text-red-600'}`}>
                {stat.trendUp ? '↑' : '↓'} {stat.trend}
              </span>
            )}
          </div>
          <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
          <p className="text-sm text-gray-500 mt-1">{stat.label}</p>
        </div>
      ))}
    </div>
  );
}
""",
    "file_path": "frontend/src/components/StatsCards.tsx"
},

# ═══ WEBSOCKET REAL-TIME ═══
{
    "id": "websocket_realtime",
    "category": "backend",
    "subcategory": "realtime",
    "description": "FastAPI WebSocket for real-time features",
    "tags": ["websocket", "realtime", "fastapi", "chat"],
    "code": """
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room: str):
        await websocket.accept()
        if room not in self.active_connections:
            self.active_connections[room] = []
        self.active_connections[room].append(websocket)

    def disconnect(self, websocket: WebSocket, room: str):
        if room in self.active_connections:
            self.active_connections[room].remove(websocket)

    async def broadcast(self, message: dict, room: str):
        if room in self.active_connections:
            for connection in self.active_connections[room]:
                await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@router.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    await manager.connect(websocket, room)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await manager.broadcast({"message": message, "room": room}, room)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
        await manager.broadcast({"message": "A user left", "room": room}, room)
""",
    "file_path": "backend/websocket.py"
},

# ═══ FILE UPLOAD ═══
{
    "id": "file_upload_backend",
    "category": "backend",
    "subcategory": "files",
    "description": "FastAPI file upload with validation and storage",
    "tags": ["upload", "files", "fastapi", "storage"],
    "code": """
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "application/pdf"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {file.content_type}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(contents)
    return {"filename": filename, "url": f"/uploads/{filename}", "size": len(contents)}
""",
    "file_path": "backend/uploads.py"
},

# ═══ VITE CONFIG TEMPLATE ═══
{
    "id": "vite_config_template",
    "category": "frontend",
    "subcategory": "config",
    "description": "vite.config.ts for React TypeScript project with API proxy",
    "tags": ["vite", "config", "typescript", "react", "proxy"],
    "code": """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
""",
    "file_path": "frontend/vite.config.ts"
},

# ═══ REACT ROUTER V6 APP WITH PROTECTED ROUTES ═══
{
    "id": "react_router_protected_routes",
    "category": "frontend",
    "subcategory": "routing",
    "description": "React Router v6 App.tsx with public and protected routes, AuthProvider wrapping",
    "tags": ["react-router", "routing", "protected", "auth", "typescript"],
    "code": """
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import DashboardLayout from './layouts/DashboardLayout';

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return <div className="flex items-center justify-center h-screen"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" /></div>;
  return user ? <>{children}</> : <Navigate to="/login" replace />;
}

function PublicRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  return !user ? <>{children}</> : <Navigate to="/dashboard" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
          <Route path="/register" element={<PublicRoute><Register /></PublicRoute>} />
          <Route path="/dashboard" element={
            <PrivateRoute>
              <DashboardLayout><Dashboard /></DashboardLayout>
            </PrivateRoute>
          } />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
""",
    "file_path": "frontend/src/App.tsx"
},

# ═══ AXIOS API SERVICE WITH INTERCEPTORS ═══
{
    "id": "axios_api_service",
    "category": "frontend",
    "subcategory": "api",
    "description": "Axios instance with auth token interceptor and 401 redirect",
    "tags": ["axios", "api", "interceptor", "auth", "typescript"],
    "code": """
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
""",
    "file_path": "frontend/src/services/api.ts"
},

# ═══ REACT ERROR BOUNDARY ═══
{
    "id": "react_error_boundary",
    "category": "frontend",
    "subcategory": "components",
    "description": "React class-based error boundary with fallback UI",
    "tags": ["error-boundary", "react", "typescript", "error-handling"],
    "code": """
import React, { Component, ReactNode } from 'react';

interface Props { children: ReactNode; fallback?: ReactNode; }
interface State { hasError: boolean; error: Error | null; }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-8 text-center">
          <div className="text-4xl mb-4">&#9888;</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Something went wrong</h2>
          <p className="text-gray-500 text-sm mb-4">{this.state.error?.message}</p>
          <button onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
""",
    "file_path": "frontend/src/components/ErrorBoundary.tsx"
},

# ═══ LOADING SKELETON COMPONENT ═══
{
    "id": "loading_skeleton",
    "category": "frontend",
    "subcategory": "components",
    "description": "Skeleton loading placeholder with pulse animation using TailwindCSS",
    "tags": ["skeleton", "loading", "react", "tailwind", "ux"],
    "code": """
import React from 'react';

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse bg-gray-200 rounded ${className}`} />;
}

export function CardSkeleton() {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-8 w-1/2" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-2/3" />
    </div>
  );
}

export function TableRowSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i} className="border-b border-gray-100">
          {Array.from({ length: 4 }).map((_, j) => (
            <td key={j} className="px-6 py-4"><Skeleton className="h-4 w-full" /></td>
          ))}
        </tr>
      ))}
    </>
  );
}

export default Skeleton;
""",
    "file_path": "frontend/src/components/Skeleton.tsx"
},

# ═══ MODAL COMPONENT ═══
{
    "id": "modal_component",
    "category": "frontend",
    "subcategory": "components",
    "description": "Accessible modal dialog with backdrop click to close and escape key support",
    "tags": ["modal", "dialog", "react", "tailwind", "accessibility"],
    "code": """
import React, { useEffect, ReactNode } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  size?: 'sm' | 'md' | 'lg';
}

const sizeClasses = { sm: 'max-w-sm', md: 'max-w-md', lg: 'max-w-2xl' };

export default function Modal({ isOpen, onClose, title, children, size = 'md' }: ModalProps) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    if (isOpen) document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      <div className={`relative bg-white rounded-2xl shadow-2xl w-full ${sizeClasses[size]} max-h-[90vh] overflow-y-auto`}>
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
        </div>
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
}
""",
    "file_path": "frontend/src/components/Modal.tsx"
},

# ═══ TOAST NOTIFICATION HOOK ═══
{
    "id": "toast_hook",
    "category": "frontend",
    "subcategory": "hooks",
    "description": "Lightweight toast notification system using React state — no external library needed",
    "tags": ["toast", "notification", "react", "hook", "tailwind"],
    "code": """
import React, { useState, useCallback, createContext, useContext, ReactNode } from 'react';

type ToastType = 'success' | 'error' | 'info';
interface Toast { id: number; message: string; type: ToastType; }
interface ToastContextType { showToast: (message: string, type?: ToastType) => void; }

const ToastContext = createContext<ToastContextType | null>(null);
let _counter = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((message: string, type: ToastType = 'info') => {
    const id = ++_counter;
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500);
  }, []);

  const colors = { success: 'bg-green-500', error: 'bg-red-500', info: 'bg-blue-500' };
  const icons = { success: '&#10003;', error: '&#10005;', info: 'i' };

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        {toasts.map(t => (
          <div key={t.id} className={`${colors[t.type]} text-white px-4 py-3 rounded-lg shadow-lg text-sm flex items-center gap-2`}>
            <span dangerouslySetInnerHTML={{ __html: icons[t.type] }} />
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}
""",
    "file_path": "frontend/src/hooks/useToast.tsx"
},

# ═══ DOCKER-COMPOSE FOR GENERATED PROJECTS ═══
{
    "id": "docker_compose_generated",
    "category": "devops",
    "subcategory": "docker",
    "description": "docker-compose.yml for a generated full-stack app (backend + frontend + PostgreSQL)",
    "tags": ["docker", "docker-compose", "postgresql", "devops", "deployment"],
    "code": """version: '3.8'

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-password}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:${POSTGRES_PASSWORD:-password}@db:5432/appdb
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
""",
    "file_path": "docker-compose.yml"
},

# ═══ DOCKERFILE FOR GENERATED BACKEND ═══
{
    "id": "dockerfile_backend_generated",
    "category": "devops",
    "subcategory": "docker",
    "description": "Dockerfile for a FastAPI backend with asyncpg",
    "tags": ["docker", "dockerfile", "fastapi", "python", "backend"],
    "code": """FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    "file_path": "backend/Dockerfile"
},

# ═══ DOCKERFILE FOR GENERATED FRONTEND (VITE) ═══
{
    "id": "dockerfile_frontend_generated",
    "category": "devops",
    "subcategory": "docker",
    "description": "Multi-stage Dockerfile for a Vite React app — builds with Node, serves with Nginx",
    "tags": ["docker", "dockerfile", "vite", "react", "nginx", "frontend"],
    "code": """FROM node:18-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
""",
    "file_path": "frontend/Dockerfile"
},

# ═══ NGINX CONFIG FOR GENERATED FRONTEND ═══
{
    "id": "nginx_conf_generated",
    "category": "devops",
    "subcategory": "nginx",
    "description": "nginx.conf for Vite React SPA — handles React Router client-side routing",
    "tags": ["nginx", "spa", "react", "routing", "frontend"],
    "code": """server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
""",
    "file_path": "frontend/nginx.conf"
},

# ═══ ZUSTAND GLOBAL STATE STORE ═══
{
    "id": "zustand_store",
    "category": "frontend",
    "subcategory": "state",
    "description": "Zustand store for global client state — lighter than Redux, no boilerplate",
    "tags": ["zustand", "state", "react", "typescript", "store"],
    "code": """
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User { id: number; email: string; name: string; }

interface AppStore {
  user: User | null;
  token: string | null;
  sidebarOpen: boolean;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  toggleSidebar: () => void;
  logout: () => void;
}

export const useStore = create<AppStore>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      sidebarOpen: true,
      setUser: (user) => set({ user }),
      setToken: (token) => set({ token }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      logout: () => set({ user: null, token: null }),
    }),
    { name: 'app-store', partialize: (s) => ({ token: s.token }) }
  )
);
""",
    "file_path": "frontend/src/store/useStore.ts"
},

# ═══ FASTAPI BACKGROUND TASKS PATTERN ═══
{
    "id": "fastapi_background_tasks",
    "category": "backend",
    "subcategory": "tasks",
    "description": "FastAPI async background job with status polling — avoids request timeouts for long operations",
    "tags": ["fastapi", "background-tasks", "async", "job", "polling"],
    "code": """
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any

router = APIRouter()
_jobs: Dict[str, Dict[str, Any]] = {}

async def _process(job_id: str, payload: dict) -> None:
    try:
        _jobs[job_id]["status"] = "running"
        result = {"processed": True, "data": payload}
        _jobs[job_id] = {"status": "complete", "result": result, "error": None}
    except Exception as e:
        _jobs[job_id] = {"status": "failed", "result": None, "error": str(e)}

@router.post("/jobs")
async def create_job(payload: dict, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}
    background_tasks.add_task(_process, job_id, payload)
    return {"job_id": job_id, "status": "queued"}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_jobs[job_id]}
""",
    "file_path": "backend/jobs.py"
},

]
