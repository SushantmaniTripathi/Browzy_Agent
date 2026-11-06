"""
Browzy Database - SQLite persistence for chats, tasks, vault, etc.
"""
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


DB_PATH = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Browzy" / "browzy.db"
    if os.name == "nt"
    else Path.home() / ".browzy" / "browzy.db"
)


def get_db_path() -> Path:
    """Get database path, ensuring directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize database with all tables."""
    with get_db() as conn:
        # Chats table - stores all chat messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
                content TEXT NOT NULL,
                model TEXT,
                type TEXT DEFAULT 'normal',  -- 'normal', 'error', 'log'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for session queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_session_id
            ON chats(session_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_timestamp
            ON chats(timestamp)
        """)

        # Sessions table - stores session metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT,
                message_count INTEGER DEFAULT 0
            )
        """)

        # Tasks table - stores automation task state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                instruction TEXT NOT NULL,
                state TEXT NOT NULL,  -- 'idle', 'planning', 'executing', 'paused', 'done', 'error', 'stopped'
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                result_data TEXT  -- JSON
            )
        """)

        # Vault profiles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault_profiles (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Vault fields table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                encrypted_value TEXT NOT NULL,
                field_type TEXT,
                FOREIGN KEY (profile_id) REFERENCES vault_profiles(id) ON DELETE CASCADE
            )
        """)

        # Vault documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault_documents (
                id TEXT PRIMARY KEY,
                profile_id TEXT,
                filename TEXT NOT NULL,
                mime_type TEXT,
                encrypted_content TEXT NOT NULL,
                extracted_fields TEXT,  -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (profile_id) REFERENCES vault_profiles(id) ON DELETE SET NULL
            )
        """)

        conn.commit()


# ============================================================================
# CHAT FUNCTIONS
# ============================================================================

def save_message(session_id: str, role: str, content: str, model: str = None, msg_type: str = 'normal') -> int:
    """Save a chat message to the database."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO chats (session_id, role, content, model, type) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, model, msg_type)
        )
        # Update session message count and timestamp
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, title, model) VALUES (?, ?, ?)",
            (session_id, content[:50] if role == 'user' else 'New Chat', model or '')
        )
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )
        conn.commit()
        return cursor.lastrowid


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a session."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_recent_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent sessions with their latest message."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.*,
                   strftime('%Y-%m-%dT%H:%M:%SZ', s.updated_at) as updated_at,
                   (SELECT content FROM chats WHERE session_id = s.id ORDER BY timestamp DESC LIMIT 1) as last_message,
                   (SELECT timestamp FROM chats WHERE session_id = s.id ORDER BY timestamp DESC LIMIT 1) as last_timestamp
            FROM sessions s
            WHERE s.message_count > 0
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def delete_session(session_id: str) -> bool:
    """Delete one saved conversation (messages + session row)."""
    with get_db() as conn:
        conn.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0


def update_session_title(session_id: str, title: str):
    """Update session title."""
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id)
        )
        conn.commit()


# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def save_task(task_id: str, instruction: str, state: str = 'idle',
              current_step: int = 0, total_steps: int = 0, result_data: dict = None):
    """Save or update a task."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO tasks (id, instruction, state, current_step, total_steps, result_data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                instruction = excluded.instruction,
                state = excluded.state,
                current_step = excluded.current_step,
                total_steps = excluded.total_steps,
                result_data = excluded.result_data,
                updated_at = CURRENT_TIMESTAMP
        """, (task_id, instruction, state, current_step, total_steps, json.dumps(result_data) if result_data else None))
        conn.commit()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get a task by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row:
            data = dict(row)
            if data.get('result_data'):
                data['result_data'] = json.loads(data['result_data'])
            return data
        return None


def get_recent_tasks(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent tasks."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM tasks
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


# ============================================================================
# VAULT FUNCTIONS
# ============================================================================

def save_vault_profile(profile_id: str, label: str):
    """Save a vault profile."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO vault_profiles (id, label) VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET label = excluded.label, updated_at = CURRENT_TIMESTAMP
        """, (profile_id, label))
        conn.commit()


def get_vault_profiles() -> List[Dict[str, Any]]:
    """Get all vault profiles."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM vault_profiles ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def save_vault_field(profile_id: str, field_name: str, encrypted_value: str, field_type: str = None):
    """Save a vault field."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO vault_fields (profile_id, field_name, encrypted_value, field_type)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_id, field_name) DO UPDATE SET
                encrypted_value = excluded.encrypted_value,
                field_type = excluded.field_type
        """, (profile_id, field_name, encrypted_value, field_type))
        conn.commit()


def get_vault_fields(profile_id: str) -> List[Dict[str, Any]]:
    """Get all fields for a profile."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM vault_fields WHERE profile_id = ?",
            (profile_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def delete_vault_profile(profile_id: str):
    """Delete a vault profile and all its fields."""
    with get_db() as conn:
        conn.execute("DELETE FROM vault_profiles WHERE id = ?", (profile_id,))
        conn.commit()


# Initialize database on import
init_db()