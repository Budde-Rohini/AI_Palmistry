"""
Database Operations Module
SQLite database setup and query functions for storing and retrieving palm analysis readings.
"""
import sqlite3
import json
import os
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Readings Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            image_path TEXT NOT NULL,
            processed_image_path TEXT,
            overlay_image_path TEXT,
            hand_type TEXT,
            palm_features TEXT,
            line_features TEXT,
            interpretation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Ensure user_id column exists if table was created earlier
    cursor.execute("PRAGMA table_info(readings)")
    columns = [col['name'] for col in cursor.fetchall()]
    if 'user_id' not in columns:
        cursor.execute("ALTER TABLE readings ADD COLUMN user_id INTEGER")

    conn.commit()
    conn.close()

def create_user(username, email, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username.strip().lower(), email.strip().lower(), password_hash))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError as e:
        conn.close()
        err_msg = str(e).lower()
        if 'username' in err_msg:
            return False, "Username is already taken."
        elif 'email' in err_msg:
            return False, "Email address is already registered."
        return False, "User account creation failed."

def get_user_by_username_or_email(identifier):
    conn = get_db_connection()
    cursor = conn.cursor()
    clean_id = identifier.strip().lower()
    cursor.execute('''
        SELECT * FROM users WHERE username = ? OR email = ?
    ''', (clean_id, clean_id))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, created_at FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def create_reading(reading_id, image_path, processed_image_path=None, user_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO readings (id, image_path, processed_image_path, user_id)
        VALUES (?, ?, ?, ?)
    ''', (reading_id, image_path, processed_image_path, user_id))
    conn.commit()
    conn.close()

def get_reading(reading_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM readings WHERE id = ?', (reading_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_parsed_reading(reading_id):
    """Fetches reading and parses JSON fields into python dictionaries."""
    reading = get_reading(reading_id)
    if not reading:
        return None

    # Parse JSON fields
    for field in ['palm_features', 'line_features', 'interpretation']:
        if reading.get(field):
            try:
                reading[field] = json.loads(reading[field])
            except Exception:
                pass
        else:
            reading[field] = {}

    return reading

def update_reading_analysis(reading_id, overlay_image_path, hand_type, palm_features, line_features, interpretation):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE readings
        SET overlay_image_path = ?,
            hand_type = ?,
            palm_features = ?,
            line_features = ?,
            interpretation = ?
        WHERE id = ?
    ''', (
        overlay_image_path,
        hand_type,
        json.dumps(palm_features) if isinstance(palm_features, (dict, list)) else palm_features,
        json.dumps(line_features) if isinstance(line_features, (dict, list)) else line_features,
        json.dumps(interpretation) if isinstance(interpretation, (dict, list)) else interpretation,
        reading_id
    ))
    conn.commit()
    conn.close()

def list_user_readings(user_id, limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, image_path, overlay_image_path, hand_type, created_at
        FROM readings
        WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?
    ''', (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def list_readings(limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, image_path, hand_type, created_at FROM readings ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]



