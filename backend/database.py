import os
import psycopg2
from contextlib import contextmanager
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_SERVICE_URI"))
        yield conn
    except psycopg2.Error as e:
        raise RuntimeError(f"Database connection error: {e}")
    finally:
        if conn is not None:
            conn.close()

@contextmanager
def get_db_cursor(conn: psycopg2.extensions.connection) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager for database cursors"""
    cur = None
    try:
        cur = conn.cursor()
        yield cur
    except psycopg2.Error as e:
        conn.rollback()
        raise RuntimeError(f"Database cursor error: {e}")
    finally:
        if cur is not None:
            cur.close()
