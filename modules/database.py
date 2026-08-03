"""
database.py

Handles all SQLite database operations for the
End-to-End Information Retrieval System.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict

from config import DATABASE_PATH


class DatabaseManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self._initialize_database()

    def get_connection(self):
        """Return a SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        """Create required tables if they do not exist."""

        conn = self.get_connection()
        cursor = conn.cursor()

        # -----------------------------
        # Metadata Table
        # -----------------------------
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata(

            document_id INTEGER PRIMARY KEY AUTOINCREMENT,

            url TEXT UNIQUE,

            title TEXT,

            author TEXT,

            publish_date TEXT,

            language TEXT,

            crawl_date TEXT,

            content_hash TEXT,

            content_length INTEGER
        )
        """)

        # -----------------------------
        # Documents Table
        # -----------------------------
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents(

            document_id INTEGER PRIMARY KEY,

            content TEXT,

            FOREIGN KEY(document_id)
            REFERENCES metadata(document_id)
        )
        """)

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # INSERT
    # --------------------------------------------------

    def insert_document(
            self,
            url: str,
            title: str,
            author: str,
            publish_date: str,
            language: str,
            crawl_date: str,
            content_hash: str,
            content: str):

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""

        INSERT OR IGNORE INTO metadata(

            url,
            title,
            author,
            publish_date,
            language,
            crawl_date,
            content_hash,
            content_length

        )

        VALUES(?,?,?,?,?,?,?,?)

        """, (

            url,
            title,
            author,
            publish_date,
            language,
            crawl_date,
            content_hash,
            len(content)

        ))

        conn.commit()

        cursor.execute(
            "SELECT document_id FROM metadata WHERE url=?",
            (url,)
        )

        row = cursor.fetchone()

        if row:

            document_id = row[0]

            cursor.execute("""

            INSERT OR REPLACE INTO documents(

                document_id,
                content

            )

            VALUES(?,?)

            """, (

                document_id,
                content

            ))

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # GET ALL DOCUMENTS
    # --------------------------------------------------

    def get_all_documents(self) -> List[Dict]:

        conn = self.get_connection()

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("""

        SELECT

            m.document_id,
            m.url,
            m.title,
            m.author,
            m.publish_date,
            m.language,
            m.crawl_date,
            m.content_hash,
            d.content

        FROM metadata m

        JOIN documents d

        ON m.document_id=d.document_id

        """)

        rows = cursor.fetchall()

        conn.close()

        return [dict(row) for row in rows]

    # --------------------------------------------------
    # GET SINGLE DOCUMENT
    # --------------------------------------------------

    def get_document(self, document_id: int):

        conn = self.get_connection()

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("""

        SELECT

            *

        FROM metadata

        JOIN documents

        ON metadata.document_id=documents.document_id

        WHERE metadata.document_id=?

        """, (document_id,))

        row = cursor.fetchone()

        conn.close()

        if row:
            return dict(row)

        return None

    # --------------------------------------------------
    # DELETE DOCUMENT
    # --------------------------------------------------

    def delete_document(self, document_id: int):

        conn = self.get_connection()

        cursor = conn.cursor()

        cursor.execute(

            "DELETE FROM documents WHERE document_id=?",
            (document_id,)
        )

        cursor.execute(

            "DELETE FROM metadata WHERE document_id=?",
            (document_id,)
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # COUNT DOCUMENTS
    # --------------------------------------------------

    def total_documents(self):

        conn = self.get_connection()

        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM metadata")

        count = cursor.fetchone()[0]

        conn.close()

        return count

    # --------------------------------------------------
    # CHECK DUPLICATE URL
    # --------------------------------------------------

    def url_exists(self, url: str):

        conn = self.get_connection()

        cursor = conn.cursor()

        cursor.execute(

            "SELECT 1 FROM metadata WHERE url=?",
            (url,)
        )

        exists = cursor.fetchone() is not None

        conn.close()

        return exists

    # --------------------------------------------------
    # CHECK DUPLICATE HASH
    # --------------------------------------------------

    def hash_exists(self, content_hash: str):

        conn = self.get_connection()

        cursor = conn.cursor()

        cursor.execute(

            "SELECT 1 FROM metadata WHERE content_hash=?",
            (content_hash,)
        )

        exists = cursor.fetchone() is not None

        conn.close()

        return exists
    
    def clear_documents(self):

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Delete child records first
            cursor.execute(
                "DELETE FROM documents"
            )

            # Delete parent records
            cursor.execute(
                "DELETE FROM metadata"
            )
            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e

        finally:
            conn.close()

# Singleton object
db = DatabaseManager()