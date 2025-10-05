# database_manager.py
import sqlite3
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any
import streamlit as st

from config import DatabaseConfig

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    def connect(self):
        """Establish connection to database based on selected type"""
        if self.config.db_type == "postgres":
            try:
                conn = psycopg2.connect(
                    host=self.config.host,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    port=self.config.port
                )
                return conn
            except Exception as e:
                st.error(f"PostgreSQL connection error: {e}")
                return None
        else:  # SQLite
            try:
                conn = sqlite3.connect(self.config.sqlite_path)
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                return conn
            except Exception as e:
                st.error(f"SQLite connection error: {e}")
                return None
    
    def execute_query(self, conn, query: str, params: Optional[Tuple] = None):
        """Execute query based on database type"""
        cursor = conn.cursor()
        if self.config.db_type == "postgres":
            cursor.execute(query, params)
        else:
            if params:
                query = query.replace('%s', '?')
            cursor.execute(query, params)
        return cursor
    
    def setup_database(self) -> bool:
        """Create necessary tables for log analysis if they don't exist"""
        conn = self.connect()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            # Determine autoincrement syntax based on database type
            autoinc = "SERIAL PRIMARY KEY" if self.config.db_type == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"
            
            # Create log_entries table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id {autoinc},
                    timestamp TIMESTAMP,
                    level VARCHAR(20),
                    service VARCHAR(100),
                    message TEXT,
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    response_time FLOAT,
                    status_code INTEGER,
                    endpoint VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            self._create_indexes(cursor)
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Database setup error: {e}")
            return False
    
    def _create_indexes(self, cursor):
        """Create database indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_log_timestamp ON log_entries(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_log_level ON log_entries(level)",
            "CREATE INDEX IF NOT EXISTS idx_log_service ON log_entries(service)",
            "CREATE INDEX IF NOT EXISTS idx_log_status ON log_entries(status_code)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except:
                if self.config.db_type == "sqlite":
                    continue
                raise
    
    def clear_all_data(self) -> bool:
        """Clear all log data from database"""
        conn = self.connect()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM log_entries")
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error clearing data: {e}")
            return False
    
    def get_database_stats(self) -> Optional[pd.Series]:
        """Get database statistics"""
        conn = self.connect()
        if not conn:
            return None
            
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_logs,
                COUNT(DISTINCT service) as unique_services,
                COUNT(DISTINCT endpoint) as unique_endpoints,
                MIN(timestamp) as earliest_log,
                MAX(timestamp) as latest_log
            FROM log_entries
            """
            stats = pd.read_sql_query(stats_query, conn).iloc[0]
            conn.close()
            return stats
        except Exception as e:
            st.error(f"Error getting statistics: {e}")
            return None