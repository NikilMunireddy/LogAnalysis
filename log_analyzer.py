# log_analyzer.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from database_manager import DatabaseManager

class LogAnalyzer:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def analyze_time_series(self, days: int = 7) -> pd.DataFrame:
        """Analyze logs over time"""
        query = """
        SELECT 
            DATE(timestamp) as date,
            level,
            COUNT(*) as count
        FROM log_entries 
        WHERE timestamp >= ?
        GROUP BY DATE(timestamp), level
        ORDER BY date, level
        """
        
        start_date = datetime.now() - timedelta(days=days)
        conn = self.db_manager.connect()
        if not conn:
            return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()
        return df
    
    def analyze_error_patterns(self, days: int = 7) -> pd.DataFrame:
        """Analyze error patterns and frequencies"""
        query = """
        SELECT 
            level,
            service,
            endpoint,
            status_code,
            COUNT(*) as error_count,
            AVG(response_time) as avg_response_time
        FROM log_entries 
        WHERE timestamp >= ? AND level IN ('ERROR', 'FATAL', 'WARN')
        GROUP BY level, service, endpoint, status_code
        ORDER BY error_count DESC
        LIMIT 20
        """
        
        start_date = datetime.now() - timedelta(days=days)
        conn = self.db_manager.connect()
        if not conn:
            return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()
        return df
    
    def analyze_performance_metrics(self, days: int = 7) -> pd.DataFrame:
        """Analyze performance metrics"""
        db_type = self.db_manager.config.db_type
        percentile_func = self._get_percentile_function(db_type)
        
        query = f"""
        SELECT 
            service,
            endpoint,
            COUNT(*) as request_count,
            AVG(response_time) as avg_response_time,
            {percentile_func} as p95_response_time,
            MAX(response_time) as max_response_time,
            MIN(response_time) as min_response_time
        FROM log_entries 
        WHERE timestamp >= ? AND response_time IS NOT NULL
        GROUP BY service, endpoint
        ORDER BY avg_response_time DESC
        LIMIT 15
        """
        
        start_date = datetime.now() - timedelta(days=days)
        conn = self.db_manager.connect()
        if not conn:
            return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn, params=(start_date,))
        conn.close()
        return df
    
    def _get_percentile_function(self, db_type: str) -> str:
        """Get appropriate percentile function for database type"""
        if db_type == "postgres":
            return "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time)"
        else:
            return """(SELECT response_time FROM log_entries 
                      WHERE response_time IS NOT NULL 
                      ORDER BY response_time 
                      LIMIT 1 OFFSET (SELECT COUNT(*) * 0.95 
                      FROM log_entries WHERE response_time IS NOT NULL))"""
    
    def get_recent_logs(self, limit: int = 100) -> pd.DataFrame:
        """Get recent logs for analysis"""
        query = """
        SELECT 
            timestamp,
            level,
            service,
            message,
            response_time,
            status_code,
            endpoint
        FROM log_entries 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        conn = self.db_manager.connect()
        if not conn:
            return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    
    def generate_log_summary(self, days: int = 7) -> str:
        """Generate comprehensive log summary for AI analysis"""
        start_date = datetime.now() - timedelta(days=days)
        conn = self.db_manager.connect()
        if not conn:
            return ""
            
        try:
            # Get summary statistics
            summary_query = """
            SELECT 
                COUNT(*) as total_logs,
                COUNT(DISTINCT service) as unique_services,
                COUNT(DISTINCT endpoint) as unique_endpoints,
                AVG(response_time) as overall_avg_response_time,
                SUM(CASE WHEN level IN ('ERROR', 'FATAL') THEN 1 ELSE 0 END) as error_count
            FROM log_entries 
            WHERE timestamp >= ?
            """
            summary = pd.read_sql_query(summary_query, conn, params=(start_date,)).iloc[0]
            
            # Get top errors
            errors_query = """
            SELECT message, COUNT(*) as count
            FROM log_entries 
            WHERE level IN ('ERROR', 'FATAL') AND timestamp >= ?
            GROUP BY message
            ORDER BY count DESC
            LIMIT 5
            """
            top_errors = pd.read_sql_query(errors_query, conn, params=(start_date,))
            
            # Prepare context
            context = f"""
            Log Analysis Summary for the last {days} days:
            - Total logs: {summary['total_logs']}
            - Unique services: {summary['unique_services']}
            - Unique endpoints: {summary['unique_endpoints']}
            - Overall average response time: {summary['overall_avg_response_time']:.2f}ms
            - Total errors: {summary['error_count']}
            
            Top error messages:
            {top_errors.to_string(index=False)}
            """
            
            return context
        finally:
            conn.close()