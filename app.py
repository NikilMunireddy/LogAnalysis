# app.py
import streamlit as st
import ollama
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import re

from config import ConfigManager
from database_manager import DatabaseManager
from log_analyzer import LogAnalyzer
from document_processor import DocumentProcessor

class LogAnalysisApp:
    def __init__(self):
        self.config_manager = ConfigManager()
        self._initialize_session_state()
        
        # Initialize managers
        self.db_manager = DatabaseManager(self.config_manager.db_config)
        self.log_analyzer = LogAnalyzer(self.db_manager)
        self.doc_processor = DocumentProcessor(
            self.config_manager.model_config, 
            self.config_manager.app_config
        )
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "log_insights" not in st.session_state:
            st.session_state.log_insights = []
        
        if "app_config" not in st.session_state:
            st.session_state.app_config = self.config_manager.to_dict()
        else:
            # Update config manager with session state
            self.config_manager.from_dict(st.session_state.app_config)
    
    def render_sidebar(self):
        """Render configuration sidebar"""
        with st.sidebar:
            st.header("🔧 Configuration")
            self._render_database_config()
            self._render_database_actions()
            self._render_import_section()
            self._render_sample_data()
            self._render_statistics()
    
    def _render_database_config(self):
        """Render database configuration section"""
        st.subheader("Database Type")
        db_type = st.radio(
            "Select Database", 
            ["sqlite", "postgres"], 
            index=0 if self.config_manager.db_config.db_type == "sqlite" else 1,
            key="db_type_radio"
        )
        self.config_manager.db_config.db_type = db_type
        
        if db_type == "postgres":
            self._render_postgres_config()
        else:
            self._render_sqlite_config()
    
    def _render_postgres_config(self):
        """Render PostgreSQL configuration"""
        st.subheader("PostgreSQL Connection")
        self.config_manager.db_config.host = st.text_input(
            "Host", 
            self.config_manager.db_config.host,
            key="pg_host"
        )
        self.config_manager.db_config.database = st.text_input(
            "Database", 
            self.config_manager.db_config.database,
            key="pg_database"
        )
        self.config_manager.db_config.user = st.text_input(
            "User", 
            self.config_manager.db_config.user,
            key="pg_user"
        )
        self.config_manager.db_config.password = st.text_input(
            "Password", 
            self.config_manager.db_config.password, 
            type="password",
            key="pg_password"
        )
        self.config_manager.db_config.port = st.text_input(
            "Port", 
            self.config_manager.db_config.port,
            key="pg_port"
        )
    
    def _render_sqlite_config(self):
        """Render SQLite configuration"""
        st.subheader("SQLite Configuration")
        self.config_manager.db_config.sqlite_path = st.text_input(
            "Database File Path", 
            self.config_manager.db_config.sqlite_path,
            key="sqlite_path"
        )
        st.info("SQLite database will be created locally for testing")
    
    def _render_database_actions(self):
        """Render database action buttons"""
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Initialize Database"):
                if self.db_manager.setup_database():
                    st.success("✅ Database initialized successfully!")
                else:
                    st.error("❌ Failed to initialize database")
        
        with col2:
            if st.button("Clear All Data"):
                if self.db_manager.clear_all_data():
                    st.success("✅ All log data cleared!")
                else:
                    st.error("❌ Failed to clear data")
    
    def _render_import_section(self):
        """Render log import section"""
        st.subheader("Import Log Data")
        log_file = st.file_uploader("Upload log file", type=["txt", "log"], key="log_uploader")
        if log_file and st.button("Import Log File"):
            imported_count = self._import_log_file(log_file)
            if imported_count:
                st.success(f"✅ Imported {imported_count} log entries!")
            else:
                st.error("❌ Failed to import log file")
    
    def _import_log_file(self, file) -> int:
        """Import log data from file"""
        conn = self.db_manager.connect()
        if not conn:
            return 0
            
        try:
            content = file.getvalue().decode("utf-8")
            lines = content.split('\n')
            
            cursor = conn.cursor()
            imported_count = 0
            
            for line in lines:
                if line.strip():
                    # Simple log parsing - adjust this based on your log format
                    try:
                        # Example log format: "2024-01-01 10:00:00 INFO web-server /api/users 200 150ms"
                        parts = line.split()
                        if len(parts) >= 6:
                            timestamp_str = f"{parts[0]} {parts[1]}"
                            level = parts[2]
                            service = parts[3]
                            endpoint = parts[4]
                            status_code = int(parts[5])
                            response_time = float(parts[6].replace('ms', ''))
                            
                            # Insert into database
                            cursor.execute("""
                                INSERT INTO log_entries 
                                (timestamp, level, service, endpoint, status_code, response_time, message)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (timestamp_str, level, service, endpoint, status_code, response_time, line))
                            imported_count += 1
                    except:
                        # Skip malformed lines
                        continue
            
            conn.commit()
            cursor.close()
            return imported_count
            
        except Exception as e:
            st.error(f"Error importing log file: {e}")
            return 0
        finally:
            conn.close()
    
    def _render_sample_data(self):
        """Render sample data section"""
        st.subheader("Sample Data")
        if st.button("Generate Sample Log Data"):
            self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample log data"""
        conn = self.db_manager.connect()
        if not conn:
            return
            
        try:
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM log_entries")
            
            # Generate sample log data
            services = ['web-server', 'api-gateway', 'auth-service', 'database', 'cache-service']
            levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
            endpoints = ['/api/users', '/api/auth/login', '/api/products', '/health', '/metrics']
            
            sample_data = []
            base_time = datetime.now() - timedelta(days=7)
            
            for i in range(1000):
                timestamp = base_time + timedelta(minutes=i*10)
                level = levels[i % len(levels)]
                service = services[i % len(services)]
                endpoint = endpoints[i % len(endpoints)]
                status_code = 200 if level in ['INFO', 'DEBUG'] else (400 if level == 'WARN' else 500)
                response_time = max(10, min(500, (i % 100) * 5))
                
                sample_data.append((
                    timestamp, level, service, 
                    f"Sample log message {i} for {service}",
                    f"192.168.1.{i % 255}",
                    f"Mozilla/5.0 Sample User Agent {i}",
                    response_time,
                    status_code,
                    endpoint
                ))
            
            # Insert sample data
            placeholders = "?" if self.config_manager.db_config.db_type == "sqlite" else "%s"
            cursor.executemany(f"""
                INSERT INTO log_entries 
                (timestamp, level, service, message, ip_address, user_agent, response_time, status_code, endpoint)
                VALUES ({placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders})
            """, sample_data)
            
            conn.commit()
            cursor.close()
            st.success("✅ Sample log data generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
        finally:
            conn.close()
    
    def _render_statistics(self):
        """Render database statistics"""
        st.subheader("Database Statistics")
        if st.button("Show Statistics"):
            stats = self.db_manager.get_database_stats()
            if stats is not None:
                st.write(f"**Total Logs:** {stats['total_logs']:,}")
                st.write(f"**Unique Services:** {stats['unique_services']}")
                st.write(f"**Unique Endpoints:** {stats['unique_endpoints']}")
                st.write(f"**Date Range:** {stats['earliest_log']} to {stats['latest_log']}")
    
    def render_document_rag_tab(self):
        """Render Document RAG tab"""
        st.header("Document Analysis with RAG")
        
        # Load or create vector store
        vectorstore = self.doc_processor.load_existing_vectorstore()
        if vectorstore:
            st.success("✅ Loaded existing vector index.")
        
        # Process uploaded files
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX)", 
            type=["pdf", "txt", "docx"], 
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        if uploaded_files:
            vectorstore = self.doc_processor.process_documents(uploaded_files)
            if vectorstore:
                st.success("Documents processed and indexed!")
        
        # Query interface
        if vectorstore:
            self._render_chat_interface(vectorstore)
    
    def _render_chat_interface(self, vectorstore):
        """Render chat interface for document queries"""
        query = st.chat_input("Ask a question about your documents:")
        
        if query:
            docs = self.doc_processor.search_documents(vectorstore, query)
            
            if not docs:
                answer = "I couldn't find any relevant information in the documents."
                self._add_to_chat_history("user", query)
                self._add_to_chat_history("assistant", answer)
            else:
                context = self.doc_processor.format_context(docs)
                answer = self._get_ai_response(query, context)
                self._add_to_chat_history("user", query)
                self._add_to_chat_history("assistant", answer)
        
        # Display chat history
        self._display_chat_history()
    
    def _get_ai_response(self, query: str, context: str) -> str:
        """Get AI response using Ollama"""
        system_prompt = """
You are a strict assistant. Your answers must be based solely on the provided context. 
If the answer is not present in the context, respond ONLY with: "I couldn't find any relevant information in the documents."
"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in st.session_state.chat_history:
            messages.append(msg)
        
        # Inject current query with context
        full_query = f"""CONTEXT:\n\"\"\"\n{context}\n\"\"\"\n\nQUESTION:\n{query}\n\nANSWER:"""
        messages.append({"role": "user", "content": full_query})
        
        try:
            with st.spinner("Querying AI..."):
                response = ollama.chat(
                    model=self.config_manager.model_config.name,
                    messages=messages
                )
                raw_answer = response['message']['content']
                # Clean response (remove thinking tags if any)
                cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                return cleaned_answer
        except Exception as e:
            return f"💥 Failed to get response from AI: {str(e)}"
    
    def _add_to_chat_history(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_history.append({"role": role, "content": content})
    
    def _display_chat_history(self):
        """Display chat history"""
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["content"])
    
    def render_log_analysis_tab(self):
        """Render Log Analysis tab"""
        st.header(f"{self.config_manager.db_config.db_type.upper()} Log Analysis Dashboard")
        
        conn = self.db_manager.connect()
        if not conn:
            st.error("Please configure and connect to database in the sidebar.")
            return
        
        st.success(f"✅ Connected to {self.config_manager.db_config.db_type.upper()} database")
        conn.close()
        
        # Analysis controls
        analysis_type, days = self._render_analysis_controls()
        
        # Perform analysis
        self._perform_analysis(analysis_type, days)
    
    def _render_analysis_controls(self):
        """Render analysis controls"""
        col1, col2, col3 = st.columns(3)
        with col1:
            days = st.slider("Analysis timeframe (days)", 1, 30, 7, key="analysis_days")
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type", 
                ["Time Series", "Error Patterns", "Performance Metrics", "Recent Logs"],
                key="analysis_type"
            )
        with col3:
            if st.button("Refresh Analysis", key="refresh_analysis"):
                st.rerun()
        
        return analysis_type, days
    
    def _perform_analysis(self, analysis_type: str, days: int):
        """Perform the selected analysis"""
        if analysis_type == "Time Series":
            self._render_time_series_analysis(days)
        elif analysis_type == "Error Patterns":
            self._render_error_analysis(days)
        elif analysis_type == "Performance Metrics":
            self._render_performance_analysis(days)
        elif analysis_type == "Recent Logs":
            self._render_recent_logs()
    
    def _render_time_series_analysis(self, days: int):
        """Render time series analysis"""
        st.subheader("📈 Log Volume Over Time")
        time_data = self.log_analyzer.analyze_time_series(days)
        
        if not time_data.empty:
            fig = px.line(time_data, x='date', y='count', color='level', 
                         title=f"Log Volume by Level (Last {days} Days)")
            st.plotly_chart(fig, use_container_width=True)
            
            total_logs = time_data['count'].sum()
            st.metric("Total Logs", f"{total_logs:,}")
        else:
            st.info("No log data available for the selected timeframe.")
    
    def _render_error_analysis(self, days: int):
        """Render error analysis"""
        st.subheader("🚨 Error Analysis")
        error_data = self.log_analyzer.analyze_error_patterns(days)
        
        if not error_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(error_data.head(10), x='error_count', y='endpoint', 
                            color='level', orientation='h',
                            title="Top Error Endpoints")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(error_data, values='error_count', names='service',
                            title="Error Distribution by Service")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Error Data")
            st.dataframe(error_data)
        else:
            st.info("No error data available for the selected timeframe.")
    
    def _render_performance_analysis(self, days: int):
        """Render performance analysis"""
        st.subheader("⚡ Performance Analysis")
        perf_data = self.log_analyzer.analyze_performance_metrics(days)
        
        if not perf_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(perf_data.head(10), x='avg_response_time', y='endpoint',
                            title="Average Response Time by Endpoint")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(perf_data, x='request_count', y='avg_response_time',
                                size='p95_response_time', color='service',
                                title="Request Count vs Response Time")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Performance Metrics Details")
            st.dataframe(perf_data)
        else:
            st.info("No performance data available for the selected timeframe.")
    
    def _render_recent_logs(self):
        """Render recent logs"""
        st.subheader("🔍 Recent Log Entries")
        recent_logs = self.log_analyzer.get_recent_logs(100)
        
        if not recent_logs.empty:
            self._render_log_metrics(recent_logs)
            st.dataframe(recent_logs, use_container_width=True)
        else:
            st.info("No recent logs available.")
    
    def _render_log_metrics(self, logs_df: pd.DataFrame):
        """Render log metrics"""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Logs", len(logs_df))
        with col2:
            error_count = len(logs_df[logs_df['level'].isin(['ERROR', 'FATAL'])])
            st.metric("Errors", error_count)
        with col3:
            avg_response = logs_df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response:.1f}ms")
        with col4:
            unique_services = logs_df['service'].nunique()
            st.metric("Unique Services", unique_services)
    
    def render_ai_insights_tab(self):
        """Render AI Insights tab"""
        st.header("🤖 AI-Powered Log Insights")
        
        conn = self.db_manager.connect()
        if not conn:
            st.error("Please configure and connect to database in the sidebar.")
            return
        
        st.subheader("AI Analysis of Log Data")
        
        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing logs with AI..."):
                log_context = self.log_analyzer.generate_log_summary(days=7)
                insights = self._generate_ai_insights(log_context)
                
                if insights:
                    st.subheader("AI Insights")
                    st.write(insights)
                    
                    # Save insights
                    st.session_state.log_insights.append({
                        "timestamp": datetime.now(),
                        "insights": insights
                    })
        
        # Display previous insights
        self._render_previous_insights()
        
        conn.close()
    
    def _generate_ai_insights(self, log_context: str) -> str:
        """Generate AI insights from log context"""
        system_prompt = """
You are an experienced DevOps engineer and log analysis expert. Analyze the provided log data and provide:
1. Key insights and patterns
2. Potential issues or anomalies
3. Recommendations for improvement
4. Security concerns if any

Be concise but thorough in your analysis.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze this log data:\n\n{log_context}"}
        ]
        
        try:
            response = ollama.chat(
                model=self.config_manager.model_config.name,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            st.error(f"Failed to generate AI insights: {str(e)}")
            return ""
    
    def _render_previous_insights(self):
        """Render previous AI insights"""
        if st.session_state.log_insights:
            st.subheader("Previous Insights")
            for insight in reversed(st.session_state.log_insights[-5:]):
                with st.expander(f"Insights from {insight['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(insight['insights'])
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="DeepSeek & Ollama + Log Analysis", 
            layout="wide"
        )
        st.title("📚 Database Log Analysis")
        
        # Render sidebar
        self.render_sidebar()
        
        # Update session state with current config
        st.session_state.app_config = self.config_manager.to_dict()
        
        # Create main tabs
        tab1, tab2, tab3 = st.tabs(["📚 Document RAG", "📊 Log Analysis", "🤖 AI Log Insights"])
        
        with tab1:
            self.render_document_rag_tab()
        
        with tab2:
            self.render_log_analysis_tab()
        
        with tab3:
            self.render_ai_insights_tab()

# Run the application
if __name__ == "__main__":
    app = LogAnalysisApp()
    app.run()