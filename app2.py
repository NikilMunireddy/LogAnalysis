import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import os
import re
import psycopg2
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- Setup ---
st.set_page_config(page_title="DeepSeek & Gemini + Log Analysis", layout="wide")
st.title("📚 Database Log Analysis with Gemini")

VECTORSTORE_DIR = "."
INDEX_NAME = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Gemini
if "gemini_model" not in st.session_state:
    try:
        # Configure Gemini - you'll need to set GOOGLE_API_KEY in your environment
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        st.session_state.gemini_model = genai.GenerativeModel('gemini-pro')
        st.success("✅ Gemini model initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini: {e}")
        st.session_state.gemini_model = None

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Database configuration
if "db_config" not in st.session_state:
    st.session_state.db_config = {
        "db_type": "sqlite",  # Default to SQLite for local testing
        "host": "localhost",
        "database": "log_analysis",
        "user": "postgres",
        "password": "",
        "port": "5432",
        "sqlite_path": "log_analysis.db"
    }

def load_document(file):
    suffix = file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if suffix == "pdf":
        return PyPDFLoader(tmp_path).load()
    elif suffix == "txt":
        with open(tmp_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [Document(page_content=content)]
    elif suffix == "docx":
        return Docx2txtLoader(tmp_path).load()
    else:
        st.error(f"Unsupported file type: {suffix}")
        return []

def connect_to_database():
    """Establish connection to database based on selected type"""
    db_type = st.session_state.db_config["db_type"]
    
    if db_type == "postgres":
        try:
            conn = psycopg2.connect(
                host=st.session_state.db_config["host"],
                database=st.session_state.db_config["database"],
                user=st.session_state.db_config["user"],
                password=st.session_state.db_config["password"],
                port=st.session_state.db_config["port"]
            )
            return conn
        except Exception as e:
            st.error(f"PostgreSQL connection error: {e}")
            return None
    else:  # SQLite
        try:
            conn = sqlite3.connect(st.session_state.db_config["sqlite_path"])
            # Enable foreign keys and better performance
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            return conn
        except Exception as e:
            st.error(f"SQLite connection error: {e}")
            return None

def execute_query(conn, query, params=None):
    """Execute query based on database type"""
    cursor = conn.cursor()
    if st.session_state.db_config["db_type"] == "postgres":
        cursor.execute(query, params)
    else:
        # SQLite uses ? placeholders instead of %s
        if params:
            query = query.replace('%s', '?')
        cursor.execute(query, params)
    return cursor

def setup_database():
    """Create necessary tables for log analysis if they don't exist"""
    conn = connect_to_database()
    if conn:
        try:
            cursor = conn.cursor()
            
            # Determine autoincrement syntax based on database type
            autoinc = "SERIAL PRIMARY KEY" if st.session_state.db_config["db_type"] == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"
            
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
            
            # Create indexes for better performance
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
                    # SQLite doesn't support IF NOT EXISTS in CREATE INDEX, so we'll ignore errors
                    if st.session_state.db_config["db_type"] == "sqlite":
                        pass
                    else:
                        raise
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Database setup error: {e}")
            return False
    return False

def analyze_logs_time_series(conn, days=7):
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
    df = pd.read_sql_query(query, conn, params=(start_date,))
    return df

def analyze_error_patterns(conn, days=7):
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
    df = pd.read_sql_query(query, conn, params=(start_date,))
    return df

def analyze_performance_metrics(conn, days=7):
    """Analyze performance metrics"""
    if st.session_state.db_config["db_type"] == "postgres":
        percentile_func = "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time)"
    else:
        # SQLite approximation for percentile
        percentile_func = "(SELECT response_time FROM log_entries WHERE response_time IS NOT NULL ORDER BY response_time LIMIT 1 OFFSET (SELECT COUNT(*) * 0.95 FROM log_entries WHERE response_time IS NOT NULL))"
    
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
    df = pd.read_sql_query(query, conn, params=(start_date,))
    return df

def get_recent_logs(conn, limit=100):
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
    
    df = pd.read_sql_query(query, conn, params=(limit,))
    return df

def generate_log_insights(conn, days=7):
    """Generate comprehensive log insights using AI"""
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
    
    start_date = datetime.now() - timedelta(days=days)
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
    
    # Prepare context for AI analysis
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

def import_log_file(file):
    """Import log data from a text file"""
    conn = connect_to_database()
    if not conn:
        return False
    
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
        conn.close()
        return imported_count
        
    except Exception as e:
        st.error(f"Error importing log file: {e}")
        return False

def query_gemini(prompt, context=""):
    """Query Gemini model with error handling"""
    if st.session_state.gemini_model is None:
        return "❌ Gemini model is not available. Please check your API key configuration."
    
    try:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = st.session_state.gemini_model.generate_content(full_prompt)
        return response.text
    except google_exceptions.InvalidArgument as e:
        return f"❌ Invalid argument error: {e}"
    except google_exceptions.PermissionDenied as e:
        return f"❌ Permission denied: {e}"
    except google_exceptions.ResourceExhausted as e:
        return f"❌ API quota exceeded: {e}"
    except Exception as e:
        return f"❌ Error querying Gemini: {e}"

# --- Sidebar for Database Configuration ---
with st.sidebar:
    st.header("🔧 Configuration")
    
    st.subheader("AI Model Configuration")
    api_key = st.text_input("Google API Key", type="password", 
                           help="Enter your Google AI Studio API key")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-pro')
            st.success("✅ Gemini configured successfully!")
        except Exception as e:
            st.error(f"❌ Failed to configure Gemini: {e}")
    
    st.subheader("Database Type")
    db_type = st.radio("Select Database", ["sqlite", "postgres"], 
                       index=0 if st.session_state.db_config["db_type"] == "sqlite" else 1)
    st.session_state.db_config["db_type"] = db_type
    
    if db_type == "postgres":
        st.subheader("PostgreSQL Connection")
        st.session_state.db_config["host"] = st.text_input("Host", st.session_state.db_config["host"])
        st.session_state.db_config["database"] = st.text_input("Database", st.session_state.db_config["database"])
        st.session_state.db_config["user"] = st.text_input("User", st.session_state.db_config["user"])
        st.session_state.db_config["password"] = st.text_input("Password", st.session_state.db_config["password"], type="password")
        st.session_state.db_config["port"] = st.text_input("Port", st.session_state.db_config["port"])
    else:
        st.subheader("SQLite Configuration")
        st.session_state.db_config["sqlite_path"] = st.text_input("Database File Path", st.session_state.db_config["sqlite_path"])
        st.info("SQLite database will be created locally for testing")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize Database"):
            if setup_database():
                st.success("✅ Database initialized successfully!")
            else:
                st.error("❌ Failed to initialize database")
    
    with col2:
        if st.button("Clear All Data"):
            conn = connect_to_database()
            if conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM log_entries")
                conn.commit()
                cursor.close()
                conn.close()
                st.success("✅ All log data cleared!")
    
    st.subheader("Import Log Data")
    log_file = st.file_uploader("Upload log file", type=["txt", "log"])
    if log_file and st.button("Import Log File"):
        imported_count = import_log_file(log_file)
        if imported_count:
            st.success(f"✅ Imported {imported_count} log entries!")
        else:
            st.error("❌ Failed to import log file")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📚 Document RAG with Gemini", "📊 Log Analysis", "🤖 AI Log Insights"])

with tab1:
    # --- Document RAG Section for Log Analysis with Gemini ---
    st.header("🔍 Log File Analysis for Microservice RCA with Gemini")
    
    # Try to load persisted index
    index_path = os.path.join(VECTORSTORE_DIR, INDEX_NAME)
    vectorstore = None
    
    if os.path.exists(index_path + ".faiss") and os.path.exists(index_path + ".pkl"):
        try:
            vectorstore = FAISS.load_local(index_path, embedding_model)
            st.success("✅ Loaded existing log analysis index.")
        except Exception as e:
            st.warning(f"⚠️ Failed to load saved index. Rebuilding. ({e})")

    # Process uploaded log files
    uploaded_files = st.file_uploader("Upload Log Files (TXT, LOG, PDF, DOCX)", 
                                    type=["txt", "log", "pdf", "docx"], 
                                    accept_multiple_files=True,
                                    help="Upload application logs, error logs, or system logs for analysis")
    
    if uploaded_files:
        all_docs = []
        for file in uploaded_files:
            docs = load_document(file)
            all_docs.extend(docs)
            st.success(f"✅ Processed {file.name} - {len(docs)} documents")

        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)
            texts = [chunk.page_content for chunk in chunks]

            with st.spinner("🔧 Building log analysis index..."):
                vectorstore = FAISS.from_texts(texts, embedding_model)
                vectorstore.save_local(index_path)

            st.success(f"✅ Log files processed! Created {len(chunks)} analyzable chunks.")

    # Pre-defined RCA queries
    st.subheader("Quick RCA Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Find Errors & Exceptions"):
            st.session_state.predefined_query = "Find all error messages, exceptions, stack traces, and failure patterns in the logs. Identify the most frequent errors and their contexts."
    
    with col2:
        if st.button("⚡ Performance Issues"):
            st.session_state.predefined_query = "Analyze performance issues, slow responses, timeouts, high latency patterns, and resource bottlenecks in the microservices."
    
    with col3:
        if st.button("🔗 Dependency Failures"):
            st.session_state.predefined_query = "Identify service dependencies failures, network issues, database connection problems, API gateway errors, and inter-service communication failures."

    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("🔄 Retry Patterns"):
            st.session_state.predefined_query = "Find retry patterns, circuit breaker triggers, failed retries, and exponential backoff scenarios in the logs."
    
    with col5:
        if st.button("📊 Resource Issues"):
            st.session_state.predefined_query = "Look for memory leaks, CPU spikes, disk space issues, thread pool exhaustion, and garbage collection problems."
    
    with col6:
        if st.button("🎯 Root Cause Summary"):
            st.session_state.predefined_query = "Provide a comprehensive root cause analysis summary including primary failure points, contributing factors, and impact assessment."

    # Initialize predefined_query in session state
    if "predefined_query" not in st.session_state:
        st.session_state.predefined_query = ""

    # Custom query input
    query = st.text_area(
        "Or ask custom questions about your logs:",
        value=st.session_state.predefined_query,
        placeholder="e.g., Why is the payment service failing? What's causing the database timeouts?",
        height=100
    )

    if st.button("🚀 Analyze Logs with Gemini") and query and vectorstore:
        if st.session_state.gemini_model is None:
            st.error("❌ Please configure Gemini API key in the sidebar first.")
        else:
            with st.spinner("🔎 Analyzing logs for root causes with Gemini..."):
                # Perform similarity search
                docs = vectorstore.similarity_search(query, k=5)
                
                if not docs:
                    answer = "❌ I couldn't find any relevant log entries matching your query. Please try different search terms or upload more log files."
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    # Build context from relevant log entries
                    context = "\n\n--- LOG ENTRIES ---\n"
                    for i, doc in enumerate(docs, 1):
                        context += f"\n[ENTRY {i}]:\n{doc.page_content}\n"
                    
                    # Enhanced system prompt for RCA analysis
                    system_prompt = """
You are an expert Site Reliability Engineer (SRE) and DevOps specialist specializing in microservice architecture and log analysis. Your task is to perform Root Cause Analysis (RCA) and provide actionable insights.

ANALYSIS FRAMEWORK:
1. **Problem Identification**: Clearly state the main issue
2. **Root Cause**: Identify the underlying cause with evidence from logs
3. **Impact Assessment**: Describe the service impact
4. **Immediate Fixes**: Provide short-term solutions
5. **Long-term Prevention**: Suggest architectural improvements

SPECIALIZED KNOWLEDGE AREAS:
- Microservice communication patterns
- Database and cache issues
- Network latency and timeouts
- Memory leaks and resource exhaustion
- Configuration errors
- Dependency failures
- Security vulnerabilities

FORMAT REQUIREMENTS:
- Use clear sections with headings
- Provide specific log evidence
- Suggest concrete fixes with priority levels
- Include monitoring and alerting recommendations

If the logs don't contain enough information, be honest about limitations and suggest what additional logging would help.
"""
                    # Enhanced query with structured context
                    enhanced_query = f"""
Based on the following log entries and analysis request, please provide a comprehensive Root Cause Analysis:

ANALYSIS REQUEST: {query}

RELEVANT LOG ENTRIES:
{context}

Please provide a structured analysis following this format:
1. **Problem Summary**: Brief overview of the main issue
2. **Key Findings**: Specific evidence from the logs
3. **Root Cause Analysis**: Underlying causes identified
4. **Immediate Actions**: Short-term fixes to implement now
5. **Long-term Solutions**: Architectural improvements
6. **Monitoring Recommendations**: What to watch for in the future

Be specific and reference the actual log entries in your analysis.
"""
                    
                    try:
                        # Use Gemini for analysis
                        response = query_gemini(enhanced_query, system_prompt)
                        
                        # Append to session history
                        st.session_state.chat_history.append({"role": "user", "content": query})
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"💥 Failed to analyze logs with Gemini: {str(e)}"
                        st.session_state.chat_history.append({"role": "user", "content": query})
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    # Display analysis results
    if vectorstore:
        st.subheader("📋 Analysis Results")
        
        # Clear chat button
        if st.button("Clear Analysis History"):
            st.session_state.chat_history = []
            st.session_state.predefined_query = ""
            st.rerun()
        
        # Display chat history with enhanced formatting
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write("**🔍 Query:**")
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    # Add some basic formatting to the response
                    content = msg["content"]
                    
                    # Create expandable sections for long responses
                    with st.expander("View Full Analysis", expanded=True):
                        st.markdown(content)
                    
                    # Quick actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📋 Copy", key=f"copy_{i}"):
                            st.code(content, language="markdown")
                    with col2:
                        if st.button("⭐ Save", key=f"save_{i}"):
                            # Save to session state for later reference
                            if "saved_insights" not in st.session_state:
                                st.session_state.saved_insights = []
                            st.session_state.saved_insights.append({
                                "timestamp": datetime.now(),
                                "query": st.session_state.chat_history[i-1]["content"] if i > 0 else "Unknown",
                                "analysis": content
                            })
                            st.success("Insight saved!")
                    with col3:
                        if st.button("🔄 Similar Logs", key=f"similar_{i}"):
                            # Find more similar logs based on this analysis
                            if vectorstore and i > 0:
                                user_query = st.session_state.chat_history[i-1]["content"]
                                similar_docs = vectorstore.similarity_search(user_query, k=3)
                                st.info("**Additional relevant logs:**")
                                for doc in similar_docs:
                                    st.text(doc.page_content[:500] + "...")

    else:
        st.info("👆 Upload log files to start analyzing microservice issues and perform root cause analysis.")

    # Saved insights section
    if "saved_insights" in st.session_state and st.session_state.saved_insights:
        st.subheader("💾 Saved Insights")
        for insight in reversed(st.session_state.saved_insights[-5:]):
            with st.expander(f"Analysis from {insight['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Query:** {insight['query']}")
                st.markdown("**Analysis:**")
                st.markdown(insight['analysis'])
                
                if st.button("🗑️ Delete", key=f"delete_{insight['timestamp'].timestamp()}"):
                    st.session_state.saved_insights.remove(insight)
                    st.rerun()

with tab2:
    # --- Log Analysis Section ---
    st.header(f"{'SQLite' if st.session_state.db_config['db_type'] == 'sqlite' else 'PostgreSQL'} Log Analysis Dashboard")
    
    conn = connect_to_database()
    if conn:
        # Database info
        db_type = st.session_state.db_config["db_type"]
        db_info = f"Connected to {db_type.upper()} database"
        if db_type == "sqlite":
            db_info += f" at: {st.session_state.db_config['sqlite_path']}"
        st.success(f"✅ {db_info}")
        
        # Analysis timeframe selection
        col1, col2, col3 = st.columns(3)
        with col1:
            days = st.slider("Analysis timeframe (days)", 1, 30, 7)
        with col2:
            analysis_type = st.selectbox("Analysis Type", 
                                       ["Time Series", "Error Patterns", "Performance Metrics", "Recent Logs"])
        with col3:
            if st.button("Refresh Analysis"):
                st.rerun()
        
        if analysis_type == "Time Series":
            st.subheader("📈 Log Volume Over Time")
            time_data = analyze_logs_time_series(conn, days)
            if not time_data.empty:
                fig = px.line(time_data, x='date', y='count', color='level', 
                             title=f"Log Volume by Level (Last {days} Days)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                total_logs = time_data['count'].sum()
                st.metric("Total Logs", f"{total_logs:,}")
            else:
                st.info("No log data available for the selected timeframe.")
        
        elif analysis_type == "Error Patterns":
            st.subheader("🚨 Error Analysis")
            error_data = analyze_error_patterns(conn, days)
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
        
        elif analysis_type == "Performance Metrics":
            st.subheader("⚡ Performance Analysis")
            perf_data = analyze_performance_metrics(conn, days)
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
        
        elif analysis_type == "Recent Logs":
            st.subheader("🔍 Recent Log Entries")
            recent_logs = get_recent_logs(conn, 100)
            if not recent_logs.empty:
                # Add color coding for log levels
                def colorize_level(level):
                    if level in ['ERROR', 'FATAL']:
                        return "🔴"
                    elif level == 'WARN':
                        return "🟡"
                    else:
                        return "🟢"
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Logs", len(recent_logs))
                with col2:
                    error_count = len(recent_logs[recent_logs['level'].isin(['ERROR', 'FATAL'])])
                    st.metric("Errors", error_count)
                with col3:
                    avg_response = recent_logs['response_time'].mean()
                    st.metric("Avg Response Time", f"{avg_response:.1f}ms")
                with col4:
                    unique_services = recent_logs['service'].nunique()
                    st.metric("Unique Services", unique_services)
                
                st.dataframe(recent_logs, use_container_width=True)
            else:
                st.info("No recent logs available.")
        
        conn.close()
    else:
        st.error("Please configure and connect to database in the sidebar.")

with tab3:
    # --- AI Log Insights Section with Gemini ---
    st.header("🤖 AI-Powered Log Insights with Gemini")
    
    conn = connect_to_database()
    if conn:
        st.subheader("AI Analysis of Log Data")
        
        if st.button("Generate AI Insights with Gemini"):
            if st.session_state.gemini_model is None:
                st.error("❌ Please configure Gemini API key in the sidebar first.")
            else:
                with st.spinner("Analyzing logs with Gemini..."):
                    log_context = generate_log_insights(conn, days=7)
                    
                    analysis_prompt = f"""
As an experienced DevOps engineer and log analysis expert, analyze the following log data and provide:

1. **Key Insights and Patterns**: Identify main trends and anomalies
2. **Potential Issues**: Highlight any concerning patterns or errors
3. **Performance Analysis**: Evaluate response times and service performance
4. **Recommendations**: Provide actionable improvement suggestions
5. **Security Concerns**: Identify any potential security issues

Log Data Summary:
{log_context}

Please provide a comprehensive but concise analysis with specific recommendations.
"""
                    
                    try:
                        insights = query_gemini(analysis_prompt)
                        st.subheader("AI Insights from Gemini")
                        st.write(insights)
                        
                        # Save insights to session state
                        if "log_insights" not in st.session_state:
                            st.session_state.log_insights = []
                        st.session_state.log_insights.append({
                            "timestamp": datetime.now(),
                            "insights": insights
                        })
                        
                    except Exception as e:
                        st.error(f"Failed to generate AI insights: {str(e)}")
        
        # Display previous insights
        if "log_insights" in st.session_state and st.session_state.log_insights:
            st.subheader("Previous Insights")
            for insight in reversed(st.session_state.log_insights[-5:]):  # Show last 5 insights
                with st.expander(f"Insights from {insight['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(insight['insights'])
        
        conn.close()
    else:
        st.error("Please configure and connect to database in the sidebar.")

# --- Sample Data Generator (for testing) ---
with st.sidebar:
    st.subheader("Sample Data")
    if st.button("Generate Sample Log Data"):
        conn = connect_to_database()
        if conn:
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
                placeholders = "?" if st.session_state.db_config["db_type"] == "sqlite" else "%s"
                cursor.executemany(f"""
                    INSERT INTO log_entries 
                    (timestamp, level, service, message, ip_address, user_agent, response_time, status_code, endpoint)
                    VALUES ({placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders}, {placeholders})
                """, sample_data)
                
                conn.commit()
                cursor.close()
                conn.close()
                st.success("✅ Sample log data generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating sample data: {e}")

# --- Database Statistics ---
with st.sidebar:
    st.subheader("Database Statistics")
    if st.button("Show Statistics"):
        conn = connect_to_database()
        if conn:
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
                
                st.write(f"**Total Logs:** {stats['total_logs']:,}")
                st.write(f"**Unique Services:** {stats['unique_services']}")
                st.write(f"**Unique Endpoints:** {stats['unique_endpoints']}")
                st.write(f"**Date Range:** {stats['earliest_log']} to {stats['latest_log']}")
                
                conn.close()
            except Exception as e:
                st.error(f"Error getting statistics: {e}")