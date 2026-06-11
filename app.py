import streamlit as st
from rag import ask_rag
from scraper import crawl_website
from chunker import chunk_pages
from vectorstore import build_vectorstore

# Set page configuration
st.set_page_config(
    page_title="RAG Web Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="auto"  # Automatically collapses on mobile, expanded on desktop
)

# Custom CSS for a clean, premium, modern, and responsive aesthetic
st.markdown("""
<style>
    /* Load professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700;800&display=swap');

    /* Global layout modifications */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        animation: fadeIn 0.5s ease-out;
        padding-top: 2rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Gradient animated header box */
    .header-container {
        background: linear-gradient(-45deg, #1e3c72, #2a5298, #1d4ed8, #1e3c72);
        background-size: 300% 300%;
        animation: gradientShift 12s ease infinite;
        padding: 2.2rem 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px rgba(30, 60, 114, 0.15);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin: 0 !important;
        color: white !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.6rem;
        color: #f1f5f9;
        font-weight: 400;
        max-width: 750px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Sidebar styling refinements */
    .sidebar-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 1.25rem;
        background: linear-gradient(135deg, #1e3c72, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    html[data-theme="dark"] .sidebar-title {
        background: linear-gradient(135deg, #f1f5f9, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Style global element primitives */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    .stTextInput input, .stNumberInput input {
        border-radius: 8px !important;
    }
    
    /* Live Status Pulse CSS */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseAnimation 1.6s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Onboarding Card styling */
    .onboarding-card {
        background: rgba(30, 60, 114, 0.03);
        padding: 2.2rem 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(30, 60, 114, 0.08);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.015);
    }
    html[data-theme="dark"] .onboarding-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* --- RESPONSIVENESS & TEXTBOX DEPTH ADJUSTMENTS --- */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8rem !important;
        }
        .header-subtitle {
            font-size: 0.95rem !important;
        }
        .header-container {
            padding: 1.5rem 1rem !important;
            margin-bottom: 1.5rem !important;
        }
    }

    /* Scale down metrics inside the sidebar to fit 2 columns cleanly */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }

    /* Adjust the textbox depth to be a little higher on first run */
    div[data-testid="stTextInput"] input {
        height: 3.2rem !important;
        font-size: 1.05rem !important;
        padding: 0.5rem 1rem !important;
    }
    div[data-testid="stNumberInput"] input {
        height: 3.2rem !important;
        font-size: 1.05rem !important;
    }
    div[data-testid="stChatInput"] textarea {
        min-height: 3.8rem !important;
        font-size: 1.05rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Render Custom Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">💬 Web RAG Chatbot</h1>
    <p class="header-subtitle">Scrape any website and converse with its content instantly using Weaviate Hybrid Search</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session States
if "ready" not in st.session_state:
    st.session_state.ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

# Sidebar - Configuration and Ingestion
with st.sidebar:
    st.markdown('<p class="sidebar-title">🌐 Knowledge base</p>', unsafe_allow_html=True)
    
    if st.session_state.ready:
        st.markdown('<div class="status-badge"><span class="pulse-dot"></span>Index Live</div>', unsafe_allow_html=True)
        st.markdown("### 📊 Index Statistics")
        
        # Clean dashboard columns inside the sidebar
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages Crawled", st.session_state.num_pages)
        with col2:
            st.metric("Vector Chunks", st.session_state.num_chunks)
            
        st.caption(f"**Target:** {st.session_state.current_url}")
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Option to index a different site
        with st.expander("🔄 Index a Different Website", expanded=False):
            new_url = st.text_input(
                "New Website URL",
                placeholder="https://example.com",
                key="side_url"
            )
            new_max_pages = st.number_input(
                "Max Pages to Crawl",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="side_max_pages"
            )
            if st.button("Build New Index", use_container_width=True, type="primary"):
                if not new_url:
                    st.error("Please enter a URL first.")
                else:
                    status = st.empty()
                    try:
                        status.info("🕷️ Crawling...")
                        pages = crawl_website(new_url, maxpages=new_max_pages)
                        chunks = chunk_pages(pages)
                        if not chunks:
                            status.error("❌ No content extracted.")
                        else:
                            st.session_state.num_pages = len(pages)
                            st.session_state.num_chunks = len(chunks)
                            st.session_state.current_url = new_url
                            build_vectorstore(chunks)
                            status.success("✅ Re-indexed!")
                            st.rerun()
                    except Exception as e:
                        status.error(f"❌ Error: {e}")
                        
        st.markdown("---")
        # Reset Session Button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.markdown('<div class="status-badge" style="background:rgba(245,158,11,0.1); color:#f59e0b;"><span class="pulse-dot" style="background-color:#f59e0b; box-shadow:0 0 0 0 rgba(245,158,11,0.7)"></span>Awaiting Setup</div>', unsafe_allow_html=True)
        st.info("💡 Complete the setup form in the main panel to start chatting.")

# Main Body Panel (Onboarding Form or Chat Interface)
if not st.session_state.ready:
    # Onboarding Form in the Center (extremely responsive on mobile)
    col_left, col_center, col_right = st.columns([1, 4, 1])
    with col_center:
        st.markdown("""
        <div class="onboarding-card">
            <div style="font-size: 2.8rem; margin-bottom: 0.8rem;">🌐</div>
            <h3 style="margin: 0 0 0.5rem 0; font-family: 'Outfit', sans-serif; font-weight: 700; color: #1e3c72; font-size: 1.5rem;">Setup Your Knowledge Base</h3>
            <p style="margin: 0; color: #64748b; font-size: 0.95rem; line-height: 1.5;">Enter the URL of the website you want to scrape and query. Once processed, you can converse with its content instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the root URL of the website you want to index.",
            key="main_url"
        )
        
        max_pages = st.number_input(
            "Max Pages to Crawl",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Limit the number of pages to scrape concurrently.",
            key="main_max_pages"
        )
        
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        process_btn = st.button("Process & Index Site", use_container_width=True, type="primary", key="main_process_btn")
        
        if process_btn:
            if not url:
                st.error("Please enter a URL first.")
            else:
                status = st.empty()
                try:
                    status.info("🕷️ Scraping pages in parallel...")
                    pages = crawl_website(url, maxpages=max_pages)
                    
                    status.info("✂️ Splitting content into chunks...")
                    chunks = chunk_pages(pages)
                    
                    if not chunks:
                        status.error("❌ Could not extract text from this website. Try another one.")
                    else:
                        st.session_state.num_pages = len(pages)
                        st.session_state.num_chunks = len(chunks)
                        st.session_state.current_url = url
                        
                        status.info("⚡ Embedding and indexing into Weaviate...")
                        build_vectorstore(chunks)
                        
                        st.session_state.ready = True
                        status.success("✅ Knowledge base built successfully!")
                        st.rerun()
                except Exception as e:
                    status.error(f"❌ Error indexing site: {e}")
else:
    # Chat Panel (Unlocks when ready)
    # Render chat message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Reference Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"- [{src}]({src})")
                        
    # Chat input box at the bottom
    if prompt := st.chat_input("Ask a question about the indexed website..."):
        # Display user query
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Call RAG pipeline and display response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    result = ask_rag(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 View Reference Sources", expanded=False):
                            for src in sources:
                                st.markdown(f"- [{src}]({src})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    err_msg = f"Sorry, I encountered an error while searching the database: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg
                    })
