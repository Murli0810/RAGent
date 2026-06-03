import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import re

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2536 !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* ── Main content padding ── */
[data-testid="stMainBlockContainer"] {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #1a2236 !important;
    border: 1px solid #2a3650 !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.85rem 1.1rem !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #111827 !important;
    border: 1px solid #1e2a3d !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 0.85rem 1.1rem !important;
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #161b27 !important;
    border: 1px solid #2a3650 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1a2236 !important;
    border: 1.5px dashed #2a3650 !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1e3a5f !important;
    color: #93c5fd !important;
    border: 1px solid #2d5a9e !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2d5a9e !important;
    color: #bfdbfe !important;
    border-color: #3b82f6 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a3650; border-radius: 4px; }

/* ── Code blocks ── */
code {
    font-family: 'DM Mono', monospace !important;
    background: #1a2236 !important;
    color: #7dd3fc !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
}

/* ── Tool badge ── */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #162032;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #60a5fa;
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
}

/* ── Status indicator ── */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-active { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.status-inactive { background: #475569; }

/* ── Sidebar section ── */
.sidebar-section {
    padding: 1rem 1.2rem;
    border-bottom: 1px solid #1e2536;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #475569;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.tool-row {
    display: flex;
    align-items: center;
    padding: 0.4rem 0;
    font-size: 0.85rem;
    color: #94a3b8;
}
.tool-icon { margin-right: 8px; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "last_tool" not in st.session_state:
    st.session_state.last_tool = None
if "model" not in st.session_state:
    st.session_state.model = None


# ── Initialize model ───────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


# ── PDF processing ─────────────────────────────────────────────────────────────
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)
    return vs, len(chunks)


# ── Tools ──────────────────────────────────────────────────────────────────────
def make_rag_tool(retriever, model):
    @tool
    def RAG_agent(query: str) -> str:
        """Answer questions from the uploaded PDF document. Use this whenever the user asks about document content."""
        template = """You are a helpful assistant. Answer using ONLY the context below.
        If the answer is not in the context, say so clearly.
        Context: {context}"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | model | StrOutputParser()
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        try:
            return chain.invoke({"input": query, "context": context, "history": []})
        except Exception as e:
            return f"Error: {e}"
    return RAG_agent


@tool
def get_current_time() -> str:
    """Returns the current system date and time. Use ONLY when user explicitly asks for time or date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluates a safe mathematical expression. Use for math calculations only."""
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\*\*]+$', expression):
        return "Error: Invalid characters in expression."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div class="sidebar-section" style="padding-top:1.5rem; padding-bottom:1.5rem;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:1.6rem;">⚡</div>
            <div>
                <div style="font-weight:600; font-size:1.1rem; color:#e2e8f0; letter-spacing:-0.02em;">RAGent</div>
                <div style="font-size:0.7rem; color:#475569; font-family:'DM Mono',monospace;">v1.0 · Layer 1</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PDF Upload
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            vs, n_chunks = process_pdf(uploaded_file)
            st.session_state.vector_store = vs
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            st.session_state.chat_history = []
        st.success(f"✓ {n_chunks} chunks indexed")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-top:0.5rem;
                    background:#0f2a1a; border:1px solid #166534; border-radius:8px;
                    padding:0.5rem 0.75rem;">
            <span style="font-size:1rem;">📄</span>
            <span style="font-size:0.8rem; color:#4ade80; font-family:'DM Mono',monospace;
                         white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                         max-width:160px;">{st.session_state.pdf_name}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Tool status
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Tools</div>', unsafe_allow_html=True)

    rag_active = st.session_state.vector_store is not None
    tavily_active = bool(os.getenv("TAVILY_API_KEY"))

    tools_info = [
        ("📄", "RAG Search",     rag_active),
        ("🌐", "Web Search",     tavily_active),
        ("🧮", "Calculator",     True),
        ("🕐", "Clock",          True),
    ]
    for icon, name, active in tools_info:
        dot_class = "status-active" if active else "status-inactive"
        status_text = "ready" if active else "inactive"
        st.markdown(f"""
        <div class="tool-row">
            <span class="tool-icon">{icon}</span>
            <span style="flex:1">{name}</span>
            <span style="display:flex; align-items:center;">
                <span class="status-dot {dot_class}"></span>
                <span style="font-size:0.75rem; font-family:'DM Mono',monospace;
                             color:{'#22c55e' if active else '#475569'};">{status_text}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.last_tool = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats
    if st.session_state.messages:
        turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-label">Session</div>
            <div style="font-size:0.8rem; color:#475569; font-family:'DM Mono',monospace;">
                {turns} message{'s' if turns != 1 else ''} · {len(st.session_state.chat_history)} in memory
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div style="padding: 2rem 0 1.5rem 0; border-bottom: 1px solid #1e2536; margin-bottom: 1.5rem;">
    <div style="font-size: 1.5rem; font-weight: 600; color: #f1f5f9;
                letter-spacing: -0.03em; margin-bottom: 0.25rem;">
        AI Research Assistant
    </div>
    <div style="font-size: 0.85rem; color: #475569;">
        Chat with your documents · Search the web · Do math · Ask anything
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome state
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #94a3b8; margin-bottom: 0.5rem;">
            Upload a PDF to get started
        </div>
        <div style="font-size: 0.85rem; color: #334155;">
            Or just ask a question — RAGent will use web search and its own knowledge
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("tool"):
            tool_icons = {
                "RAG_agent": "📄",
                "tavily_search": "🌐",
                "calculator": "🧮",
                "get_current_time": "🕐",
            }
            icon = tool_icons.get(msg["tool"], "🔧")
            st.markdown(f'<div class="tool-badge">{icon} {msg["tool"].replace("_", " ")}</div>',
                        unsafe_allow_html=True)
        st.markdown(msg["content"])


# ── Chat input & agent logic ───────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build tools list
    model = get_model()
    active_tools = [get_current_time, calculator]

    if st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        rag_tool = make_rag_tool(retriever, model)
        active_tools.insert(0, rag_tool)

    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(
            max_results=3,
            description="Search the internet for real-time or current information not found in the document."
        )
        active_tools.append(tavily)

    model_with_tools = model.bind_tools(active_tools)

    SYSTEM = SystemMessage(content="""You are a helpful AI research assistant with access to tools.
Only call a tool when the user's message explicitly requires it.
Do not call tools for greetings, acknowledgments, or simple conversational replies.""")

    messages = [SYSTEM] + st.session_state.chat_history[-10:] + [HumanMessage(content=prompt)]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = model_with_tools.invoke(messages)
                st.session_state.chat_history.append(HumanMessage(content=prompt))

                if response.tool_calls:
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    tool_icons = {
                        "RAG_agent": "📄",
                        "tavily_search": "🌐",
                        "calculator": "🧮",
                        "get_current_time": "🕐",
                    }
                    icon = tool_icons.get(tool_name, "🔧")
                    st.markdown(f'<div class="tool-badge">{icon} {tool_name.replace("_", " ")}</div>',
                                unsafe_allow_html=True)

                    # Execute tool
                    tool_map = {t.name: t for t in active_tools}
                    if tool_name in tool_map:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    # Final response
                    final_prompt = (f"The user asked: '{prompt}'. "
                                    f"Tool '{tool_name}' returned: '{tool_result}'. "
                                    f"Give a clean, natural response.")
                    final_response = model.invoke(final_prompt)
                    clean_text = (final_response.content if isinstance(final_response.content, str)
                                  else "".join(
                                      b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b))
                                      for b in final_response.content))

                    st.markdown(clean_text)
                    st.session_state.messages.append({
                        "role": "assistant", "content": clean_text, "tool": tool_name
                    })
                    st.session_state.chat_history.append(AIMessage(content=clean_text))

                else:
                    clean_text = (response.content if isinstance(response.content, str)
                                  else "".join(
                                      b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b))
                                      for b in response.content))
                    st.markdown(clean_text)
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                    st.session_state.chat_history.append(AIMessage(content=clean_text))

            except Exception as e:
                err = f"Something went wrong: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})