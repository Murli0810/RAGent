# 🤖 RAGent

An intelligent AI agent that combines **Retrieval-Augmented Generation (RAG)**, **web search**, **math calculations**, and **real-time awareness** — all powered by Google Gemini and LangChain.

---

## ✨ What RAGent Can Do

| Tool | Capability |
|------|-----------|
| 📄 **RAG Tool** | Answer questions from your own PDF documents |
| 🌐 **Web Search** | Fetch real-time information from the internet via Tavily |
| 🧮 **Calculator** | Evaluate mathematical expressions safely |
| 🕐 **Clock** | Return the current system date and time |
| 🧠 **Memory** | Remember the last 10 messages of your conversation |
| 💬 **General Knowledge** | Answer questions from Gemini's own intelligence |

---

#### Deployed Link: https://ragent-08.streamlit.app/

---
## 🛠️ Tech Stack

- **LLM** — Google Gemini (`gemini-1.5-flash`)
- **Framework** — LangChain
- **Embeddings** — Google Gemini Embeddings (`gemini-embedding-2-preview`)
- **Vector Store** — FAISS
- **PDF Loader** — PyPDF (via LangChain)
- **Web Search** — Tavily AI
- **Agent Pattern** — ReAct (Reasoning + Acting)
> UI built with Streamlit. Frontend styling assisted by AI.
---

## 📁 Project Structure

```
RAGent/
│
├── app.py              # Main application entry point
├── .env                # API keys (never commit this)
├── .gitignore          # Ignores .env and venv
├── requirements.txt    # Python dependencies
└── your_document.pdf   # PDF to chat with
```

---

## ⚙️ Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/Murli0810/RAGent.git
cd RAGent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Git Bash)
source venv/Scripts/activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys here:
- **Google API Key** → [aistudio.google.com](https://aistudio.google.com)
- **Tavily API Key** → [tavily.com](https://tavily.com) *(free tier: 1,000 searches/month)*

### 5. Add your PDF

Place the PDF you want to chat with in the project root and update this line in `app.py`:
```python
loader = PyPDFLoader("your_document.pdf")
```

### 6. Run RAGent

### CLI Version
```bash
python cli_app.py
```
### Web UI Version
```bash
streamlit run streamlit_app.py
```
---

## 💬 Example Conversations

```
You: Who is the CEO of NovaMind?
RAGent: The CEO of NovaMind is Dr. Aisha Patel.

You: Where did she study?
RAGent: She studied at IIT Bombay, where she earned a PhD in Computer Science.

You: What is the latest version of LangChain?
RAGent: [searches web] LangChain is currently on version...

You: What is 15% of 8400?
RAGent: 15% of 8400 is 1260.

You: What time is it?
RAGent: It is 3:56 PM on June 3, 2026.
```

---

## 🧠 How It Works

```
User Question
     ↓
Agent decides which tool to use
     ↓
┌────────────────────────────────┐
│  📄 RAG      → searches PDF    │
│  🌐 Tavily   → searches web    │
│  🧮 Calc     → evaluates math  │
│  🕐 Clock    → gets time       │
└────────────────────────────────┘
     ↓
Tool result passed back to LLM
     ↓
Clean natural response to user
     ↓
Appended to chat history
```

---

## 📦 Requirements

```
langchain
langchain-community
langchain-google-genai
langchain-tavily
google-generativeai
faiss-cpu
pypdf
python-dotenv
asteval
```

Install all at once:
```bash
pip install langchain langchain-community langchain-google-genai langchain-tavily google-generativeai faiss-cpu pypdf python-dotenv numexpr
```

---

## 🚧 Known Limitations

- Processes **one tool per turn** — multi-tool questions (e.g. *"What's the time and NovaMind's revenue?"*) will only answer one part. This will be resolved in **RAGent v2** using LangGraph.
- PDF is loaded and embedded **at startup** — changing the PDF requires a restart.
- Chat history is **in-memory only** — restarting the app clears the conversation.

---

## 🗺️ Roadmap

- [ ] **v2** — Multi-step agent using LangGraph (handles multi-tool questions)
- [ ] **v3** — Streamlit web UI
- [ ] **v4** — Support multiple PDFs simultaneously
- [ ] **v5** — Persistent chat history with database storage

---

## 👨‍💻 Author

Built by **Murli Agarwal** as part of a hands-on AI Engineering learning journey.

Layer 1 of 4 on the path to **AI Application Architect**.

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
