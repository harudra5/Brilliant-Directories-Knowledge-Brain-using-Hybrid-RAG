
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Brilliant Directories | Knowledge Brain",
    page_icon="🧠",
    layout="wide"
)

#---------------------
# ---------------- CUSTOM UI STYLING ----------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: #f5f7fc;
}

/* Hide Streamlit default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

/* Main content spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Main title */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #172554;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

/* Subtitle */
.main-subtitle {
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 25px;
}

/* Welcome message */
.welcome-card {
    background: linear-gradient(120deg, #e0ecff, #f0eaff);
    border: 1px solid #dbeafe;
    padding: 18px 22px;
    border-radius: 14px;
    color: #1e3a8a;
    margin: 20px 0;
}

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e5eaf3;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 16px;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    border: 1px solid #dbe3f0;
    font-weight: 600;
}

/* Expander / sources */
[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #e2e8f0;
}

/* User and assistant chat distinction */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #eef4ff;
    border: 1px solid #d6e4ff;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff;
    border: 1px solid #e2e8f0;
}

/* Chat text readability */
[data-testid="stChatMessage"] p {
    color: #1e293b;
    line-height: 1.7;
    font-size: 15px;
}

/* Premium chat input */
[data-testid="stChatInput"] {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    transition: all 0.2s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
}

[data-testid="stChatInput"] textarea {
    color: #1e293b;
    font-size: 15px;
}

/* Clear Chat button styling */
[data-testid="stSidebar"] .stButton button {
    background: #ef4444;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #dc2626;
    color: #ffffff !important;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------

#----------------------


# Initialize RAG components
@st.cache_resource
def initialize_rag():

    # Embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Connect to existing ChromaDB
    vector_store = Chroma(
        collection_name="brillient_directories",
        persist_directory=r"C:\Users\user\Knowledge_brain_chroma",
        embedding_function=embedding
    )

    # Load existing documents for BM25
    stored_data = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(
            stored_data["documents"],
            stored_data["metadatas"]
        )
    ]

    # Vector retriever
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    print("Total documents retrieved:", len(documents))

    if not documents:
        raise ValueError("No documents found in ChromaDB. Check vector database loading.")

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    return ensemble_retriever, llm


# Generate RAG response
def RAG_SYSTEM(query):

    retrieved_docs = ensemble_retriever.invoke(query)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('url', 'Not available')}\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    SYSTEM_PROMPT = f"""
    You are Brilliant Directories support assistant.

    Instructions:
    - Answer using only the retrieved context.
    - Never fabricate information.
    - If the answer is unavailable, respond:
      "I couldn't find sufficient information in the retrieved documents."
    - Keep answers clear and concise.
    - Include relevant article links when available.

    Retrieved Context:
    {context}

    Question:
    {query}

    Grounded Answer:
    """

    response = llm.invoke(SYSTEM_PROMPT)

    return response.content, retrieved_docs


# Initialize application
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please check your .env file.")
    st.stop()

ensemble_retriever, llm = initialize_rag()


# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

# Application Header
#=======================
# ---------------- PREMIUM HEADER ----------------

st.html("""
<div style="margin-bottom:30px;">

    <div style="
        display:inline-block;
        background:#e0e7ff;
        color:#4338ca;
        padding:7px 14px;
        border-radius:20px;
        font-size:13px;
        font-weight:600;
        margin-bottom:15px;">
        ✦ AI POWERED CUSTOMER SUPPORT
    </div>

    <h1 style="
        font-size:42px;
        font-weight:800;
        color:#172554;
        margin:0;">
        🧠 Knowledge Brain
    </h1>

    <p style="font-size:18px; color:#475569; margin-top:10px;">
        Brilliant Directories AI Support Assistant
    </p>

    <p style="font-size:14px; color:#94a3b8;">
        Ask questions. Find solutions. Get instant answers from support documentation.
    </p>

</div>
""")

st.divider()

# ---------------- SIDEBAR DESIGN ----------------

with st.sidebar:

    st.markdown("""
        <div style="text-align:center; padding: 20px 5px;">
            <div style="font-size:48px;">🧠</div>
            <h2 style="color:white; margin-bottom:5px;">
                Knowledge Brain
            </h2>
            <p style="color:#94a3b8; font-size:13px;">
                Brilliant Directories
            </p>
        </div>
        <hr style="border-color:#334155;">
    """, unsafe_allow_html=True)

    st.markdown("### ✨ About")

    st.markdown("""
        <p style="color:#cbd5e1; font-size:14px; line-height:1.6;">
            Your AI-powered support assistant.
            Get accurate answers from Brilliant Directories
            documentation.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🚀 Capabilities")
    st.markdown("""
        - 🔍 Intelligent document search
        - 🤖 AI-powered responses
        - 📚 Retrieved source references
        - ⚡ Instant support
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("""
        <div style="text-align:center; color:#94a3b8; font-size:12px;">
            Powered by RAG Architecture<br>
            Built with Streamlit
        </div>
    """, unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Welcome message

if not st.session_state.messages:

    st.markdown("""
    <div class="welcome-card">
        <h3 style="margin-top:0; color:#1e3a8a;">
            👋 Welcome to Knowledge Brain!
        </h3>
        <p style="margin-bottom:0; color:#334155;">
            I'm your Brilliant Directories AI assistant.
            Ask me anything about membership plans, subscriptions,
            registration, billing, and website configuration.
        </p>
    </div>
    """, unsafe_allow_html=True)


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View Retrieved Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}: {doc.metadata.get('title', 'Support Article')}**")

                    url = doc.metadata.get("url")
                    if url:
                        st.markdown(f"[Open Article]({url})")

                    st.caption(doc.page_content[:300])
                    st.divider()


# User input
query = st.chat_input("Ask your question here...")

if query:

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base and generating answer..."):

            try:
                answer, retrieved_docs = RAG_SYSTEM(query)

                st.markdown(answer)

                with st.expander("📚 View Retrieved Sources"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(
                            f"**Source {i}: "
                            f"{doc.metadata.get('title', 'Support Article')}**"
                        )

                        url = doc.metadata.get("url")
                        if url:
                            st.markdown(f"[Open Article]({url})")

                        st.caption(doc.page_content[:300])
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_docs
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
