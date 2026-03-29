# =========================
# IMPORTS
# =========================
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Safe Groq import
try:
    from groq import Groq
except:
    Groq = None

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Resume Assistant", page_icon="🤖", layout="wide")

# =========================
# 🔥 MODERN UI CSS (WORKING)
# =========================
st.markdown("""
<style>

/* ================= LIGHT MODE ================= */
[data-theme="light"] .stApp {
    background-color: #F8F8E1;
    color: #1e293b;
}

[data-theme="light"] section[data-testid="stSidebar"] {
    background-color: #FF90BB;
    color: white;
}

/* User */
[data-theme="light"] [data-testid="stChatMessage"]:nth-child(even) {
    background-color: #FF90BB;
    color: white;
    border-left: 4px solid #8ACCD5;
}

/* Bot */
[data-theme="light"] [data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #FFC1DA;
    color: #1e293b;
    border-left: 4px solid #FF90BB;
}

/* ================= DARK MODE ================= */
[data-theme="dark"] .stApp {
    background-color: #020617;
    color: #e2e8f0;
}

[data-theme="dark"] section[data-testid="stSidebar"] {
    background-color: #BE5985;
    color: white;
}

/* User */
[data-theme="dark"] [data-testid="stChatMessage"]:nth-child(even) {
    background-color: #FF90BB;
    color: black;
    border-left: 4px solid #8ACCD5;
}

/* Bot */
[data-theme="dark"] [data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #1e293b;
    color: #e2e8f0;
    border-left: 4px solid #FF90BB;
}

/* ================= COMMON ================= */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 12px;
}

/* Input */
textarea {
        padding: 12px !important;
    border-radius: 14px !important;
    border: 2px solid #8ACCD5 !important;
}

/* Buttons */
button {
    background-color: #8ACCD5 !important;
    color: #1e293b !important;
    border-radius: 10px !important;
}

button:hover {
    background-color: #FF90BB !important;
    color: white !important;
}

/* Header */
h1 {
    color: #FF90BB;
}

/* Expander */
details {
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)
# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Controls")

    USE_LOCAL = st.toggle("📍 Use Local LLM (Ollama)", value=False)

    mode = "🏠︎ Local (Ollama)" if USE_LOCAL else "☁️ Cloud (Groq)"
    st.markdown(f"### Mode: {mode}")

    st.divider()

    uploaded_file = st.file_uploader("📄 Upload Resume PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(documents)

            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vector_store = Chroma.from_documents(docs, embedder)

        st.success("✅ Resume Loaded!")

# =========================
# GROQ SETUP
# =========================
client = None
if Groq and not USE_LOCAL:
    client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def query_llm(prompt):
    if client is None:
        return "⚠️ Groq not available."

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert resume analyst.
Focus on experience and timeline.
"""
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center; font-size:42px; color:#FF90BB;'>
🤖 AI Resume Assistant
</h1>

<p style='text-align:center; font-size:16px; color:#8ACCD5;'>
Soft AI • Smart Insights • Beautiful UI 🌸
</p>
""", unsafe_allow_html=True)

# =========================
# DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
if question := st.chat_input(" 💬 Ask about experience, skills, projects..."):

    # USER MESSAGE SHOW
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # BOT RESPONSE
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):

            if st.session_state.vector_store is None:
                answer = "⚠️ Upload a PDF first."
                sources = []
            else:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 10}
                )

                enhanced_query = f"""
{question}
Focus on experience, roles, jobs, timeline, years.
"""

                docs = retriever.invoke(enhanced_query)

                # FILTER
                filtered_docs = []
                for doc in docs:
                    if any(k in doc.page_content.lower() for k in [
                        "experience","developer","engineer","2023","2024","2025"
                    ]):
                        filtered_docs.append(doc)

                if filtered_docs:
                    docs = filtered_docs

                unique_docs = list({d.page_content: d for d in docs}.values())
                context = "\n\n".join([d.page_content for d in unique_docs])

                prompt = f"""
Context:
{context}

Question:
{question}
"""

                if USE_LOCAL:
                    from langchain_community.chat_models import ChatOllama
                    llm = ChatOllama(model="llama3")
                    answer = llm.invoke(prompt).content
                else:
                    answer = query_llm(prompt)

                sources = unique_docs[:3]

        st.markdown(answer)

        # CITATIONS
        if sources:
            with st.expander("📄 Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})