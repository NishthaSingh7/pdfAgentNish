# =========================
# IMPORTS
# =========================
import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS

load_dotenv()

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =========================
# API KEY
# =========================
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = os.getenv("API_KEY")

# =========================
# LLM
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
)

# =========================
# SIDEBAR - PDF UPLOAD
# =========================
with st.sidebar:
    st.header("📄 Upload PDFs")

    if st.button("Reset Knowledge Base"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.success("Knowledge base cleared!")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(file_path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["page"] = doc.metadata.get("page", "Unknown")

                all_docs.extend(documents)
                os.remove(file_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150
            )
            docs = splitter.split_documents(all_docs)

            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=API_KEY
            )

            # ✅ NO persistence (cloud-safe)
            st.session_state.vector_store = Chroma.from_documents(
                docs,
                embedding_model
            )

        st.success("✅ Knowledge base created!")

# =========================
# TOOLS
# =========================
def search_pdf(query):
    if st.session_state.vector_store is None:
        return "No PDF knowledge base found. Please upload PDFs first."
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant info found."
    return "\n\n".join(
        [f"{doc.page_content}" for doc in docs]
    )

pdf_tool = Tool(
    name="PDF Knowledge Base",
    func=search_pdf,
    description="Answer questions from uploaded PDFs."
)

def web_search(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join([f"{r['title']}\n{r['body']}" for r in results])

web_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Useful for general knowledge questions."
)

# =========================
# AGENT
# =========================
tools = [pdf_tool, web_tool]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# =========================
# ROUTING (IMPORTANT 🔥)
# =========================
def is_simple_chat(query):
    casual = ["hi", "hello", "how are you", "who are you"]
    return any(word in query.lower() for word in casual)

# =========================
# MAIN UI
# =========================
st.title("🤖 AI Agent")
st.write("Upload PDFs in sidebar, then chat with AI!")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                # 🔥 ROUTING
                if is_simple_chat(question):
                    response = llm.invoke(question)
                    answer = response.content
                else:
                    response = agent_executor.invoke({"input": question})
                    answer = response.get("output", str(response))

            except Exception:
                # fallback
                response = llm.invoke(question)
                answer = response.content

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})