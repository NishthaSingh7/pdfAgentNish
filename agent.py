# =========================
# IMPORTS
# =========================
from langchain.agents import create_react_agent, AgentExecutor
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS

load_dotenv()

PERSIST_DIRECTORY = "vector_store"

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =========================
# LLM + PROMPT
# =========================
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
      google_api_key=API_KEY,
)
prompt = hub.pull("hwchase17/react")

# =========================
# SIDEBAR - PDF UPLOAD
# =========================
with st.sidebar:
    st.header("📄 Upload PDFs")

    if st.button("Reset Knowledge Base"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        st.success("Knowledge base cleared!")

    # Load existing vector store from disk
    if st.session_state.vector_store is None and os.path.exists(PERSIST_DIRECTORY):
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("API_KEY")
        )
        st.session_state.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model
        )
        st.success("✅ Loaded existing knowledge base!")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.vector_store is None:
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
                google_api_key=os.getenv("API_KEY")
            )

            st.session_state.vector_store = Chroma.from_documents(
                docs,
                embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            st.session_state.vector_store.persist()

        st.success("✅ Knowledge base created!")

# =========================
# TOOLS
# =========================

# TOOL 1 - PDF Search
def search_pdf(query):
    if st.session_state.vector_store is None:
        return "No PDF knowledge base found. Please upload PDFs first."
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the PDFs."
    return "\n\n".join(
        [f"Source: {doc.metadata['source']} (Page {doc.metadata['page']})\n{doc.page_content}" for doc in docs]
    )

pdf_tool = Tool(
    name="PDF Knowledge Base",
    func=search_pdf,
    description="Useful for answering questions from uploaded PDF documents."
)

# TOOL 2 - Web Search
def web_search(query):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join([f"{r['title']}\n{r['body']}" for r in results])

web_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Useful for answering questions about general knowledge not found in PDFs."
)

# =========================
# AGENT
# =========================
tools = [pdf_tool, web_tool]
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# =========================
# MAIN UI
# =========================
st.title("🤖 AI Agent")
st.write("Upload PDFs in the sidebar, then ask me anything!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if question := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": question})
            answer = response["output"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})