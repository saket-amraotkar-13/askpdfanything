import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Ask me from document", page_icon="📄")
# st.title("Ask me from document")

# ── Session state ──────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None          # chain only built after file is added

def clear_history():
    # fix: was deleting 'history' — correct key is 'store' and 'messages'
    st.session_state.store = {}
    st.session_state.messages = []
    st.session_state.chain = None


# ── Sidebar: file upload ───────────────────────────────────────────────────────
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a file to query:", type=['pdf', 'txt', 'docx'])
    add_file = st.button('Add File', on_click=clear_history)


# ── Load, chunk, embed — only when file is added ──────────────────────────────
# fix: was outside if-block → crashed when no file uploaded
if uploaded_file and add_file:
    # Save uploaded file to disk
    file_path = os.path.join('./', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    # fix: select loader based on file type
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'docx':
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    document = loader.load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(document)

    retriever = FAISS.from_documents(
        chunks, OpenAIEmbeddings(api_key=api_key)
    ).as_retriever()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=api_key)

    # Contextualize chain — rewrites follow-up → standalone question
    contextualize_chain = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user question as a standalone question using chat history. Return it as-is if already standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]) | llm | StrOutputParser()

    def get_context(input: dict):
        if input.get("chat_history"):
            return retriever.invoke(contextualize_chain.invoke(input))
        return retriever.invoke(input["input"])

    # QA chain
    qa_chain = ChatPromptTemplate.from_messages([
        ("system", "Answer using the context below. Say 'I don't know' if unsure.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]) | llm | StrOutputParser()

    rag_chain = RunnablePassthrough.assign(context=get_context) | qa_chain

    def get_history(session_id):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # fix: store chain in session_state so it survives reruns
    st.session_state.chain = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    st.sidebar.success(f"✅ '{uploaded_file.name}' loaded successfully!")


# ── Streamlit chat UI ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# fix: guard chat input — only active after file is loaded
if st.session_state.chain is None:
    st.info("👈 Upload a file from the sidebar and click **Add File** to start chatting.")
else:
    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": "default"}}
                )
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()
