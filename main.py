import streamlit as st
import shutil
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from operator import itemgetter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import tempfile

# Constants
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Caching ---
@st.cache_resource
def get_embedding_function():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# Initialize session state
if "articles" not in st.session_state:
    st.session_state.articles = []  # List of dicts: {'url': url, 'title': title}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding_function()
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = "" # Initialize as empty
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Function to add article
def add_article(url):
    try:
        # Prevent adding duplicate URLs
        if any(a['url'] == url for a in st.session_state.articles):
            return None, "This article has already been added."

        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return None, "No content loaded from URL."
        
        # Extract title (simplified; assumes first doc has metadata with title)
        title = docs[0].metadata.get('title', url)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        
        # Add to vectorstore with metadata
        for split in splits:
            split.metadata['source_url'] = url
            split.metadata['title'] = title
        
        st.session_state.vectorstore.add_documents(splits)
        st.session_state.vectorstore.persist()
        
        # Add to articles list
        st.session_state.articles.append({'url': url, 'title': title})
        
        return title, None
    except Exception as e:
        return None, str(e)

# Function to process uploaded files
def process_uploaded_file(uploaded_file):
    """Loads, splits, and adds an uploaded file to the vector store."""
    try:
        # Prevent adding duplicate files by name
        if any(a['title'] == uploaded_file.name for a in st.session_state.articles):
            return None, f"File '{uploaded_file.name}' has already been added."

        # Save to a temporary file to get a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Determine loader based on file type
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        else:
            os.remove(tmp_file_path)
            return None, f"Unsupported file type: {uploaded_file.type}"

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)

        for split in splits:
            split.metadata['source_url'] = f"file://{uploaded_file.name}"
            split.metadata['title'] = uploaded_file.name

        st.session_state.vectorstore.add_documents(splits)
        st.session_state.vectorstore.persist()
        st.session_state.articles.append({'url': f"file://{uploaded_file.name}", 'title': uploaded_file.name})

        return uploaded_file.name, None
    except Exception as e:
        return None, str(e)
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Setup LLM and RAG chain
def setup_rag_chain(api_key):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=GROQ_MODEL,
    )
    
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Chain that returns a dictionary with answer and sources
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            answer=(lambda x: {"context": format_docs(x["context"]), "question": x["question"]}) | prompt | llm | StrOutputParser()
        )
    )
    return rag_chain

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # API Key input
    api_key_input = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key)
    # If key is new or has changed, update state, reset the chain, and confirm.
    if api_key_input and api_key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key_input
        st.session_state.rag_chain = None  # Reset chain to be rebuilt with the new key
        st.success("API Key updated! The chat is ready.")
        st.rerun() # Rerun to update the UI state immediately
    
    # Sources list
    with st.expander("Sources", expanded=True):
        if not st.session_state.articles:
            st.write("No sources added yet.")
        for article in st.session_state.articles:
            if article['url'].startswith('http'):
                st.write(f"- [{article['title']}]({article['url']})")
            else:
                st.write(f"- {article['title']} (File)")
    
    st.divider()

    if st.button("Reset Chat and Articles"):
        # Clear session state
        st.session_state.messages = []
        st.session_state.articles = []
        st.session_state.rag_chain = None
        
        # Delete the persisted vector store directory
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        
        # Re-initialize the vectorstore in session state
        st.session_state.vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=get_embedding_function())
        st.rerun()

# Main UI
st.title("RAG Chatbot with Online Articles")

tab1, tab2 = st.tabs(["Add from URL", "Upload Files"])

with tab1:
    st.subheader("Add New Article from URL")
    url_input = st.text_input("Enter article URL:")
    if st.button("Add Article"):
        if url_input:
            with st.spinner("Processing article..."):
                title, error = add_article(url_input)
                if error:
                    st.error(f"Error adding article: {error}")
                else:
                    st.success(f"Added article: {title}")
        else:
            st.warning("Please enter a URL.")

with tab2:
    st.subheader("Upload and Process Files")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if st.button("Process Uploaded Files"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    title, error = process_uploaded_file(uploaded_file)
                    if error:
                        st.error(f"Error with {uploaded_file.name}: {error}")
                    else:
                        st.success(f"Processed file: {title}")

st.divider()

st.subheader("Chat About Your Articles")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if API key is available
api_key_available = bool(st.session_state.groq_api_key)

if not api_key_available:
    st.warning("Please enter your Groq API key in the sidebar to enable the chat.")

# Chat input, disabled if no API key
if prompt := st.chat_input("Ask a question...", disabled=not api_key_available):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Ensure RAG chain is set up
        if not st.session_state.rag_chain:
            st.session_state.rag_chain = setup_rag_chain(st.session_state.groq_api_key)

        full_response = ""
        context_docs = []
        response_container = st.empty()
        
        try:
            # Stream the response from the RAG chain
            for chunk in st.session_state.rag_chain.stream(prompt):
                if "context" in chunk:
                    context_docs = chunk["context"]
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)

            if context_docs:
                with st.expander("View Sources"):
                    for i, doc in enumerate(context_docs):
                        source_url = doc.metadata.get('source_url', 'N/A')
                        source_title = doc.metadata.get('title', 'N/A')
                        st.write(f"**Source {i+1}: [{source_title}]({source_url})**")
                        st.caption(f"Content: {doc.page_content[:250]}...")
        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            response_container.error(full_response)

    # Add the final response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})