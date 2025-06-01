import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.title("Multi-Doc QA Chatbot (with Ollama + FAISS)")

uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

embeddings = HuggingFaceEmbeddings()
vectorstore_path = "vector_store"

def load_file(file):
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file.name)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(file.name)
    elif file.name.endswith(".txt"):
        loader = TextLoader(file.name)
    else:
        return None
    return loader.load()

if uploaded_files:
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.read())
        docs = load_file(file)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(vectorstore_path)
    st.success("Documents uploaded and vector store updated!")

# Load existing vectorstore
elif os.path.exists(vectorstore_path):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings,allow_dangerous_deserialization = True)
else:
    st.warning("Upload documents to get started.")
    st.stop()

# Question answering
query = st.text_input("Ask a question about the uploaded documents:")
if query:
    llm = Ollama(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa.run(query)
    st.write("Answer:", answer)