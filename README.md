# QA Chatbot (with Ollama + FAISS)
Overview
This repository features a multi-document question-answering chatbot, enhanced with web scraping and web crawling techniques to streamline document retrieval and eliminate unnecessary manual searches across multiple websites. By leveraging Ollama for LLM-based reasoning and FAISS for efficient vector storage, the chatbot enables users to upload and query PDF, DOCX, and TXT files seamlessly through Streamlit.

### Technologies Used
- Python – The core programming language.
- Streamlit – Provides a web interface for document uploads and querying.
- FAISS – Efficient vector storage and retrieval.
- HuggingFace Embeddings – Converts text into vector representations.
- LangChain – Implements LLM-based retrieval and question answering.
- Ollama – Provides a local model (mistral) for answering queries.
- Web Scraping & Crawling – Automated techniques to collect and structure relevant content from various sources without manual navigation.
- PyPDFLoader, Docx2txtLoader, TextLoader – Extract text from various document formats.
- RecursiveCharacterTextSplitter – Splits long texts into manageable chunks.


### Usage
- Run the Streamlit app:
streamlit run app.py
- Upload your documents (PDF, DOCX, TXT).
- Enter your question in the text input field.
- Get AI-powered answers based on your uploaded documents!
Features
- Automated Web Crawling & Scraping: Efficiently retrieves relevant data from different sources, eliminating the need to manually drill down into various websites.
- Vectorized Search: Documents are stored in a FAISS vector database for optimized retrieval.
- Local LLM Processing: Uses Ollama's Mistral model to generate responses.
