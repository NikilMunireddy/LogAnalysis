# document_processor.py
import os
import tempfile
import re
from typing import List, Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import ModelConfig, AppConfig

class DocumentProcessor:
    def __init__(self, model_config: ModelConfig, app_config: AppConfig):
        self.model_config = model_config
        self.app_config = app_config
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_config.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=model_config.chunk_size,
            chunk_overlap=model_config.chunk_overlap
        )
    
    def load_document(self, file) -> List[Document]:
        """Load document based on file type"""
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                return PyPDFLoader(tmp_path).load()
            elif suffix == "txt":
                with open(tmp_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                return [Document(page_content=content)]
            elif suffix == "docx":
                return Docx2txtLoader(tmp_path).load()
            else:
                st.error(f"Unsupported file type: {suffix}")
                return []
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def process_documents(self, uploaded_files) -> Optional[FAISS]:
        """Process uploaded documents and create vector store"""
        all_docs = []
        for file in uploaded_files:
            all_docs.extend(self.load_document(file))
        
        if not all_docs:
            return None
        
        chunks = self.text_splitter.split_documents(all_docs)
        texts = [chunk.page_content for chunk in chunks]

        with st.spinner("Generating vector index..."):
            vectorstore = FAISS.from_texts(texts, self.embedding_model)
            self._save_vectorstore(vectorstore)
        
        return vectorstore
    
    def load_existing_vectorstore(self) -> Optional[FAISS]:
        """Load existing vector store from disk"""
        index_path = os.path.join(self.app_config.vectorstore_dir, self.app_config.index_name)
        
        if (os.path.exists(index_path + ".faiss") and 
            os.path.exists(index_path + ".pkl")):
            try:
                vectorstore = FAISS.load_local(index_path, self.embedding_model)
                return vectorstore
            except Exception as e:
                st.warning(f"Failed to load saved index: {e}")
        
        return None
    
    def _save_vectorstore(self, vectorstore: FAISS):
        """Save vector store to disk"""
        index_path = os.path.join(self.app_config.vectorstore_dir, self.app_config.index_name)
        vectorstore.save_local(index_path)
    
    def search_documents(self, vectorstore: FAISS, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        return vectorstore.similarity_search(query, k=k)
    
    @staticmethod
    def format_context(docs: List[Document]) -> str:
        """Format documents into context string"""
        return "\n\n".join([f"{doc.page_content}" for doc in docs])