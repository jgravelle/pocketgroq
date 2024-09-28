from typing import List, Dict, Any, Callable
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import time

class RAGManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None

    def load_and_process_documents(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                                   progress_callback: Callable[[int, int], None] = None, 
                                   timeout: int = 300):  # 5 minutes timeout
        start_time = time.time()
        
        if source.startswith(('http://', 'https://')):
            loader = WebBaseLoader(source)
        elif os.path.isfile(source):
            loader = TextLoader(source)
        else:
            raise ValueError(f"Unsupported source: {source}. Must be a valid URL or file path.")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)
        
        total_chunks = len(split_documents)
        for i, doc in enumerate(split_documents):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Document processing exceeded the {timeout} seconds timeout.")
            
            if not self.vector_store:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            else:
                self.vector_store.add_documents([doc])
            
            if progress_callback:
                progress_callback(i + 1, total_chunks)


    def query_documents(self, llm, query: str) -> Dict[str, Any]:
        if not self.vector_store:
            raise ValueError("Documents have not been loaded. Call load_and_process_documents first.")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context only.
            Provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain.invoke({"input": query})