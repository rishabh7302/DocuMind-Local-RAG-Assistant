#step 2
# ye script pdf read krega chunka me todega 
# embedding banayega aur FAISS mein save krega

import os
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# step A: Data folder se sari PDFs load karo

loader=DirectoryLoader(
    "data",                 # PDF files jis folder me hain
    glob="*.pdf",           # sirf .pdf files uthao
    loader_cls=PyPDFLoader  # har PDF ko is loader se padhega
      
) 

documents=loader.load()
print(f"Total pages loaded:{len(documents)}")


# step B: Documents ko chhote chunks mein todo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,   # hrr chunk me -800 characters honge
    chunk_overlap=100 # consecutive chubks ke beech 100 characters overalap (context na tute) 
)

chunks=text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# step c: Ollama ka embedding model load karo
embeddings=OllamaEmbeddings(
    model="nomic-embed-text" # text ko vector mein convert karega
)

# step D :Chunks ko embed karke FAISS mein store karo
vector_store=FAISS.from_documents(chunks,embeddings)

# step E : FAISS index ko disk pe save karo (baar baar re-embed na karna pade)
vector_store.save_local("faiss_index")
print("🎉 Faiss index save successfully in 'faiss_index' folder!")