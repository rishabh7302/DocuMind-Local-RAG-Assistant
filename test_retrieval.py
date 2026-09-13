# Step 3: 
# Retrieval Test (Plain Script) — Pehle Ye Check 
# Karte Hain Ki Search Kaam Kar Rahi Hai

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# step A : wahi embedding model load karo jo ingest.py mein use kiya tha
embedding=OllamaEmbeddings(model="nomic-embed-text")

# step B : Saved FAISS index ko disk se load karo
vector_store=FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True # apna khud ka trusted data hai isliye safe hai
) 

# step c : test query do
query ="what are your skills ?" #  <-- isse apne PDF ke topic se related question se replace karo

# step D: top 3 sabse relevant chunks dhoondo
results=vector_store.similarity_search(query,k=3)

#step E : results print karo
print(f"query: {query}\n")
for i,doc in enumerate(results):
    print(f"--chunk{i+1}---")
    print(doc.page_content)
    print(f"(source:{doc.metadata})\n")