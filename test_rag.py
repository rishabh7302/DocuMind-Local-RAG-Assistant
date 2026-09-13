# Step 4: LLM Se Answer Generate Karna 
# (Retrieval + Generation Jodna)

# Ab hum retrieved chunks ko context ke roop mein 
# LLM (qwen3-vl:2b) ko denge, taaki wo ek proper, 
# natural-language answer bana sake — na ki sirf 
# raw chunks dikhaye.

from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# step A: Embedding model aur FAISS index load karo
embedding =OllamaEmbeddings(model="nomic-embed-text")
vector_store=FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# step B :LLM load karo (answer generate karne ke liye)
llm=ChatOllama(model="qwen3-vl:2b")

# step c: Prompt template banao -LLM ko instructions dene ke liye
prompt=ChatPromptTemplate.from_template("""
You are a help full Ai assistant give answers on the basis of below context
If answer is not available in context just say "There is nothing that related."
Do not made up anything outside of the context (no hallucination).

Context:
{context}

Question:{question}

Answer:
""") 

# step D : user ka query
query="tell me about yourself?"

# step E: Relevant chunks retrieve karo
results=vector_store.similarity_search(query,k=3)

print(results)
# Step F : Retrieval chunks ko ek single text mein jodo
context_text="\n\n".join([doc.page_content for doc in results])

# step G : prompt ko actual data se fill karo

final_prompt=prompt.format(context=context_text,question=query)

# step H : LLM ko bhejo aur answer lo

response = llm.invoke([HumanMessage(content=final_prompt)])

# answer
print("=== FULL RESPONSE OBJECT ===")
print(response)
print("\n=== TYPE ===")
print(type(response))
print("\n=== CONTENT ===")
print(repr(response.content))
