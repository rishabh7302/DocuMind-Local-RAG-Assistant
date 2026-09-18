import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 PDF Document Q&A")

# Step A: File uploader - multiple PDFs allow karo
uploaded_files = st.file_uploader(
    "Apni PDF(s) upload karo",
    type="pdf",
    accept_multiple_files=True
)

# Step B: Process button
process_clicked = st.button("Process PDFs")

# Step C: Jab button dabaya jaye, tabhi processing karo
if process_clicked and uploaded_files:
    with st.spinner("PDFs process ho rahi hain..."):
        all_chunks = []

        for uploaded_file in uploaded_files:
            # Uploaded file ko temporarily disk pe save karo
            # (kyunki PyPDFLoader ko ek file PATH chahiye, raw bytes nahi)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # PDF load karo
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            # Original filename metadata mein daal do (temp naam ke bajaye)
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name

            all_chunks.extend(chunks)

            # Temp file delete kar do (cleanup)
            os.remove(tmp_path)

        # Embeddings banao aur FAISS index banao
        embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=st.secrets["OLLAMA_URL"],
        client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"}}
)
        vector_store = FAISS.from_documents(all_chunks, embeddings)

        # session_state mein store karo - taaki reruns ke beech yaad rahe
        st.session_state.vector_store = vector_store
        st.session_state.processed = True

    st.success(f"✅ {len(uploaded_files)} PDF(s) process ho gayi! Ab sawaal pucho.")

# Step D: LLM ko cache karo (ye baar baar reload nahi hoga)
@st.cache_resource
def load_llm():
       return ChatOllama(
        model="qwen3-vl:2b",
        base_url=st.secrets["OLLAMA_URL"],
        client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"}}
        )

llm = load_llm()

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Give answers on the basis of below context only.
If answer is not available in context, just say "Mujhe iska jawab document mein nahi mila."
Do not make up anything outside of the context (no hallucination).

Context:
{context}

Question: {question}

Answer:
""")

# Step E: Sirf tabhi query box dikhao jab kam se kam ek baar processing ho chuki ho
if st.session_state.get("processed"):
    query = st.text_input("Apna sawaal yahan likho:")

    if query:
        with st.spinner("🤔Sochte hue... (thoda time lag sakta hai)"):
            vector_store = st.session_state.vector_store
            results = vector_store.similarity_search(query, k=3)
            context_text = "\n\n".join([doc.page_content for doc in results])
            final_prompt = prompt.format(context=context_text, question=query)
            response = llm.invoke(final_prompt)

        st.subheader("Answer:")
        st.write(response.content)

        with st.expander("📌 Sources dekhein"):
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk {i+1}** — Page {doc.metadata.get('page_label')}, File: `{doc.metadata.get('source')}`")
                st.text(doc.page_content[:300] + "...")
else:
    st.info("👆 Pehle PDF(s) upload karke 'Process PDFs' button dabao.")