import streamlit as st
import wikipedia
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load embedding model (on GPU if available)
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load LLM (use Mistral or TinyLlama)
@st.cache_resource
def get_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# Fetch and split Wikipedia text
def get_wiki_chunks(topic):
    try:
        page = wikipedia.page(topic)
        raw_text = page.content
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(raw_text)
    except:
        return []

# Streamlit App
st.title("🌍 General Knowledge Chatbot")
topic = st.text_input("Enter a topic :")

if topic:
    with st.spinner("Fetching and indexing..."):
        chunks = get_wiki_chunks(topic)
        if not chunks:
            st.error("Topic not found or no content.")
        else:
            embedder = get_embedder()
            vectorstore = FAISS.from_texts(chunks, embedder)
            retriever = vectorstore.as_retriever()

            llm = get_llm()
            rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

            question = st.text_input("Ask a question about this topic:")
            if question:
                answer = rag_chain.run(question)
                st.write("**🤖 Answer:**", answer)

if st.button("⬅ back to main page"):
    # Your redirect or logic here
    st.switch_page("main.py")
