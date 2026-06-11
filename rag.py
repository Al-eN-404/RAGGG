from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from retriever import retrieve

import os

load_dotenv()

def get_api_key():
    # Try multiple standard keys from streamlit secrets or env variables
    keys = ["API-KEY", "GROQ_API_KEY", "API_KEY", "OPENAI_API_KEY"]
    
    # 1. Try Streamlit Secrets (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        for key in keys:
            if key in st.secrets:
                return st.secrets[key]
    except Exception:
        pass
        
    # 2. Try environment variables (for local deployment / env files)
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
            
    return None

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=get_api_key(),
    base_url="https://api.groq.com/openai/v1"
)

def ask_rag(question, domain=None):
    """
    Query the vector store with hybrid search and answer using Groq with streaming response.
    Supports domain filtering to isolate source context.
    """
    retrieved_chunks = retrieve(question, domain=domain)

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )
    
    sources = list({
        chunk["url"]
        for chunk in retrieved_chunks
    })

    system_prompt = f"""You are an intelligent AI assistant that answers user questions using the provided webpage context.

Instructions:
1. Answer the user's question directly and naturally.
2. NEVER use phrases like:
   - "Based on the retrieved content"
   - "According to the documents"
   - "The scraped content says"
   - "Document 1"
   - "Document 2"
   - "It appears that"
   - "The context mentions"
3. Do NOT explain how retrieval works.
4. Give clean, human-like answers as if you already know the information.
5. Use ONLY the provided context below as your knowledge source.
6. If the answer is not available in the context, respond ONLY with:
   "I could not find information related to this question on this current website."
7. Keep answers concise, professional, clear, and natural.
8. Preserve technical accuracy for APIs, versions, function names, and identifiers.
9. If multiple context chunks contain useful information, combine them naturally into one answer.
10. Never mention chunks, retrieval, embeddings, vector databases, metadata, scores, or the scraping process.

Context:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Stream the response from the LLM
    stream = llm.stream(messages)
    answer_stream = (chunk.content for chunk in stream if chunk.content)

    return {
        "answer_stream": answer_stream,
        "sources": sources
    }
