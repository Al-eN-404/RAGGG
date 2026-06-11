from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from retriever import retrieve

import os
# import content

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("API-KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def ask_rag(question):

    retrieved_chunks = retrieve(question)

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )
    
    sources=list({
        chunk["url"]
        for chunk in retrieved_chunks
    })

    prompt = f"""
    Answer ONLY using the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources
    }



# question = "What is transport?"

# while True:

#     question = input("\nAsk: ")

#     if question.lower() == "exit":
#         break

#     retrieved_chunks = retrieve(question)

#     context = "\n\n".join(
#         chunk["text"]
#         for chunk in retrieved_chunks
#     )

#     prompt = f"""
#     Answer ONLY using the provided context.

#     Context:
#     {context}

#     Question:
#     {question}
#     """

#     response = llm.invoke(prompt)

#     print("\nAnswer:")
#     print(response.content)
    