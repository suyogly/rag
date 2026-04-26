from email.mime import message

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

if os.path.exists(".env"):
    load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set")

def chat_groq(model: str = "openai/gpt-oss-120b", max_tokens: int = 1024, temperature: int = 0.7, reasoning_effort: str = "medium", stream: bool = True):
    '''
    model = "openai/gpt-oss-120b"
    model = "llama-3.1-8b-instant"
    model = "llama-3.3-70b-versatile"
    model = "openai/gpt-oss-20b"
    model = "groq/compound"
    '''

    llm = ChatGroq(
        model=model,
        verbose=True,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return llm

def build_context(results, k: int = 4):
    if not results:
        return ""

    chunks = [res.payload["text"] for res in results[:k]]
    return "\n\n".join(chunks)

def build_prompt(context: str, query: str):
    return f"""
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

def generate_answer(query: str, results):
    llm = chat_groq()

    context = build_context(results)
    prompt = build_prompt(context, query)

    response = llm.invoke(prompt)

    return response.content

if __name__ == "__main__":
    generate_answer()
