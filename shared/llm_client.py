from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

if os.path.exists(".env"):
    load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set")

def chat_groq():
    pass