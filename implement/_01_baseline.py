from corpus.loader import load_document
from shared.chunker import fixed_splitter
from shared.embedder import embed_with_BGE_384
from shared.llm_client import chat_groq