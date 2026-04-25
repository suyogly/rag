from corpus.loader import load_document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    LangchainNodeParser
)
from llama_index.core import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

_EMBEDDING_MODEL = None

lc_doc = load_document()

llama_doc = [
    Document(
        text = doc.page_content,
        metadata = doc.metadata
    ) for doc in lc_doc
]

def _load_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
    return _EMBEDDING_MODEL

def recursive_splitter(chunk_size: int = 500, chunk_overlap: int = 250):
    lc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    node_parser = LangchainNodeParser(lc_splitter=lc_splitter)
    nodes = node_parser.get_nodes_from_documents(
        documents=llama_doc
    )
    return nodes

def fixed_splitter(chunk_size: int = 250, chunk_overlap: int = 0, sepr: str = "\n"):
    lc_splitter = CharacterTextSplitter(
            separator=sepr,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
    node_parser = LangchainNodeParser(lc_splitter=lc_splitter)
    nodes = node_parser.get_nodes_from_documents(
        documents=llama_doc
    )       
    return nodes

def semantic_splitter(buffer_size: int = 1, bep_thres: int = 95, embed_model: str | object = None):
    
    if embed_model is None:
        embed_model = _load_model()
        
    splitter = SemanticSplitterNodeParser(
    buffer_size=buffer_size, 
    breakpoint_percentile_threshold=bep_thres, 
    embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(llama_doc)
    return nodes
        