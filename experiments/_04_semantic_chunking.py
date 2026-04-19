from corpus.loader import load_document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter, # yet to know this shit
    SemanticSplitterNodeParser
)

_EMBEDDING_MODEL = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

langchain_doc = load_document()

# changing Langchain Document to Llamaindex supported Document
docs = [
    Document(
        text = doc.page_content,
        metadata = doc.metadata
    ) 
    for doc in langchain_doc
]

splitter = SemanticSplitterNodeParser(
    buffer_size=1, 
    breakpoint_percentile_threshold=95, 
    embed_model=_EMBEDDING_MODEL
)

nodes = splitter.get_nodes_from_documents(docs)

for idx, node in enumerate(nodes):
    print(f"id: {idx} \n{node.text}")
    print("\n---\n")


