from corpus.loader import load_document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    LangchainNodeParser,
)
from shared.qdrant_client import create_qdrant_collection, ingest_vectors, dense_retrieval


lc_doc = load_document()

llama_docs = [
    Document(
        text=doc.page_content, 
        metadata=doc.metadata
    ) for doc in lc_doc
]

lc_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size = 500,
    chunk_overlap = 250
)

node_parser = LangchainNodeParser(lc_splitter=lc_splitter)
nodes = node_parser.get_nodes_from_documents(
    documents=llama_docs
)

# for idx, node in enumerate(nodes):
#     print(f"id: {idx} \n{node.text}")
#     print("\n---\n")

_COLLECTION_NAME = create_qdrant_collection(collection_name="naive", dim=384)
print(_COLLECTION_NAME)

ingest = ingest_vectors(collection_name="naive", nodes=nodes)
print(ingest)

query = "what is your name?"

res = dense_retrieval(query=query, collection_name="naive")
print(res)