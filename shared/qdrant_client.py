from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from shared.embedder import embed_with_BGE_384
from uuid import uuid4
import hashlib

client = QdrantClient(
    host="localhost",
    port=6333
)

def create_qdrant_collection(collection_name, dim: int):
    if client.collection_exists(collection_name):
        print(f"Collection already exists with {collection_name} and dimension of {dim}")
        return client

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        )
    )
    print(f"Qdrant Collection created with {collection_name} and dimension {dim}")
    return client, client.get_collection


def ingest_vectors(collection_name: str, nodes):
    if not nodes:
        return {"error": "there are no nodes"}
    
    text = [node.text for node in nodes]
    embeddings = embed_with_BGE_384(nodes=text)

    points = []
    for i, node in enumerate(nodes):
        points.append(
            PointStruct(
                id=hashlib.md5(node.text.encode("utf-8")).hexdigest(), # using hash because if i reembed and upsert vectors, it might duplicate the vectors
                vector=embeddings[i],
                payload={
                    "text": node.text,
                    "metadata": node.metadata,
                    # "dataset": "my_dataset"
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return len(points)

def dense_retrieval(query: str, collection_name, top_k: int = 4):
    query_vector = embed_with_BGE_384(query)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        with_payload=True,
        limit=top_k
    ).points

    return results
