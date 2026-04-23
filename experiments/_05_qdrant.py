# from corpus.loader import load_document
from shared.embedder import BGE_384
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# _DOC = load_document()
# _our_DOC = _DOC[0].page_content

collection = "test"
client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists("test"):
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=4, distance=Distance.DOT)
    )

from qdrant_client.models import PointStruct

operation_info = client.upsert(
    collection_name=collection,
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
)

search_res = client.query_points(
    collection_name=collection,
    query=[0.2, 0.1, 0.9, 0.7],
    with_payload=True,
    limit=3
)

print(search_res)
