from sentence_transformers import SentenceTransformer

def BGE_384(data):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
    embeddings = model.encode(data, normalize_embeddings=True)
    return embeddings

