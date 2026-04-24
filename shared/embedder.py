from sentence_transformers import SentenceTransformer

# singleton pattern to load the model only once
_model = None

def _get_BGE_model_384():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed_with_BGE_384(nodes) -> list[float]:
    model = _get_BGE_model_384()
    return model.encode(nodes, normalize_embeddings=True).tolist()

