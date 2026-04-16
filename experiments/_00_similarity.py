from corpus.data import _SENTENCES
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import matplotlib.pyplot as plt


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = model.encode(_SENTENCES, normalize_embeddings=True)
n = len(embeddings)

# UMAP reduction (high-dim -> 2D)
reducer = umap.UMAP(n_neighbors=2, min_dist=0.3, metric="cosine")
reduced = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
for i, txt in enumerate(_SENTENCES):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, txt, fontsize=9)

plt.title("UMAP Projection of Sentence Similarity")
plt.show()

# results = []
# for i in range(n):
#     for j in range(i+1, n):
#         similarity = np.dot(embeddings[i], embeddings[j])

#         results.append({
#             "indices": (i, j),
#             "result": similarity
#         })

# results.sort(key=lambda x: x["result"], reverse=True)
# print(results)

    