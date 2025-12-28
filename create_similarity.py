import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your movies CSV
df = pd.read_csv("data/movies_with_cached_posters.csv")

# Combine relevant text features
features = (
    df["genres"].fillna("") + " " +
    df["keywords"].fillna("") + " " +
    df["tagline"].fillna("") + " " +
    df["cast"].fillna("") + " " +
    df["director"].fillna("")
)

# Compute TF-IDF vectors and similarity matrix
vectors = TfidfVectorizer().fit_transform(features)
sim_matrix = cosine_similarity(vectors)

# Save similarity matrix as .npy
np.save("data/similarity.npy", sim_matrix)

print("✅ similarity.npy created successfully!")
