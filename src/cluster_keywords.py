import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load keywords
df = pd.read_csv('../data/keywords.csv')
keywords = df['Top queries'].tolist()

# Generate embeddings
embeddings = model.encode(keywords)

# Cluster using DBSCAN
clustering_model = DBSCAN(metric='cosine', eps=0.35, min_samples=2)
clusters = clustering_model.fit_predict(embeddings)

# Add clusters back to dataframe
df['cluster'] = clusters

# Save output
df.to_csv('../data/clustered_keywords.csv', index=False)

print("Clustering complete. Results saved to clustered_keywords.csv")