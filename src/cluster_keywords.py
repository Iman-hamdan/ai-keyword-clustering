import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

# Load keywords
try:
    df = pd.read_csv(os.path.join(data_dir, 'keywords.csv'))
except FileNotFoundError:
    print(f"Error: Could not find the file at {os.path.join(data_dir, 'keywords.csv')}")
    exit(1)

keywords = df['Top queries'].tolist()

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(keywords)

# Cluster using DBSCAN
clustering_model = DBSCAN(metric='cosine', eps=0.35, min_samples=2)
clusters = clustering_model.fit_predict(embeddings)

# Add clusters back to dataframe
df['cluster'] = clusters

# Save output
output_path = os.path.join(data_dir, 'clustered_keywords.csv')
df.to_csv(output_path, index=False)

print(f"Clustering complete. Results saved to {output_path}")