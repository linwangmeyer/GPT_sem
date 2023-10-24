import sys
import csv
import pandas as pd
import numpy as np
import os
import openai
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import pandas as pd
from collections import Counter
import re


#--------------------- get words and features ----------------
with open("words_sem_dict.json", 'r') as json_file:
    word_sem = json.load(json_file)

df = pd.DataFrame(word_sem.items(), columns=['Word', 'Meanings'])
df2 = df.explode('Meanings')

word_pattern = r'\b[a-zA-Z]+\b'
valid_word_mask = ~df2['Meanings'].str.contains(word_pattern, case=False, na=False) | df2['Meanings'].str.contains('<br>', regex=False)
df3 = df2[~valid_word_mask]
df3.reset_index(drop=True, inplace=True)


#--------------------- ''manually' cluster based on cosine similarity' ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df3['Meanings']) #get phrase/sentence embedding vectors

# calculate similarity values
similarity_matrix = cosine_similarity(embeddings)

# 'cluster'/collapse elements with similarity values > 0.7
row_indices, col_indices = np.where(np.triu(similarity_matrix, k=1) > 0.7)
index_pairs = list(zip(row_indices, col_indices))

grouped_indices = {}
for row, col in index_pairs:
    if row in grouped_indices:
        grouped_indices[row].append(col)
    else:
        grouped_indices[row] = [col]

# create new features from the similar features by concatenating them
df3['newMeanings'] = df3['Meanings']
for row, cols in grouped_indices.items():
    combined_phrase = ' '.join(df3.loc[df3.index.isin(cols)]['Meanings'])
    df3.loc[df3.index == row, 'newMeanings'] = combined_phrase

df3[df3['Word']=='lime']
df3[df3['Word']=='sour']


#--------------------- DBSCAN ----------------
similarity_matrix = cosine_similarity(embeddings)
dissimilarity_matrix = 1 - similarity_matrix
min_value = np.min(dissimilarity_matrix)
max_value = np.max(dissimilarity_matrix)
scaled_matrix = (dissimilarity_matrix - min_value) / (max_value - min_value)

eps = 0.2  # Similarity threshold
dbscan = DBSCAN(eps=eps, metric="precomputed")
labels = dbscan.fit_predict(scaled_matrix)
len(set(labels))
df3['labels'] = labels

df3['newMeaning'] = df3.groupby('labels')['Meanings'].transform(lambda x: ' '.join(x))
df4 = df3.drop_duplicates(subset=['Word'])
df4.reset_index(inplace=True)


#--------------------- BERTopic ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df3['Meanings'])

umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=1000, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

topic_model = BERTopic(
  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  #representation_model=representation_model,
  top_n_words=4,
  verbose=True
)
topics, probs = topic_model.fit_transform(df3['Meanings'], embeddings)
topic_model.get_topic_info()

