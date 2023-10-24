import sys
import csv
import pandas as pd
import numpy as np
import os
import openai
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#----------- Gegerate features ---------
key_fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/lab_api_key.txt'
with open(key_fname,'r') as file:
    key = file.read()
openai.api_key = key #get it from your openai account

def generate_features(word):
    prompt = f"""
        We are conducting an investigation to understand how people interpret words for their meaning. In order to assist us in this research, we require information regarding the knowledge individuals possess about various concepts in the world. You will be provided with a list of words, each representing a specific concept. Your task is to come up with 20 properties related to the concept represented by the word. These properties can encompass a range of aspects, including physical characteristics such as internal and external features, sensory attributes (how it looks, sounds, smells, feels, or tastes), functional traits (its purpose, usage, when and where it is used, and by whom), associations or categories it belongs to, behavioral traits, and its origin. Please try to make the answer concise. Also, please start with more general characteristics and progress to more specific details.
        Please note when a noun can be interpreted as a proper name (e.g. 'apple'), it should be interpreted as noun refering to an object (e.g. 'apple' should be interpreted as a fruit).
        Here are a few examples of the words and their possible properties:
        Duck
        Is a bird
        Is an animal
        Waddles
        Flies
        Migrates
        Lays eggs
        Has webbed feet
        Has feathers
        Lives in ponds
        Lives in water
        Hunted by people
        Is edible

        Cucumber
        Is a vegetable
        Has green skin
        Has a white inside
        Has seeds inside
        Is long
        Is cylindrical
        Grows on vines
        Is edible
        Is crunchy
        Used for making pickles
        Eaten in salads

        Stove
        Is an appliance
        Produces heat
        Has elements
        Made of metal
        Is hot
        Is electrical
        Runs on wood
        Runs on gas
        Found in kitchens
        Used for baking
        Used for cooking food

        While you may think of additional or different types of properties for these concepts, these examples serve as a guide for what we are seeking.
        Could you generate at least 12 features of the word {word}?
        Make sure you only return the generated properties and say nothing else.
        """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,  # Adjust max tokens based on your requirements
        n = 1,  # Ensure only one response is generated
    )
    generated_text = response.choices[0].text
    split_text = generated_text.split('\n')
    cleaned_text = [line.strip() for line in split_text if line.strip()]

    return cleaned_text


#--------------------- get words ----------------
words_fname = r'/Users/linwang/Downloads/GPT_sem/1579words_words.txt'
with open(words_fname,'r') as file:
    words = file.read().split()

word_sem = dict()
for word in words:
    cleaned_text = generate_features(word)
    word_sem[word] = cleaned_text

# for the first word
elements = word_sem['trod'][2].split("[")[1].split("]")[0].split(",")
elements = [element.strip('"') for element in elements]
elements = [element.strip() for element in elements]
word_sem['trod'] = elements

with open("words_sem_dict.json", "w") as json_file:
    json.dump(word_sem, json_file)


#--------------------- get words and features ----------------
with open("words_sem_dict.json", 'r') as json_file:
    word_sem = json.load(json_file)

df = pd.DataFrame(word_sem.items(), columns=['Word', 'Meanings'])
df2 = df.explode('Meanings')
word_pattern = r'\b[a-zA-Z]+\b'
valid_word_mask = ~df2['Meanings'].str.contains(word_pattern, case=False, na=False) | df2['Meanings'].str.contains('<br>', regex=False)
df3 = df2[~valid_word_mask]
df3.reset_index(drop=True, inplace=True)


#-------------------Use K-means to cluster features ------------------
# get embeddings of all features
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df3['Meanings'])

# K-means: keep 2000 dimensions
n_clusters = 2000
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_
df3['labels'] = cluster_labels
df3['newMeaning'] = df3.groupby('labels')['Meanings'].transform(lambda x: ' '.join(x))


#-------get the feature summary using TF-IDF-------
# remove stop words
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)
df3['cleaned_newMeanings'] = df3['newMeaning'].apply(preprocess_text)

# use TF-IDF to get the representative words
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df3['cleaned_newMeanings'])
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names)
representative_words_per_row = []
for index, row in tfidf_scores_df.iterrows():
    # Sort words by TF-IDF score in descending order and get the top N words
    top_words = [feature_names[i] for i in row.argsort()[::-1][:5]]
    representative_words_per_row.append(top_words)

concatenated_features = ['_'.join(words) for words in representative_words_per_row]
df3['features'] = concatenated_features
df3.drop(columns=['cleaned_newMeanings'], inplace=True)
df3.to_csv('Kmeans_' + str(n_clusters) + '.csv', index = False)


#---------- get binary word-feature matrix ---------
df3 = pd.read_csv('Kmeans_' + str(n_clusters) + '.csv')

# Convert to word-feature matrix
unique_words = list(set(df3['Word'].unique()))
unique_top_words = list(df3['features'].unique())
matrix = pd.DataFrame(0, index=unique_words, columns=unique_top_words)
word_column = df3['Word']
top_words_column = df3['features']

for i, word in enumerate(word_column):
    top_word = top_words_column[i]
    matrix.loc[word, top_word] = 1
matrix.reset_index(inplace=True)
matrix.rename(columns={'index':'word'}, inplace=True)
matrix.to_csv('binary_' + str(n_clusters) + '.csv', index=False)

#---------- check results: compare to word2vec representations ---------
binary_matrix = pd.read_csv('binary_' + str(n_clusters) + '.csv')
vec = binary_matrix.iloc[:,1:]
sim_mx = cosine_similarity(vec)
sns.heatmap(sim_mx, cmap='coolwarm', annot=False)

# get word2vec similarity
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
binary_matrix['word'] = binary_matrix['word'].replace('grey', 'gray')
words = binary_matrix['Word']
num_words = len(words)
sim_w2v = np.zeros((num_words, num_words))
for i in range(num_words):
    for j in range(num_words):
        sim_w2v[i][j] = model.similarity(words[i], words[j])
sns.heatmap(sim_w2v, cmap='coolwarm', annot=False)

# correlate results between kmeans and word2vec results
indices = np.triu_indices(sim_mx.shape[0],k=1)
kmn = sim_mx[indices]
w2v = sim_w2v[indices]
corr, p_value = pearsonr(kmn,w2v)
print("correlation coefficient:", corr)
print("P-value:", p_value)

# check similar pairs
row_indices, col_indices = np.where(np.triu(sim_mx, k=1) > 0.7)
index_pairs = list(zip(row_indices, col_indices))
for a,b in index_pairs[12:20]:
    print(binary_matrix.iloc[a]['Word'])
    print(binary_matrix.iloc[b]['Word'])


