import pandas as pd
import numpy as np
import re
import pickle
import torch

# Read in dataset
df_filepath = '../Data/Toydata_w_new_att_final.xlsx'
df = pd.read_excel(df_filepath)
df.rename(columns={'Unnamed: 0': 'review_id'}, inplace=True)
df.columns = [col.lower() for col in df.columns]

# Clean up
df.price_in_usd.fillna(df.price_in_usd.mean(),inplace = True)
df.preferred_age.fillna(df.preferred_age.median(),inplace = True)
df["manufacturer_clean"] = df.manufacturer.str.strip()
df["manufacturer_clean"] = df["manufacturer"].apply(lambda x: str(x).lower())
df["reviews_clean"] = df.reviews.str.strip()
df["reviews_clean"] = df["reviews_clean"].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',str(x)))
df["reviews_clean"] = df["reviews_clean"].apply(lambda x: x.lower())

# Create corpus
corpus = [str(d) for d in df['reviews_clean']]

# Convert corpus into embeddings for semantic search
def create_embeddings(corpus):
  embedder = SentenceTransformer('all-mpnet-base-v2')

  if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")
  else:
    print("GPU Found!")
    embedder =  embedder.to('cuda')

  corpus_embedding = embedder.encode(corpus, show_progress_bar=True)
  return corpus_embedding

corpus_embeddings = create_embeddings(corpus)
embedding_filepath = '../Model/corpus_embeddings.pkl'
with open(embedding_filepath, 'wb') as file:
  pickle.dump(corpus_embeddings, file)


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

all_stopwords = list(stopwords.words('english'))
ps = PorterStemmer()
wnl = WordNetLemmatizer()

# Tokenize corpus for keyword search
def preprocess(corpus):
  corpus_processed = list()

  for text in corpus:
    # Tokenization (without special characters)
    text_tokens = word_tokenize(text)

    # Removing stop words
    text_tokens_wo_sw = [token for token in text_tokens if not token in all_stopwords]

    # Stemming
    text_tokens_wo_sw_stem = [ps.stem(token) for token in text_tokens_wo_sw]

    # Lemmatizing
    text_tokens_wo_sw_stem_lem = [wnl.lemmatize(token) for token in text_tokens_wo_sw_stem]

    text_processed = " ".join(text_tokens_wo_sw_stem_lem)

    corpus_processed.append(text_processed)
  return corpus_processed

corpus_processed = preprocess(corpus)
processed_corpus_filepath = '../Model/corpus_processed.pkl'
with open(processed_corpus_filepath, 'rb') as file:
  corpus_processed = pickle.load(file)