import sentence_transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import os
import pandas as pd
import re
import spacy
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json

all_stopwords = list(stopwords.words('english'))
ps = PorterStemmer()
wnl = WordNetLemmatizer()



def load_models():
    return SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True), CrossEncoder('cross-encoder/msmarco-MiniLM-L12-en-de-v1', trust_remote_code=True)

def load_inputs(df_filepath, embedding_filepath, processed_corpus_filepath):
    # Read in dataset
    df = pd.read_excel(df_filepath)
    df.rename(columns={'Unnamed: 0': 'review_id'}, inplace=True)
    df.columns = [col.lower() for col in df.columns]

    # Clean up
    df.price_in_usd.fillna(df.price_in_usd.mean(), inplace=True)
    df.preferred_age.fillna(df.preferred_age.median(), inplace=True)
    df["manufacturer_clean"] = df.manufacturer.str.strip()
    df["manufacturer_clean"] = df["manufacturer"].apply(
        lambda x: str(x).lower())
    df["reviews_clean"] = df.reviews.str.strip()
    df["reviews_clean"] = df["reviews_clean"].apply(
        lambda x: re.sub('[^a-zA-z0-9\s]', '', str(x)))
    df["reviews_clean"] = df["reviews_clean"].apply(lambda x: x.lower())

    # Create corpus
    corpus = [str(d) for d in df['reviews_clean']]

    # Load corpus embeddings
    with open(embedding_filepath, 'rb') as file:
        corpus_embeddings = pickle.load(file)
    corpus_embeddings_dict = {}
    i = 0
    for embedding in corpus_embeddings:
        corpus_embeddings_dict[i] = embedding
        i += 1
    assert len(corpus_embeddings_dict.keys()) == len(corpus)

    # Load processed corpus
    with open(processed_corpus_filepath, 'rb') as file:
        corpus_processed = pickle.load(file)

    return df, corpus, corpus_embeddings_dict, corpus_processed


df, corpus, corpus_embeddings_dict, corpus_processed = load_inputs(
    df_filepath='../Data/Toydata_w_new_att_final.xlsx', embedding_filepath='../Model/corpus_embeddings.pkl', processed_corpus_filepath='../Model/corpus_processed.pkl')

print("----- Complete loading inputs -----")


def search_reviews(query, corpus_embeddings_dict, embedder, top_n=5):
    # Do bi-encoder search
    query_embedding = embedder.encode(query, show_progress_bar=True)
    # print(query_embedding.shape)
    biencoder_scores = {}
    for i in corpus_embeddings_dict.keys():
        score_tensor = sentence_transformers.util.cos_sim(corpus_embeddings_dict[i], query_embedding)
        score = score_tensor[0].numpy()[0]
        biencoder_scores[i] = score

    hits = dict(sorted(biencoder_scores.items(),
                key=lambda x: x[1], reverse=True)[:top_n])
    return hits

def rerank_reviews(query, hits, cross_encoder, top_n=None):

    def get_result(idx):
        result = dict(df.iloc[idx])
        return "Product: {product_name}\nManufacturer: {manufacturer}\nPrice: {price_in_usd}\nPreferred Age: {preferred_age}\nRating: {star_rating}\nReviews: {reviews}".format(**result)

    crossencoder_inputs = [[query, get_result(idx)] for idx, score in hits.items()]
    crossencoder_scores = cross_encoder.predict(crossencoder_inputs)
    assert len(hits.keys()) == len(crossencoder_scores)
    i = 0
    for idx, score in hits.items():
        hits[idx] = 1/(1 + np.exp(-crossencoder_scores[i]))
        i += 1
    hits = dict(sorted(hits.items(), key=lambda x: x[1], reverse=True))
    if top_n is not None:
        hits = dict(list(hits.items())[:top_n])
    return hits

def preprocess(corpus):
    corpus_processed = list()
    for text in corpus:
        # Tokenization (already without special characters)
        text_tokens = word_tokenize(text)

        # Removing stop words
        text_tokens_wo_sw = [
            token for token in text_tokens if not token in all_stopwords]

        # Stemming
        text_tokens_wo_sw_stem = [ps.stem(token)
                                  for token in text_tokens_wo_sw]

        # Lemmatizing
        text_tokens_wo_sw_stem_lem = [wnl.lemmatize(
            token) for token in text_tokens_wo_sw_stem]

        text_processed = " ".join(text_tokens_wo_sw_stem_lem)
        corpus_processed.append(text_processed)

    return corpus_processed

def bm25_search_reviews(query, corpus_processed, top_n=5):
    # Do keyword search
    # Vectorize tokenized corpus
    corpus_tokenized = [text.split(" ") for text in corpus_processed]
    bm25 = BM25Okapi(corpus_tokenized)
    # Vectorize query
    query_split = query.split(" ")
    query_tokenized = [t for t in preprocess(query_split) if t != str('')]
    # Compute scores
    bm25_scores = bm25.get_scores(query_tokenized)
    top_scores_idx = np.argpartition(bm25_scores, -top_n)[-top_n:]
    hits = {}
    for idx in top_scores_idx:
        hits[idx] = bm25_scores[idx]
    hits = dict(sorted(hits.items(), key=lambda x: x[1], reverse=True))
    return hits


def fill_defaults(data, defaults):
    for key, value in defaults.items():
        if isinstance(value, dict):
            data[key] = fill_defaults(data.get(key, {}), value)
        else:
            data[key] = data.get(key, value) if data.get(key, value) else value
    return data


def gpt_decompose(query, llm_client):
    # Use GPT to decompose query
    default_dict = {'keywords': {'brand': '',
                    'price (lower bound)': 0,
                                 'price (upper bound)': 100000,
                                 'age (lower bound)': 0,
                                 'age (upper bound)': 100},
                    'subjective': ''}
    default_json = json.dumps(default_dict, indent=2)

    # Prompt generation
    prompt = f"Rewrite the following query:\n\n\"{query}\"\n\nstrictly into this {default_json}. Return a JSON parseable string within ```json and ```."

    response = llm_client.chat.completions.create(
        model="gpt-4o-large",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}], 
            temperature=0.5, 
            max_tokens=300
        )

    # print("GPT response: ", response)
    # Text output
    generated_output = response.choices[0].message.content
    # print("GPT response data type: ", type(generated_output))
    # print("GPT response text: ", generated_output)
    # Extract JSON string between triple backticks
    json_str = generated_output.strip().split("```json\n")[1].split("\n```")[0]
    # Parse JSON string into Python dictionary
    query_dict = json.loads(json_str)
    assert type(query_dict) == dict
    # query_dict = eval(generated_output)  # Dict output

    query_dict_with_defaults = fill_defaults(query_dict, default_dict)
    return query_dict_with_defaults


BRANDS = df['manufacturer'].unique().tolist()
print(BRANDS)
BRANDS = list(map(lambda x: x.lower() if type(x) == str else x, BRANDS))

def filter_data(df, query_dict, complacency=1.2):
    brand = query_dict['keywords']['brand']

    if brand == '':  # user query does not contain a brand
        df_filtered = df.copy()
    # user query has brand
    elif brand.lower() not in BRANDS:
        df_filtered = df.copy()
        print("Sorry, the brand you are looking for is not available!")
    else:
        df_filtered = df[df['manufacturer_clean'] == brand.lower()]

    df_filtered = df_filtered.loc[(df_filtered['price_in_usd'] >= query_dict['keywords']['price (lower bound)'] / complacency)
                                  & (df_filtered['price_in_usd'] <= query_dict['keywords']['price (upper bound)'] * complacency)]
    df_filtered = df_filtered.loc[(df_filtered['preferred_age'] >= query_dict['keywords']['age (lower bound)'])
                                  & (df_filtered['preferred_age'] <= query_dict['keywords']['age (upper bound)'])]

    matched = True
    if df_filtered.shape[0] == 0:  # in case there is no rows in filtered df
        print("Sorry, no relevant results!")
        doc_indices = df.review_id.tolist()
        matched = False
    else:
        doc_indices = df_filtered.review_id.tolist()

    return doc_indices, matched


def cross_encoder_search(query, corpus_embeddings_dict, embedder, cross_encoder, top_k=25, top_n=10):
    biencoder_hits = search_reviews(query, corpus_embeddings_dict, embedder, top_k)
    crossencoder_hits = rerank_reviews(query, biencoder_hits, cross_encoder, top_n=top_n)
    return crossencoder_hits

def hybrid_search(query, df, corpus_embeddings_dict, llm_client, embedder, cross_encoder, top_k=25, top_n=10):
    query_dict = gpt_decompose(query, llm_client)
    # query_dict = {'keywords': {'brand': 'LEGO',
    #               'price (lower bound)': 50,
    #               'price (upper bound)': 100,
    #               'age (lower bound)': 0,
    #               'age (upper bound)': 10},
    #               'subjective': 'for children'}
    # print(type(query_dict))
    # print(query_dict)

    doc_indices, matched = filter_data(df, query_dict)
    # print("Length of filtered reviews: ", len(doc_indices))

    df_filtered = df.loc[(df.review_id.isin(doc_indices))]
    # print('Brands:', df_filtered.manufacturer.unique())
    # print('Min Price:', df_filtered.price_in_usd.min())
    # print('Max Price:', df_filtered.price_in_usd.max())
    # print('Min Age:', df_filtered.preferred_age.min())
    # print('Max Age:', df_filtered.preferred_age.max())

    query_subj = query_dict['subjective']
    if query_subj == "":
        query_subj = query
    # print("Subjective parts from query --> ", query_subj)

    corpus_embeddings_filtered = {}
    for idx in doc_indices:
        corpus_embeddings_filtered[idx] = corpus_embeddings_dict[idx]
    assert len(corpus_embeddings_filtered.keys()) == len(doc_indices)

    hybrid_bi_hits = search_reviews(
        query_subj, corpus_embeddings_filtered, embedder, top_n=top_k)
    hybrid_cross_hits = rerank_reviews(
        query_subj, hybrid_bi_hits, cross_encoder, top_n=top_n)
    return hybrid_cross_hits, matched


########## Display Results ##########
def display_results(hits, top_n=5):
    result_dict = {}
    for idx in hits.keys():  # loop through hits
        product = df['product_name'].loc[(df['review_id'] == idx)].tolist()[0]
        review = df['reviews'].loc[(df['review_id'] == idx)].tolist()[0]
        if product not in result_dict.keys():
            if len(result_dict.keys()) < top_n:
                product_df = df.loc[(df['product_name'] == product)]

                result_dict[product] = {
                    'manufacturer': product_df['manufacturer'].unique()[0],
                    'price': round(product_df['price_in_usd'].mean(), 2),
                    'minimum_age': round(product_df['preferred_age'].min(), 0),
                    'rating': round(product_df['star_rating'].max(), 1),
                    'reviews': [review]
                }
        else:
            result_dict[product]['reviews'].append(review)

    return result_dict