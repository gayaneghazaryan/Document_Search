from datasets import load_dataset
import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity



def extract_wiki_data(dataset, rows):
    data = load_dataset("wikipedia", dataset)
    wiki_df = pd.DataFrame(data['train'])
    subset = wiki_df.iloc[:rows, :]
    subset.to_csv("wikipedia.csv", index=False)
    
    
def normalize_text(data, mode = 'docs'):
    if mode == 'docs':
        
        #lowercase transformation
        data = data.apply(lambda x: x.lower())
        data = data.apply(remove_punctuation)
        data = data.replace('\n\n', ' ', regex=True)
        data = data.replace('\n', ' ', regex=True)
        lemmatized_data = data.apply(tokenize_and_lemmatize_text)
        # lemmatized_data = lemmatized_data.apply(remove_stopwords)
    elif mode == 'query':
        data = data.lower()
        data = remove_punctuation(data)
        lemmatized_data = tokenize_and_lemmatize_text(data)
        # lemmatized_data = remove_stopwords(lemmatized_data)
    else:
        raise ValueError('mode must be either "docs" or "query" ')

    return lemmatized_data


def remove_punctuation(text):
    #removed - for hypenated words not to be separated
    exclude = set(string.punctuation) - set('-')
    #add some specific punctuations not present in the default
    exclude.add('“')
    exclude.add('”')
    exclude.add('–')
    exclude.add('!')
    res = ''.join(ch for ch in text if ch not in exclude)
    res = res.replace(' - ', ' ')
    return res


def tokenize_and_lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens, tagset = 'universal')
    lemmatized_tokens = []

    for token, pos in pos_tags:
        if pos[0].lower() in ['n', 'v', 'a']:
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos = pos[0].lower()))
        else:
            lemmatized_tokens.append(token)


    return lemmatized_tokens

def remove_stopwords(tokens):

    stop_words = set(stopwords.words('english'))
    
    return [token for token in tokens if token not in stop_words]

def calculate_word_frequency_rank(series):
    words = pd.Series([token for tokens in series for token in tokens])
    word_frequency = words.value_counts()
    word_frequency_rank = pd.DataFrame({'word': word_frequency.index, 'frequency': word_frequency.values})
    word_frequency_rank['rank'] = np.arange(1, len(word_frequency_rank) + 1)

    return word_frequency_rank

def plot_zipfs_law(word_frequency_rank):

    word_frequency = word_frequency_rank['frequency']
    rank = word_frequency_rank['rank']
    popt, _ = curve_fit(zipf_law, rank, word_frequency)
    k, alpha = popt
    plt.figure(figsize=(10, 6))
    plt.loglog(rank, word_frequency, 'b.', markersize=8)
    plt.loglog(rank, zipf_law(rank, k, alpha), 'r-', label=f'k={k:.2f}, alpha={alpha:.2f}') 
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Word Frequencies vs Rank (Log-Log Scale)')
    plt.legend()
    plt.show()
    plt.savefig('zipfs_law.png')


def zipf_law(x, k, alpha):
    return k * (x ** (-alpha))

def calculate_tfidf(data, query=None, mode="docs"):

    cv = CountVectorizer()

    joined_text = data.apply(' '.join)

    word_count_vectorizer = cv.fit_transform(joined_text).astype('int16')
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(word_count_vectorizer)


    if mode == 'docs':

        return tfidf_matrix

    elif mode == "query":

        query = ' '.join(query)

        query_tf_vector = cv.transform([query]).toarray()

        query_tfidf_vector = tfidf_transformer.transform(query_tf_vector)

        return query_tfidf_vector

    else:
        raise ValueError('mode must be either "docs" or "query" ')
        
        
        
def search_documents(data, tfidf_matrix, query_tfidf_vector, k):
    similarities = cosine_similarity(query_tfidf_vector.reshape(1,-1), tfidf_matrix)
    
    sorted_indices = similarities.argsort()[0][::-1][:k]
    
    top_k_indices = sorted_indices[:k].flatten()
    top_k_similarities = similarities[:,top_k_indices].flatten()
    top_k_documents = data.iloc[top_k_indices]
    
    return top_k_indices, top_k_similarities, top_k_documents


if __name__ == '__main__':
    query = 'English-German dictionary, book / journal' 

    query_norm = normalize_text(query, mode = 'query')
    print(query_norm)
  