import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
import stopwordsiso as stopwords
import yaml

def read_stopwords_from_yaml(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    # Relabel the keys to match the desired codes
    relabeled_stopwords = {
        'am': data['Amharic'] if 'Amharic' in data else [],
        'ny': data['Chichewa'] if 'Chichewa' in data else [],
        'sw': list(stopwords.stopwords('sw'))  # Add Swahili stopwords directly from the library
    }
    
    return relabeled_stopwords

def train_lda(data_path, model_save_path, language, num_topics, sw):
    '''
    train_lda creates a latent dirichlet allocation 

    input: 
        data_path -the source of the training data
        model_save_path - where to save the pkl files of the trained models
        language - the language selected 
        num_topics - the number of labes in the training dataset
    output:
        None
    '''
    # Load data
    df = pd.read_csv(data_path)
    df['Text'] = df['Text'].fillna("")  # Replace NaNs with empty strings
    texts = df['Text']  # Assuming 'Text' column contains the data

    # Preprocessing: Convert text to a document-term matrix

    # custom stopwords from https://github.com/stopwords-iso/stopwords-iso
    

    vectorizer = CountVectorizer(stop_words=sw, max_features=1000)
    dt_matrix = vectorizer.fit_transform(texts)

    # Train LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dt_matrix)

    # Save the model and vectorizer
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({"model": lda_model, "vectorizer": vectorizer}, model_save_path)
    print(f"LDA Model saved to {model_save_path}")

def train_nmf(data_path, model_save_path, language, num_topics, sw):
    '''
    train_nmf creates a Non-Negative Matrix Factorization 

    input: 
        data_path -the source of the training data
        model_save_path - where to save the pkl files of the trained models
        language - the language selected 
        num_topics - the number of labes in the training dataset
    output:
        None
    '''
    # Load data
    df = pd.read_csv(data_path)
    df['Text'] = df['Text'].fillna("")  # Replace NaNs with empty strings
    texts = df['Text']

    # Determine language-specific stopwords

    # Preprocessing: Convert text to a TF-IDF weighted document-term matrix
    vectorizer = TfidfVectorizer(stop_words=sw, max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Train NMF model
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)

    # Save the model and vectorizer
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({"model": nmf_model, "vectorizer": vectorizer}, model_save_path)
    print(f"NMF Model saved to {model_save_path}")

def job():
    print("Current Working Directory:", os.getcwd())
    stopword_dict = read_stopwords_from_yaml("./data/raw/stopwords.yaml")
    
    # Configuration dictionary
    languages = {
        'Chichewa': {
            'data_path': "./data/processed/Chichewa/chichewa_clean_train.csv",
            'model_paths': {
                'lda': "./data/model/lda_chichewa.pkl",
                'nmf': "./data/model/nmf_chichewa.pkl"
            },
            'stopword_code': 'ny',
            'num_topics': 5
        },
        'Amharic': {
            'data_path': "./data/processed/Amharic/amharic_clean_train.csv",
            'model_paths': {
                'lda': "./data/model/lda_amharic.pkl",
                'nmf': "./data/model/nmf_amharic.pkl"
            },
            'stopword_code': 'am',
            'num_topics': 4
        },
        'Swahili': {
            'data_path': "./data/processed/Swahili/swahili_clean_train.csv",
            'model_paths': {
                'lda': "./data/model/lda_swahili.pkl",
                'nmf': "./data/model/nmf_swahili.pkl"
            },
            'stopword_code': 'sw',
            'num_topics': 6
        }
    }

    # Loop through each language and train both models
    for language, config in languages.items():
        print(f"Training models for {language}")
        data_path = config['data_path']
        lda_model_path = config['model_paths']['lda']
        nmf_model_path = config['model_paths']['nmf']
        stopword_code = config['stopword_code']
        num_topics = config['num_topics']
        
        language_stopwords = stopword_dict[stopword_code]
        # Train LDA model
        train_lda(data_path, lda_model_path, stopword_code, num_topics, language_stopwords)

        # Train NMF model
        train_nmf(data_path, nmf_model_path, stopword_code, num_topics, language_stopwords)

if __name__ == "__main__":
    job()
