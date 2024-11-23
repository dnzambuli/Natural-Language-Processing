import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

def train_lda(data_path, model_save_path, num_topics=5):
    # Load data
    df = pd.read_csv(data_path)
    df['Text'] = df['Text'].fillna("")  # Replace NaNs with empty strings
    texts = df['Text']  # Assuming 'Text' column contains the data

    # Preprocessing: Convert text to a document-term matrix
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    dt_matrix = vectorizer.fit_transform(texts)

    # Train LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dt_matrix)

    # Save the model and vectorizer
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({"model": lda_model, "vectorizer": vectorizer}, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Paths for training data
    chichewa_path = "../data/processed/Chichewa/chichewa_clean_train.csv"
    amharic_path = "../data/processed/Amharic/amharic_clean_train.csv"
    swahili_path = "../data/processed/Swahili/swahili_clean_train.csv"

    # Paths to save the models
    chichewa_model_path = "../data/model/lda_chichewa.pkl"
    amharic_model_path = "../data/model/lda_amharic.pkl"
    swahili_model_path = "../data/model/lda_swahili.pkl"

    # Train models
    train_lda(chichewa_path, chichewa_model_path, num_topics=5) # has 20 topics
    train_lda(amharic_path, amharic_model_path, num_topics=4) # amharic has 4 topics
    train_lda(swahili_path, swahili_model_path, num_topics = 6) # swahili has 6 topics
