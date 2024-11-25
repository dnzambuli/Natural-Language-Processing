import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.decomposition import NMF, LatentDirichletAllocation

def predict_topic(model_path, text):
    # Load model and vectorizer
    model_data = joblib.load(model_path)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    
    # Transform the input text
    dt_matrix = vectorizer.transform([text])
    
    # Predict topic probabilities
    if isinstance(model, LatentDirichletAllocation):
        topic_probabilities = model.transform(dt_matrix)
    elif isinstance(model, NMF):
        topic_probabilities = model.transform(dt_matrix)
    else:
        raise ValueError("Model type not supported")
    
    # Return the most probable topic
    return topic_probabilities.argmax()

def test_model(test_data_path, model_path):
    # Load test data
    df = pd.read_csv(test_data_path)
    
    # Handle missing text
    if "Text" not in df.columns:
        raise ValueError("The input dataset does not have a 'Text' column.")
    df["Text"] = df["Text"].fillna("")
    
    # Ground truth labels
    if "Label" not in df.columns:
        raise ValueError("The input dataset does not have a 'Label' column.")
    true_labels = df["Label"]
    
    # Predict topics
    predicted_labels = df["Text"].apply(lambda text: predict_topic(model_path, text))
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

if __name__ == "__main__":

    # Paths
    processed_data_dir = "./data/processed"
    model_dir = "./data/model"
    accuracy_file_path = "./data/accuracy.txt"
    
    # Ensure required directories exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)


    # Model and test data paths
    test_data = {
        "chichewa_lda": {
            "model_path": os.path.join(model_dir, "lda_chichewa.pkl"),
            "test_path": os.path.join(processed_data_dir, "Chichewa", "chichewa_clean_test.csv"),
        },
        "chichewa_nmf": {
            "model_path": os.path.join(model_dir, "nmf_chichewa.pkl"),
            "test_path": os.path.join(processed_data_dir, "Chichewa", "chichewa_clean_test.csv"),
        },
        "swahili_lda": {
            "model_path": os.path.join(model_dir, "lda_swahili.pkl"),
            "test_path": os.path.join(processed_data_dir, "Swahili", "swahili_clean_test.csv"),
        },
        "swahili_nmf": {
            "model_path": os.path.join(model_dir, "nmf_swahili.pkl"),
            "test_path": os.path.join(processed_data_dir, "Swahili", "swahili_clean_test.csv"),
        },
        "amharic_lda": {
            "model_path": os.path.join(model_dir, "lda_amharic.pkl"),
            "test_path": os.path.join(processed_data_dir, "Amharic", "amharic_clean_test.csv"),
        },
        "amharic_nmf": {
            "model_path": os.path.join(model_dir, "nmf_amharic.pkl"),
            "test_path": os.path.join(processed_data_dir, "Amharic", "amharic_clean_test.csv"),
        }
    }
    
    # Save accuracies
    with open(accuracy_file_path, "w") as file:
        file.write("Model\taccuracy\n")
        for model_name, paths in test_data.items():
            try:
                print(f"Processing {model_name}...")
                acc = test_model(paths["test_path"], paths["model_path"])
                file.write(f"{model_name}\t{acc:.4f}\n")
                print(f"Accuracy for {model_name}: {acc:.4f}")
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
