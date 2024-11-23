import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def predict_topic(model_path, text):
    # Load model and vectorizer
    model_data = joblib.load(model_path)
    lda_model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    
    # Transform the input text
    dt_matrix = vectorizer.transform([text])
    
    # Predict topic probabilities
    topic_probabilities = lda_model.transform(dt_matrix)
    
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
    os.makedirs("../data", exist_ok=True)

    os.makedirs(os.path.join(processed_data_dir, "Chichewa"), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, "Swahili"), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, "Amharic"), exist_ok=True)

    # Model and test data paths
    test_data = {
        "chichewa_model": {
            "model_path": os.path.join(model_dir, "lda_chichewa.pkl"),
            "test_path": os.path.join(processed_data_dir, "Chichewa", "chichewa_clean_test.csv"),
        },
        "swahili_model": {
            "model_path": os.path.join(model_dir, "lda_swahili.pkl"),
            "test_path": os.path.join(processed_data_dir, "Swahili", "swahili_clean_test.csv"),
        },
        "amharic_model": {
            "model_path": os.path.join(model_dir, "lda_amharic.pkl"),
            "test_path": os.path.join(processed_data_dir, "Amharic", "amharic_clean_test.csv"),
        },
    }
    
    # Save accuracies
    with open(accuracy_file_path, "w") as file:
        file.write("model_name\taccuracy\n")
        for model_name, paths in test_data.items():
            try:
                print(f"Processing {model_name}...")
                acc = test_model(paths["test_path"], paths["model_path"])
                file.write(f"{model_name}\t{acc:.4f}\n")
                print(f"Accuracy for {model_name}: {acc:.4f}")
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
