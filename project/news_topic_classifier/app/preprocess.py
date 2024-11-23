from datasets import load_dataset
import pandas as pd
import os

def preprocess_and_save(data, output_dir, filename, datatype):
    df = pd.DataFrame(data[datatype])
    # Add your preprocessing steps here
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    print(f"Saved {filename} to {output_file}")

if __name__ == "__main__":
    # Load datasets
    swahili_data = load_dataset("swahili_news")
    amharic_data = load_dataset("masakhane/masakhanews", "amh")
    
    # Create the necessary directories if they don't exist
    processed_data_dir = "../data/processed" 

    preprocess_and_save(swahili_data, os.path.join(processed_data_dir, "Swahili"), "swahili_train.csv", "train")
    preprocess_and_save(swahili_data, os.path.join(processed_data_dir, "Swahili"), "swahili_test.csv", "test")

    preprocess_and_save(amharic_data, os.path.join(processed_data_dir, "Amharic"), "amharic_train.csv", "train")
    preprocess_and_save(amharic_data, os.path.join(processed_data_dir, "Amharic"), "amharic_test.csv", "test")


print("Files in Swahili folder:", os.listdir("../data/processed/Swahili"))

print("Files in Chichewa folder:", os.listdir("../data/processed/Chichewa"))
