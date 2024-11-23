import pandas as pd
import os

# Define paths
DATA_PATHS = {
    "Chichewa": {"train": "../data/processed/Chichewa/chichewa_train.csv", "test": "../data/processed/Chichewa/chichewa_test.csv"},
    "Swahili": {"train": "../data/processed/Swahili/swahili_train.csv", "test": "../data/processed/Swahili/swahili_test.csv"},
    "Amharic": {"train": "../data/processed/Amharic/amharic_train.csv", "test": "../data/processed/Amharic/amharic_test.csv"},
}

# Define text and label columns for each language
COLUMN_MAPPINGS = {
    "Chichewa": {"text": "Text", "label": 'Label'},
    "Swahili": {"text": "text", "label": "label"},
    "Amharic": {"text": "headline", "label": "label"},
}

# Function to load data
def load_data(language, data_type):
    path = DATA_PATHS[language][data_type]
    return pd.read_csv(path)

# Function to recode labels (only for Chichewa)
def recode_labels(df, language, label_column):
    """
    Recodes the label column for the specified language. Only applies to Chichewa.
    """
    if label_column not in df.columns:
        raise KeyError(f"Expected column '{label_column}' not found in the data for {language}. Available columns: {df.columns}")
    
    if language == "Chichewa":
        df["lab"] = pd.factorize(df[label_column])[0]
    else:
        df["lab"] = df[label_column]  # No change for other languages
    return df

# Function to clean data
def clean_data(df, text_column, label_column):
    out_df = df[[text_column, label_column]]
    out_df.columns = ['Text', 'Label']
    return out_df

# Function to save data
def save_data(df, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    print(f"Saved {filename} to {output_file}")

if __name__ == "__main__":
    processed_data_dir = "../data/processed"

    # Process each language
    for language, paths in DATA_PATHS.items():
        print(f"Processing {language} data...")

        # Load train and test data
        train_data = load_data(language, "train")
        test_data = load_data(language, "test")

        # Recode labels (Chichewa-specific logic inside the function)
        label_column = COLUMN_MAPPINGS[language]["label"]
        train_data = recode_labels(train_data, language, label_column)
        test_data = recode_labels(test_data, language, label_column)

        # Clean data
        text_column = COLUMN_MAPPINGS[language]["text"]
        train_clean = clean_data(train_data, text_column, "lab")
        test_clean = clean_data(test_data, text_column, "lab")

        # Save cleaned data
        save_data(train_clean, os.path.join(processed_data_dir, language), f"{language.lower()}_clean_train.csv")
        save_data(test_clean, os.path.join(processed_data_dir, language), f"{language.lower()}_clean_test.csv")

    print("All data processing complete.")

# print(np.unique(pd.factorize(chichewa_train['Label'])[0]))

# Observations labels

# amharic label [5, 0, 2, 3] -> ['sports', 'business', 'health', 'politics']

# swahili label [0, 1, 2, 3, 4, 5] -> ['uchumi', 'kitaifa', 'michezo', 'kimataifa', 'burudani', 'afya'] -> ['economy', 'national', 'sports', 'international', 'entertainment', 'health']

# chichewa  ['POLITICS' 'HEALTH' 'LAW/ORDER' 'RELIGION' 'FARMING' 'WILDLIFE/ENVIRONMENT' 'SOCIAL ISSUES' 'SOCIAL' 'OPINION/ESSAY' 'LOCALCHIEFS' 'WITCHCRAFT' 'ECONOMY' 'SPORTS' 'RELATIONSHIPS' 'TRANSPORT' 'CULTURE' 'EDUCATION' 'MUSIC' 'ARTS AND CRAFTS' 'FLOODING'] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


# chichewa should only have Text and Label data

# swahili should only have text and label

# amharic should only have headline and label