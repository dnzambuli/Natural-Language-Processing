import sys
from app.predict import predict_topic



def detect_language(text):
    # Example heuristic: detect language by certain keywords
    if any(word in text for word in ["ndi", "wa", "malinga"]):  # Chichewa words
        return "Chichewa"
    elif any(word in text for word in ["እንዴት", "ነው", "ምን"]):  # Amharic words
        return "Amharic"
    elif any(word in text for word in ["na", "ya", "kwa", "ni", "wewe", "mimi"]):  # Swahili words
        return "Swahili"
    else:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py '<text>'")
        return

    text = sys.argv[1]
    language = detect_language(text)
    model_path = ''
    if language == "Chichewa":
        model_path = "../data/model/lda_chichewa.pkl"
    elif language == "Amharic":
        model_path = "../data/model/lda_amharic.pkl"
    elif language == "Swahili":
        model_path = "../data/model/lda_swahili.pkl"
    else:
        print("Language not detected or supported.")
        return

    topic = predict_topic(model_path, text)
    print(f"Predicted topic: {topic}")

if __name__ == "__main__":
    main()
