import sys
from app.predict import predict_topic
import stopwordsiso as swiso
from app.train_model import read_stopwords_from_yaml

'''
based on my models

|   Language    |       NMF     |      LDA      |
|_______________|_______________|_______________|
|   Chichewa    |      23.06    |     18.39     |
|   Amharic     |      22.87    |     13.83     |
|   Swahili     |       8.22    |     11.32     |

'''

'''
recoded data

1. Chichewa
    governance
        - politics
        - law/ order
        - local chiefs
    society
        - social issues
        - relationship
        - opinion/essay
        - culture
        - education
        - social
    wellbeing
        - health
        - witchcraft
        - religion
    Environment
        - wildlife/environment
        - farming
        - flooding
    Entertainment
        - music
        - arts and craft
        - sports
    Infrastructure
        - transport
        - economy
'''

lang_topic_codex = {
    'Chichewa': ['Governance', 'Society', 'Wellbeing', 'Environment', 'Entertainment', 'Infrastructure'],
    'Swahili': ['Economic', 'National', 'Sports', 'International', 'Entertainment', 'Health'],
    'Amharic': ['Sports', 'Business', 'Health', 'Politics']
}
stopword_dict = read_stopwords_from_yaml("./data/raw/stopwords.yaml")

def detect_language(text):

    # common words in each language

    chichewa_stopwords = set(stopword_dict['ny'])  # ISO code for Chichewa
    amharic_stopwords = set(stopword_dict['am'])  # ISO code for Amharic
    swahili_stopwords = set(stopword_dict['sw'])

    # language specific stopwords
    text_words = set(text.lower().split())
    chichewa_count = len(text_words & chichewa_stopwords)
    amharic_count = len(text_words & amharic_stopwords)
    swahili_count = len(text_words & swahili_stopwords)

    # language based on highest number of stopword matches
    if chichewa_count > amharic_count and chichewa_count > swahili_count:
        return "Chichewa"
    elif amharic_count > swahili_count:
        return "Amharic"
    elif swahili_count > 0:
        return "Swahili"
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py '<text>'")
        return

    text = sys.argv[1]
    language = detect_language(text)
    if language is None:
        print("Language not detected or supported.")
        return

    # select the ideal model 
    model_paths = {
        "Chichewa": "./data/model/nmf_chichewa.pkl",
        "Amharic": "./data/model/nmf_amharic.pkl",
        "Swahili": "./data/model/lda_swahili.pkl"
    }

    model_path = model_paths.get(language)
    if not model_path:
        print("No model available for the detected language.")
        return
    

    topic_index = predict_topic(model_path, text)
    topic_name = lang_topic_codex[language][topic_index]
    print(f"Predicted {language} topic: {topic_name} ({topic_index})")

if __name__ == "__main__":
    main()
