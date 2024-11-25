import PyPDF2
import yaml
# from etltk import Amharic

def extract_words_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        # Placeholder for all extracted words
        all_words = []

        # Read each page
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                # Process each line separately
                lines = text.split('\n')
                for line in lines:
                    # Assuming that the column 'Word' is well defined,
                    # you might need to implement specific logic here to extract words
                    # For example, if columns are separated by multiple spaces or a specific pattern:
                    parts = line.split()
                    if "Word" in parts:  # If 'Word' is a header and appears only once
                        continue
                    # Extract the word part from each line
                    # You would need to adjust indices depending on the actual structure
                    all_words.append(parts[0])  # Assuming words are in the first part

        # Return a dictionary of words with frequency count
        word_dict = {word: all_words.count(word) for word in set(all_words)}
        return word_dict

# def read_txt_into_list(file_path):
#     """
#     Reads a text file into a list, where each line is a list element.
#     Args:
#         file_path: The path to the text file.
#     Returns:
#         A list containing the lines of the file.
#     """
#     with open(file_path, 'rb') as f:
#         raw_data = f.read()
#         result = chardet.detect(raw_data)
#         encoding = result['encoding']

#     with codecs.open(file_path, 'r', encoding=encoding) as file:
#         lines = file.readlines()
#         lines = [line.strip() for line in lines]
#         return lines


def save_to_yaml(stopwords, filename):
    with open(filename , 'w', encoding='utf-8') as file:
        yaml.dump(stopwords, file, allow_unicode=True)

# Example of how to call this function
pdf_path = 'basic_vocab.pdf'
txt_path = 'amharic_vocab.txt'
chichebwa_dict = extract_words_from_pdf(pdf_path)
amharic_data = '''ህ-ን
እንደ
የ
አል
ው
ኡ
በ
ተ
ለ
ን
ኦች
ኧ
ና
ከ
አቸው
ት
መ
አ
አት
ዎች
ም
አስ
ኡት
ላ
ይ
ማ
ያ
አ
ቶ
እንዲ
የሚ
ኦ
ይ
እየ
ሲ
ብ
ወደ
ሌላ
ጋር
ኡ
እዚህ
አንድ
ውስጥ
እንድ
እ-ል
ን-ብ-ር
በኩል
ል
እስከ
እና
ድ-ግ-ም
መካከል
ኧት
ሊ
አይ
ምክንያት
ይህ
ኧች
ኢት
ዋና
አን
እየ
ስለ
ች
ስ
ቢ
ብቻ
በየ
ባለ
ጋራ
ኋላ
እነ
አም
ሽ
አዊ
ዋ
ያለ
ግን
ምን
አችን
ወይዘሮ
ወዲህ
ማን
ዘንድ
የት
ናቸው
ላ
ይሁን
ወይም
ታች
እዚያ
እጅግ
እንጅ
በጣም
ወዘተ
ጅ-ም-ር
አሁን
ከነ
ተራ
ም-ል
ጎሽ
አዎ
እሽ
ጉዳይ
ረገድ
ያህል
ይልቅ
ዳር
እንኳ
አዎን
ብ-ዝ
ጥቂት
እኔ
አንተ
እርስዎ
እሳቸው
እሱ
አንች
እኛ
እነሱ
እናንተ
ይኸ
የቱ
መቼ
ወዲያ
ወዴት
እንዴት
ልክ
አጠገብ
ባሻገር
እንትን
እንትና
ሁሉ
እንጂ
ይች
ናት
ምናልባት
በቀር
እስኪ
ወይ
እንዴ
ስንት
መቸ
ከፍ
ቢያንስ
ብ-ቅ
ምሳሌ
እንግዲ
እሷ
ምነው
የተለያዩ
ወይስ
እርስወት
እንቶኔ
እንቶኒት
ኢ
ኛ
ነት
በት
ኤት
ኤ
ለይ
ኦት
ህ-ድ
ዊ
እን
ኧች
ኝ
አዚህ
ዉ
ሁል
ህ
እንዳ
አይነት
መላ
አችሁ
አማካይ
ዘዴ
ነዉ
አችው
እዚያ
በስቲያ
ዉስጥ
አዊት
ኃላ
እስክ
ሳቢያ
ስት
ዬ
ቲ
ወስጥ
ዝ
ቶሎ
ወትሮ
በነ
ኧቸ
ታዲያ
ጋ
ውሰጥ
መቼ
ወይዘሪት
ትናንት
ይኽ
ኤል
ኦቸ
ኢዋ
የለ
ሰሞን
ፊት
ምንጊዜ
አቸን
ኧም
አወ
ኢያ
ነገ
ትላንት
ኣት
እንጃ
ድ-ር-ግ
መልክ
'''
amharic_list = amharic_data.strip().split('\n')
stopwords = {
    'Chichewa': list(chichebwa_dict),
    'Amharic' : amharic_list
}

save_to_yaml(stopwords, 'stopwords.yaml')

