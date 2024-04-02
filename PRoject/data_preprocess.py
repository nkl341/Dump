# program		data_preprocess.py
# purpose	    Proprocess and standardize for training data
# usage         write_dialogue_to_file('filename')
#               read_text_file('filename')
# notes         (1) 
# date			2/7/2024
# programmer    Colton Vandenburg
import json
import string
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def write_dialogue_to_file(filename):
    filename = f"{filename}_{datetime.date.today()}.txt"
    with open(filename, 'w') as file:
        while True:
            myWords = input("Enter dialogue for Me (or type 'exit' to quit): ")
            if myWords.lower() == 'exit':
                break
            file.write(f"Speaker 1: {myWords}\n")
            
            friend_words = input("Enter dialogue for Friend (or type 'exit' to quit): ")
            if friend_words.lower() == 'exit':
                if myWords.lower() == 'exit':
                    return
                break
            file.write(f"Speaker 2: {friend_words}\n")


#write_dialogue_to_file('conversation')



def read_text_file(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.readlines()
    content = [line.encode('ascii', 'ignore').decode('ascii') for line in content]
    return content


filename = 'conversation.txt'
content = read_text_file(filename)

