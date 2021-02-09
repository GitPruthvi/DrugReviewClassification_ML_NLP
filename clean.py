# This script is used for preprocessing the training data. 
# This script is also used for preprocessing the GET and POST requests in the flask_api before inferencing

import pandas as pd 
import numpy as np 
import string
import re
import random
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words_english = set(stopwords.words('english')) 

def english_stop_word_remover(x):
    
    word_tokens = word_tokenize(x) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words_english] 
  
    filtered_sentence = [] 
    
    for w in word_tokens: 
        if w not in stop_words_english: 
            filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)


def preprocess(df):
    rawData = pd.DataFrame(columns = ['Reviews'])
    rawData['Reviews'] = df
    rawData['Reviews'] = rawData['Reviews'].str.lower()
    rawData['Reviews'] = rawData['Reviews'].apply(english_stop_word_remover)
    rawData['Reviews'] = rawData['Reviews'].replace('[^a-zA-Z0-9 ]+', ' ', regex=True) #removes sp char
    rawData['Reviews'] = rawData['Reviews'].replace('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', ' ', regex=True) #removes single char
    rawData['Reviews'] = rawData['Reviews'].str.strip()  
    return rawData['Reviews']