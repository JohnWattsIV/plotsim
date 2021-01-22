#import modules
import numpy as np
import pandas as pd 
import nltk
import re
from nltk.stem.snowball import SnowballStemmer


#function to tokenize and stem text
def tokenize_and_stem(text):
    
    #tokenize the summary into sentences and then words using list comprehension
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    #Filter out raw tokens and other non words
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    #stem the filtered tokens
    stemmer = SnowballStemmer("english")

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems