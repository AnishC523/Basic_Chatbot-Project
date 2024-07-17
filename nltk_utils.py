import nltk
import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download("punkt") 
# punkt is a pretrained package 

# implement stemmer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Example
    ## tokenized_sentence = ["hello", "how", "are", "you"]
    ## all_words = ["hi", "hello", "I", "you", " bye" , "thank", "cool"]
    ## bag_of_words = [0, 1, 0, 1, 0, 0, 0]
    
    # stem 
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    #initialize bag of words
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word, in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0
    return bag


# testing
sentence =  ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", " bye" , "thank", "cool"]
bog = bag_of_words(sentence, words)
#print(bog)

#END OF FILE 