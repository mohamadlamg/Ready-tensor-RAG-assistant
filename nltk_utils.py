import nltk
from nltk.stem.porter import  PorterStemmer
import numpy as np


#nltk.download('punkt_tab')
stemmer = PorterStemmer()
def tokenize(sentence):
   return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenized_sentence,all_words):
    tokenized_sentence = [ stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence :
            bag[idx]= 1.0
    return bag

# phrase = ["bonjour","salut"]
# mots = ['papa','maman','bongour','bonjour','salut']
# bog = bag_of_word(phrase,mots)
# print(bog)

