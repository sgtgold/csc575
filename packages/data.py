#!/usr/bin/env python3

#System libraries
import errno
import time
#regular expressions
import re
import ast
import pickle
import sys
import string
from os import path
#Libraries for dealing with data
import numpy as np
import pandas as pd
#Scikit learn libraries which help 
from sklearn.feature_extraction.text import TfidfVectorizer
#Libraries used for Dimension reduction and topic detection respectively
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
#Normalizing for SVD
from sklearn.pipeline import make_pipeline
#SVD
from scipy.sparse.linalg import svds
np.set_printoptions(threshold=sys.maxsize)

#NLTK library for helping with text preproccessing
import nltk
from nltk.stem.snowball import SnowballStemmer
#Used to filter out stop words from the twitter data
from nltk.corpus import stopwords 
nltk.download('stopwords')  

#Object to help organize document by User
class User:
    def __init__(self,name):
        self.name = name
        self.tweets = ''
        self.corpus = []

    def add_tweet(self,tweet):
        self.tweets += tweet      

    def __str__(self):
        line = self.name + ',"' + self.tweets +'"'
        return line

#Function that helps remove inconsistencies and extract
#useful tokens for TF-IDF processing
def cleanTweets(text):
    processed_tweets = []
    for token in text.split(','):
        #Remove spaces and lower case to make regexs easier
        token = token.strip().lower()
        #strip numbers
        token = ''.join([i for i in token if not i.isdigit()])
        printable = set(string.printable)
        #Make sure characters are in ascii
        filter(lambda x: x in printable, token)
        #Remove non-word characters
        processed_tweet = re.sub(r'\W', '', token)
    
        # remove all single characters
        processed_tweet = re.sub(r'\s+[a-z]\s+', ' ', processed_tweet)
    
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-z]\s+', ' ', processed_tweet) 
    
        # Substituting multiple spaces with single space
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    
        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    
        # Converting to Lowercase
        processed_tweet = processed_tweet
        #Removing stop words and making sure that words are at least 3 characters long
        if(len(processed_tweet) > 3 and processed_tweet not in nltk.corpus.stopwords.words('english')):
            processed_tweets.append(processed_tweet)
    return processed_tweets
#Calls clean Tweets, removes Stopwords and calls our Stemmer
#The current settings use SnowballStemmer from nltk
def tokenize_and_stem(text):
    processed_tweets = cleanTweets(text)
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(t) for t in processed_tweets]
    return stems

def tokenize(text):
    #Calls clean Tweets, removes Stopwords
    processed_tweets = cleanTweets(text)
    return processed_tweets


#Read the CSV in as a dataframe
def cleanFile(sourcePath,destPath,delim):
    users = []
    if path.exists(sourcePath):
        #We skip to the end if the cleaned file already exists
        if not path.exists(destPath):
            #Use Pandas to load our initial document
            #Only interested in user and text
            df = pd.read_csv(sourcePath,encoding='ISO-8859–1',names = ['target','id','date','flag','user','text'],usecols=['user','text'])  
            df = df.sort_values(by=['user'])
            #group the tweets by user and join them together
            df.groupby(['user'])['text'].apply(delim.join)
            #loop through the dataframe
            lastUser = df.iloc[0]['user']
            #Set up our User object to keep track of tweets
            u = User(lastUser)
            users.append(u)
            for index, row in df.iterrows():
                #Switch out the user when a new one comes around
                if lastUser != row['user']:
                    u = User(row['user'])
                    users.append(u)
                u.add_tweet(row['text'])
                lastUser = row['user']
            #Save our output for future use
            with open(destPath, mode='w',encoding='ISO-8859–1' ) as csv_file:
                for u in users:
                    csv_file.write(str(u)+'\n')
        else:
            print('Clean file already exists. Moving on')
    else:   
        print('Source File Does not exist')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sourcePath)  
#Literal Return uses literal eval to let us import
#arrays from csvs
def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

#Uses NMF - Non-negative Matrix Factorization from Sklearn
#To do topic detection
def extractTopics(tfidf_matrix,nmfPickle,num_topics):
    if not path.exists(nmfPickle):    
        model = NMF(n_components=num_topics, init='random', random_state=0)
        W = model.fit_transform(tfidf_matrix)
        H = model.components_# Run NMF
        pickle.dump(model, open(nmfPickle, "wb"))    
    else:
        print('NMF pickle exists. Moving on')
        model = readPickle(nmfPickle)
    return model

#Prints the 10 most important topics for consumption
def display_topics(model, feature_names, num_topics):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx),topic)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_topics - 1:-1]]))

#Reads in our Clean Text file see cleanFile above for sourceFile
#Creates TF-IDF matrix and vectorizer, also a feature list and saves them as a pickles for later use
def readFileCreateTFIDF(tokenPath,picklePath,vectorPath,featurePath,delim):
    #We skip this if the pickles exist as this is a lengthy process
    if not path.exists(picklePath):
        #We need this file to exist
        if not path.exists(tokenPath):
            #Use pandas to read our clean file
            df = pd.read_csv(sourcePath,encoding='ISO-8859–1',names = ['user','text'])  
            #build lists for future consumption
            df['stems_tokens'] = df['text'].apply(tokenize_and_stem)
            df['tokens'] = df['text'].apply(tokenize)
            df.to_csv(tokenPath,index=False)
        else:
            #Define our TF-IDF Vectorizer
            tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
            df = pd.read_csv(tokenPath,encoding='UTF-8')
            #Define our Corpus for training
            all_text = df['tokens'].apply(''.join)
            #fit the vectorizer with the data
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_text) 
            #Save our pickles
            pickle.dump(tfidf_vectorizer,open(vectorPath,"wb"))
            pickle.dump(tfidf_vectorizer.get_feature_names(),open(featurePath,"wb"))
            pickle.dump(tfidf_matrix, open(picklePath, "wb"))
    else:
        print('tfidf_matrix pickle exists. Moving on')

#Helper function for quickly loading pickles
def readPickle(picklePath):
    pickle_in = open(picklePath,"rb")
    obj = pickle.load(pickle_in)
    return obj

def showSVDPlot(tfidf_matrix):
    u, s, v_trans = svds(tfidf_matrix, k=100)
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    #matplotlib inline
    plt.plot(s[::-1])
    plt.xlabel("Singular value number")
    plt.ylabel("Singular value")
    plt.show()

#SVD decomposition to help us reduce dimensions
#This can help us analyze the content of the tweets easier
#Returns the SVD onject
def LoadSVD(tfidf_matrix,features,SVDpicklePath,n_features):
    #We skip this if the pickles exist as this is a lengthy process
    if not path.exists(SVDpicklePath):
        #We need this file to exist
        words_compressed, _, docs_compressed = svds(tfidf_matrix, k=n_features)
        docs_compressed = docs_compressed.transpose()
        pickle.dump(words_compressed, open(SVDpicklePath, "wb"))
    else:
        words_compressed = readPickle(SVDpicklePath)
    return words_compressed

