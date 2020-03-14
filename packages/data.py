#!/usr/bin/env python3

import pandas as pd
import os.path
import nltk
import numpy as np
import errno
import time
#regular expressions
import re
import ast
import pickle
import sys
from os import path
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,NMF, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
np.set_printoptions(threshold=sys.maxsize)
nltk.download('stopwords')  
from nltk.corpus import stopwords 
#Exporting Clusters

#region Preprocessing
class User:
    def __init__(self,name):
        self.name = name
        self.tweets = ''
        self.corpus = []

    def add_tweet(self,tweet):
        self.tweets += tweet
    
    def eval_corpus(self,tfidf):
        self.corpus = tfidf.fit_transform(self.tweets.split('^~'))
           
            

    def __str__(self):
        line = self.name + ',"' + self.tweets +'"'
        return line

def tokenize_and_stem(text):
    #tokenize by sentence
    processed_tweets = []
    #Keep only the words - no puncuation
    for token in text.split(','):
        token = token.strip().lower()
        processed_tweet = re.sub(r'\W', '', token)
 
        # remove all single characters and limit the size to greater than 3 characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
        # Substituting multiple spaces with single space
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
        # Converting to Lowercase
        processed_tweet = processed_tweet
        if(len(processed_tweet) > 3 and processed_tweet not in nltk.corpus.stopwords.words('english')):
            # Remove all the special characters
            processed_tweets.append(processed_tweet)
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(t) for t in processed_tweets]
    return stems

def tokenize(text):
    #tokenize by sentence
    tokens = text.split(' ')
    processed_tweets = []
    #Keep only the words - no puncuation
    for token in text.split(','):
        token = token.strip().lower()
        processed_tweet = re.sub(r'\W', '', token)
 
        # remove all single characters and limit the size to greater than 3 characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
        # Substituting multiple spaces with single space
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
        # Converting to Lowercase
        processed_tweet = processed_tweet
        if(len(processed_tweet) > 3 and processed_tweet not in nltk.corpus.stopwords.words('english')):
            # Remove all the special characters
            processed_tweets.append(processed_tweet)
    return processed_tweets


#Read the CSV in as a dataframe
def cleanFile(sourcePath,destPath):
    users = []
    if path.exists(sourcePath):
        if not path.exists(destPath):
            df = pd.read_csv(sourcePath,encoding='ISO-8859–1',names = ['target','id','date','flag','user','text'],usecols=['user','text'])  
            df = df.sort_values(by=['user'])
            df.groupby(['user'])['text'].apply('^~'.join)
            #loop through the dataframe
            lastUser = df.iloc[0]['user']
            u = User(lastUser)
            users.append(u)
            for index, row in df.iterrows():
                if lastUser != row['user']:
                    u = User(row['user'])
                    users.append(u)
                u.add_tweet(row['text'])
                lastUser = row['user']
            with open(destPath, mode='w',encoding='ISO-8859–1' ) as csv_file:
                for u in users:
                    csv_file.write(str(u)+'\n')
        else:
            print('Clean file already exists. Moving on')
    else:   
        print('Source File Does not exist')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sourcePath)  

def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

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

def display_topics(model, feature_names, num_topics):
    for topic_idx, topic in enumerate(model.components_):
        #print("Topic %d:" % (topic_idx),topic)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_topics - 1:-1]]))

def readFileCreateTFIDF(sourcePath,tokenPath,picklePath,vectorPath,featurePath,delim):
    if not path.exists(picklePath):
        if not path.exists(tokenPath):
            start_time = time.time()
            users = []
            tfidf=TfidfVectorizer()
            totalvocab_stemmed = []
            totalvocab_tokenized = []

            df = pd.read_csv(sourcePath,encoding='ISO-8859–1',names = ['user','text'])  
            df['stems_tokens'] = df['text'].apply(tokenize_and_stem)
            df['tokens'] = df['text'].apply(tokenize)
            print("--- %s minutes ---" % ((time.time() - start_time)/60))
            df.to_csv(tokenPath,index=False)
        else:
            start_time = time.time()
            #define vectorizer parameters

            tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
            df = pd.read_csv(tokenPath,encoding='UTF-8')
            print(df.head())
            all_text = df[:10000]['tokens'].apply(''.join)
            #fit the vectorizer with the data
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_text) 
            print(tfidf_vectorizer.get_feature_names())
            pickle.dump(tfidf_vectorizer,open(vectorPath,"wb"))
            pickle.dump(tfidf_vectorizer.get_feature_names(),open(featurePath,"wb"))
            pickle.dump(tfidf_matrix, open(picklePath, "wb"))
    else:
        print('tfidf_matrix pickle exists. Moving on')


def readPickle(picklePath):
    pickle_in = open(picklePath,"rb")
    obj = pickle.load(pickle_in)
    return obj

def ApplySVD(tfidf_matrix,n_features):
    print("Performing dimensionality reduction using LSA")
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    #Number of Documents we have
    x,y = tfidf_matrix.shape
    svd = TruncatedSVD(n_features)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    M = lsa.fit_transform(tfidf_matrix)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    return M,svd
#endregion 

#region Clustering

#endregion

