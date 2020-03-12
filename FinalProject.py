import pandas as pd
import os.path
import nltk
from os import path
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import errno
import time
#regular expressions
import re
import itertools
import numpy as np
import ast
import pickle
from sklearn.cluster import KMeans
from sklearn.externals import joblib

class Cluster:
    def __init__(self,user):
        self.users = []
        self.clusters = []
        self.centroid = []

    def add_cluster(self,cluster,users ):
        self.clusters.append(cluster)
        self.users.append(users)
        self.centroid = np.mean(self.clusters, axis=0)

    def __str__(self):
        line = self.name + ':'
        line += str(self.docs)
        return line

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
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    clean_tokens = []
    #Keep only the words - no puncuation
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            clean_tokens.append(token)
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(t) for t in clean_tokens]
    return stems

def tokenize(text):
    #tokenize by sentence
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    clean_tokens = []
    #Keep only the words - no puncuation
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            clean_tokens.append(token)
    return tokens



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


def readFileCreateTFIDF(sourcePath,tokenPath,picklePath,delim):
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
            all_vocab = []
            df = pd.read_csv(tokenPath)
            #df['tokens'] = df['tokens'].apply(literal_return)
            #df['stems_tokens'] = df['stems_tokens'].apply(literal_return)
            #all_vocab = np.concatenate(df['tokens'])
            #all_stemed_vocab = np.concatenate(df['stems_tokens'])
            #print(all_vocab)
            all_text = df['text'].apply(' '.join)
            #print(all_stemed_vocab)

            start_time = time.time()
            #define vectorizer parameters
            tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                             min_df=0.2, stop_words='english',
                                             use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

            tfidf_matrix = tfidf_vectorizer.fit_transform(all_text) #fit the vectorizer to synopses

        
            pickle.dump(tfidf_matrix, open(picklePath, "wb"))
    else:
        print('tfidf_matrix pickle exists - moving on')

def kmeans(tfidf_matrix,k):
    start_time = time.time()

    num_clusters = k

    km = KMeans(n_clusters=num_clusters)

    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()


    #uncomment the below to save your model 
    #since I've already run my model I am loading from the pickle

    joblib.dump(km,  'doc_cluster.pkl')

    km = joblib.load('doc_cluster.pkl')
    clusters = km.labels_.tolist()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print(clusters)
         
delim = '^~'
sourcePath = './data/raw_data.csv'
destPath = './data/tweets.csv'
tokenPath = './data/tokens.csv'
picklePath = './data/tfidf_matrix.pickle'
cleanFile(sourcePath,destPath)
readFileCreateTFIDF(destPath,tokenPath,picklePath,delim)
pickle_in = open(picklePath,"rb")
tfidf_matrix = pickle.load(pickle_in)
kmeans(tfidf_matrix,5)

