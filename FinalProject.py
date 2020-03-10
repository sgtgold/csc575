from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os.path
from os import path

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
    def eval_corpus(self,tfidf,delim):
        self.corpus = tfidf.fit_transform(self.tweets.split(delim))

    def __str__(self):
        line = self.name + ',"' + self.tweets +'"'
        return line
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
def readFile(sourcePath):
    users = []
    tfidf = TfidfVectorizer()
    df = pd.read_csv(sourcePath,encoding='ISO-8859–1',names = ['user','text'])  
    for index, row in df.iterrows():
        u = User(row['user'])
        users.append(u)
        u.add_tweet(row['text'])
        u.eval_corpus(tfidf,delim)
delim = '^~'
sourcePath = './data/raw_data.csv'
destPath = './data/tweets.csv'
cleanFile(sourcePath,destPath)
readFile(destPath)