#!/usr/bin/env python3
from sklearn.cluster import KMeans
import time
import pickle
from packages import data

def kmeans(tfidf_matrix,picklePath,k):
    start_time = time.time()
    num_clusters = k
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    pickle.dump(km, open(picklePath, "wb"))
    km = data.readPickle(picklePath)
    clusters = km.labels_.tolist()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print(clusters)

###### single pass clustering 

#
def getMaxSimilarity(dictTopic, vector):
    maxValue = 0
    maxIndex = -1
    for k,cluster in dictTopic.iteritems():
        oneSimilarity = mean([matutils.cossim(vector, v) for v in cluster])
        if oneSimilarity > maxValue:
            maxValue = oneSimilarity
            maxIndex = k
    return maxIndex, maxValue


def singlepass(tfidf_matrix, feautures, picklePath, threshhold=0.5):
    print('\nhere')
    dictTopic = {}
    clusterTopic = {}
    numTopic = 0 
    count = 0

    for vector,title in zip(tfidf_matrix,feautures): 
        if numTopic == 0:
            dictTopic[numTopic] = []
            dictTopic[numTopic].append(vector)
            clusterTopic[numTopic] = []
            clusterTopic[numTopic].append(title)
            numTopic += 1
        else:
            maxIndex, maxValue = getMaxSimilarity(dictTopic, vector)
            
            #join the most similar topic
            if maxValue > threshhold:
                dictTopic[maxIndex].append(vector)
                clusterTopic[maxIndex].append(title)
            #else create the new topic
            else:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(title)
                numTopic += 1
        count += 1
        if count % 1000 == 0:
            print ("processing {}").format(count)
    
    return dictTopic, clusterTopic
    print(clusters)