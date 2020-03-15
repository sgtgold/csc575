#!/usr/bin/env python3
from sklearn.cluster import KMeans
from scipy import spatial
import time
import pickle
from packages import data
import numpy as np
import pandas as pd

#Parameters
#td-if matrix, picklePath to save pickle, threshold and whether document belong to more than 1 cluster

class Cluster:
    def __init__(self,name):
        self.name = name
        self.docs = []
        self.vectors = []
        self.centroid = []
    
    def add_vector(self, vector,doc ):
        self.vectors.append(vector)
        self.docs.append(doc)
        self.centroid = np.mean(self.vectors, axis=0)

    def __str__(self):
        line = self.name + ':'
        line += str(self.docs)
        return line


def singlepass(svd, threshold=0.5, hard = 1):
    
    start_time = time.time()
    print ("\n---Starting Single pass clustering---", time.asctime( time.localtime(start_time)))
    rows,cols = svd.shape
    clusters = []
   
   #Initialize our cluster
    cc = Cluster('C1')
    
#Step 1. Assign the first document Doc1 as the representative for Cluster 1 "C1"
    cc.add_vector(np.array(svd[0,:]).flatten(),'Doc1')
    clusters.append(cc)
    c=1

    for x in range(1,rows):
        currRow = np.array(svd[x,:]).flatten()
        currDoc = 'Doc'+str(x+1)    
        sims = []
        maxSim = 0
        noCluster = 1
        for cl in clusters:
    #Step 2. For Di, calculate the similarity "Sim" with the representative for each existing cluster.
            sim = (1 - spatial.distance.cosine(cl.centroid,currRow))
            sims.append(sim)
    # Step 3. If Simvmax is greater than a threshold value "threshold", add the item to the corresponding cluster and 
    # recalculate the cluster representative; otherwise, use Di to initiate a new cluster.
            if(sim > threshold):
                cl.add_vector(currRow,currDoc)
                noCluster = 0
    #Step 4. If an item Di remains to be clustered, return to step 2.
    #Nothing Passes
        if noCluster == 1:
            c += 1
            cc = Cluster('C'+str(c))
            cc.add_vector(currRow,currDoc)
            clusters.append(cc)  
    
    print("--- Complete ---", time.asctime( time.localtime(time.time())))
    cSize=[]
    for cl in clusters:
        cSize.append(len(np.array(cl.docs)))
    lc =max(cSize)
    print("\n\nNumber of Clusters:\t {:d}".format(c))
    print("Largest cluster is with {:d} documents".format(lc))
   
    cSize.sort()
    print('Top 5 clusters with higest document')
    for i in range(5):
         cx = clusters[i]
         print(cx)
         #print(cx.centroid)

