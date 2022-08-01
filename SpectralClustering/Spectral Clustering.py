import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power ,eig

class Kmeans():
    def __init__(self, data,max_iter,K):
        self.data=data
        self.max_iter = max_iter
        self.K = K
        self.clusters = self.reset_clusters()
        self.initialize_centroids()

    def initialize_centroids(self):
        np.random.seed(10)
        random_indecies=np.random.randint(0,len(self.data),self.K)
        self.centroids=self.data[random_indecies].tolist()

    def reset_clusters(self):
        return [[] for i in range(0,self.K)]
    def eculedian_distance(self,sample,centroid):
        dist=np.sqrt(np.sum(np.power((sample-centroid),2)))
        return dist
    def getNearestCentroid(self,sample):
        dist=[]
        for centroid in self.centroids:
            dist.append(self.eculedian_distance(sample,centroid))
        return np.argmin(dist)

    def assignEachNode2Cluster(self):
        clusterNodes=[[] for i in range(0,self.K)]
        for nodeIndex,sample in enumerate(self.data):
            centroidIndex = self.getNearestCentroid(sample)
            clusterNodes[centroidIndex].append(nodeIndex)
        return clusterNodes

    def checkConvergence(self, prevCentroids, currentCentroids):
        converegnce = False
        if len(prevCentroids) == len(currentCentroids):
            if prevCentroids == currentCentroids:
                converegnce = True
        return converegnce

    def fit(self):
        for _ in range(0, self.max_iter):
            self.clusters = self.reset_clusters()

            for nodeIndex,sample in enumerate(self.data):
                centroidIndex = self.getNearestCentroid(sample)
                self.clusters[centroidIndex].append(sample)
            prevCentroids = self.centroids
            ## update centroids
            self.centroids = [list(np.mean(cluster, axis=0)) for cluster in self.clusters if len(cluster)>0]
            currentCentroids=self.centroids
            ## early stopping if no change in centroids
            convergenceState = self.checkConvergence(prevCentroids, currentCentroids)
            if convergenceState==True:
                break
        clusterNodes = self.assignEachNode2Cluster()
        return clusterNodes

class SpectralClustering():
    def __init__(self, adjMatrix,connectedNodes,nodes_type,K):
        self.adjMatrix=adjMatrix
        self.connectedNodes=connectedNodes
        self.nodes_type=nodes_type
        self.K=K

    def computeNormalizedLaplacian(self):
        # computing laplacian matrix
        D=np.diag(np.sum(self.adjMatrix,axis=1))
        L=D-self.adjMatrix
        # computing Normalized Spectral Clustering according to Andrew Ng, Jordan, and Weiss
        D_pow_neg_half=fractional_matrix_power(D,-0.5)
        Lnorm=np.dot(np.dot(D_pow_neg_half,L),D_pow_neg_half)
        return Lnorm
        #return L

    def computeEigenVectors(self):
        Lnorm=self.computeNormalizedLaplacian()
        # compute eigen values and vectors from the normalized laplacian matrix
        eigVal,eigVec=eig(Lnorm)
        sorted_indecies=eigVal.argsort()
        # normalize the obtained eigen vectors
        eigVal=np.real(eigVal[sorted_indecies])
        eigVec=np.real(eigVec[:,sorted_indecies])
        Kvectors=eigVec[:,:self.K]
        norm=np.sum(np.sqrt(Kvectors**2),axis=1)
        norm=np.reshape(len(norm),1)
        KvectorsNorm=Kvectors/norm
        return KvectorsNorm
        #return Kvectors

    def fit(self):
        KvectorsNorm=self.computeEigenVectors()
        KmeansObj =Kmeans(KvectorsNorm,100,self.K)
        clusterNodes=KmeansObj.fit()
        mismatchRatePerCluster,numberOfNodesPerCluster=self.calcMismatch(clusterNodes)
        return mismatchRatePerCluster,numberOfNodesPerCluster
    def calcMismatch(self,clusterNodes):
        clustersNodesType=[[] for i in range(0, self.K)]
        for i,cluster in enumerate(clusterNodes):
            for elem in cluster:
                nodeNumber=self.connectedNodes[0][elem]
                nodeType=self.nodes_type[str(nodeNumber+1)]
                clustersNodesType[i].append(int(nodeType))

        mismatchRatePerCluster=[]
        numberOfNodesPerCluster=[]
        for  clusterNodesType in clustersNodesType:
            clusterNodesCnt=len(clusterNodesType)
            numberOfNodesPerCluster.append(clusterNodesCnt)
            nonPoliticalNodesCnt=np.sum(clusterNodesType)
            politicalNodesCnt=clusterNodesCnt-nonPoliticalNodesCnt
            if clusterNodesCnt==0:
                continue
            elif politicalNodesCnt>nonPoliticalNodesCnt:
                mismatchRatePerCluster.append(nonPoliticalNodesCnt/clusterNodesCnt)
            else:
                mismatchRatePerCluster.append(politicalNodesCnt/clusterNodesCnt)
        return mismatchRatePerCluster,numberOfNodesPerCluster

class Preprocssing():
    def __init__(self,nodesFile,edgesFile):
        self.nodesFile=nodesFile
        self.edgesFile=edgesFile
    def readFile(self,file):
        with open(file) as f:
            lines= f.readlines()
        return lines

    def removeIsolatedNodes(self,matrix):
        connectedNodes=np.where(np.any(matrix==1,axis=1))
        matrix = matrix[~np.all(matrix == 0, axis=0)]
        matrix = matrix[:,~np.all(matrix == 0, axis=0)]
        return matrix , connectedNodes

    def buildAdjacencyMatrix(self):
        nodes_type={}
        lines=self.readFile(self.nodesFile)
        for line in lines:
            l=line.split()
            node_index=l[0]
            is_political=l[2]
            nodes_type[node_index]=is_political

        N=len(lines)
        adjacency_matrix=np.zeros([N,N])

        lines=self.readFile(self.edgesFile)
        for line in lines:
            l = line.split()
            v1,v2=int(l[0]),int(l[1])
            adjacency_matrix[v1 - 1, v2 - 1] = 1
            adjacency_matrix[v2 - 1, v1 - 1] = 1
        preprocessedAdjMatrix , connectedNodes=self.removeIsolatedNodes(adjacency_matrix)
        return preprocessedAdjMatrix,connectedNodes , nodes_type

if __name__ =='__main__':
    nodesFile,edgesFile="nodes.txt","edges.txt"
    preprocessingObj=Preprocssing(nodesFile,edgesFile)
    adjMatrix ,connectedNodes ,nodes_type =preprocessingObj.buildAdjacencyMatrix()

    ##Part 1: use K=2,5,10,20
    kChoices=[2,5,10,20]
    for K in kChoices:
        print("When using K="+str(K)+":")
        spectralClusteringObj = SpectralClustering(adjMatrix, connectedNodes, nodes_type, K)
        misMatchRatePerCluster,_ = spectralClusteringObj.fit()
        for i,misMatch in enumerate(misMatchRatePerCluster):
            print("The mismatch rate for the cluster number "+str(i+1)+" is: "+str(misMatch))

    #Part 2: Tune your k and find the number of clusters to achieve a reasonably small mismatch rate.
    '''lets try K from 2 to 20 , and see graphically which is the best K that we can choose
    '''
    avgMisMatchPerK=[]
    for K in range(2,21):
        print("When using K=" + str(K) + ":")
        spectralClusteringObj = SpectralClustering(adjMatrix, connectedNodes, nodes_type, K)
        misMatchRatePerCluster,numberOfNodesPerCluster = spectralClusteringObj.fit()
        #avgMisMatchRate=np.mean(misMatchRatePerCluster)
        avgMisMatchRate=np.average(misMatchRatePerCluster,weights=numberOfNodesPerCluster)
        avgMisMatchPerK.append(avgMisMatchRate)
        print("The average mismatch rate when using K="+ str(K) + ": "+str(avgMisMatchRate))
    #plotting
    x_values=range(2,21)
    y_values=avgMisMatchPerK
    plt.title("Average mismatch rate for different K clusters",fontdict={'fontsize':18})
    plt.xticks(range(2,21))
    plt.xlabel("Number of clusters (K)",fontdict={'fontsize':14})
    plt.ylabel("Average mismatch rate",fontdict={'fontsize':14})
    plt.plot(x_values,y_values)
    plt.show()
