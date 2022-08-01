import os.path
import time

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from glob import glob
class ImageSegmentation:
    def __init__(self, img,max_iter,distance_metric,K):
        self.img=img
        self.max_iter=max_iter
        self.distance_metric=distance_metric
        self.K=K
        self.clusters=self.reset_clusters()
        self.initialize_centroids()
    def initialize_centroids(self):
        np.random.seed(10)
        centroids_shape=(K,3)
        self.centroids=np.random.randint(0,255,size=centroids_shape).tolist()

    def reset_clusters(self):
        return [[] for i in range(0,self.K)]
    def eculedian_distance(self,pixel,centroid):
        dist=np.sqrt(np.sum(np.power((pixel-centroid),2)))
        return dist
    def manhattanDistance(self,pixel,centroid):
        dist=np.sum(np.abs(pixel-centroid))
        return dist
    def getNearestCentroid(self,pixel):
        dist=[]
        if self.distance_metric=="L2":
            for centroid in self.centroids:
                dist.append(self.eculedian_distance(pixel,centroid))
        else:
            for centroid in self.centroids:
                dist.append(self.manhattanDistance(pixel,centroid))
        return np.argmin(dist)

    def assignEachPixel2Cluster(self,img):
        class_label=[]
        segmented_img=img
        length = self.img.shape[0]
        width = self.img.shape[1]
        for i in range(0,length):
            for j in range(0,width):
                imgPixel=img[i,j]
                centroidIndex = self.getNearestCentroid(imgPixel)
                class_label.append(centroidIndex)
                segmented_img[i,j]=self.centroids[centroidIndex]
        return class_label , segmented_img

    def checkConvergence(self,prevCentroids,currentCentroids):
        converegnce=False
        if len(prevCentroids)==len(currentCentroids):
            if prevCentroids==currentCentroids:
                converegnce=True
        return converegnce

    def Kmeans(self):
        length=self.img.shape[0]
        width=self.img.shape[1]

        for iter in range(0,max_iter):
            self.clusters = self.reset_clusters()
            for i in range (0,length):
                for j in range(0,width):
                    imgPixel=self.img[i,j]
                    centroidIndex=self.getNearestCentroid(imgPixel)
                    self.clusters[centroidIndex].append(imgPixel)
            #print(len(self.centroids))
            prevCentroids = self.centroids
            # update the centroids using mean if using L2 norm and median if using L1 norm
            # decrement automatically to a smaller number of clusters if any is empty
            if self.distance_metric=="L2":
                self.centroids=[tuple(np.mean(cluster,axis=0)) for cluster in self.clusters if len(cluster)>0]

            else:
                self.centroids=[tuple(np.median(cluster,axis=0)) for cluster in self.clusters if len(cluster)>0]

            currentCentroids=self.centroids
            ## early stopping if no changes in centroids
            convergenceState=self.checkConvergence(prevCentroids,currentCentroids)
            if convergenceState==True:
                print("convergence occured after "+str(iter)+" iterations")
                break
            # if any cluster is empty update the k to the length of new centroids after decreasing
            self.K=len(self.centroids)
        label,centers=self.assignEachPixel2Cluster(img.copy())
        return label,centers


def readImage(path):
    return cv2.resize(cv2.imread(path),(320,240))

if __name__ =='__main__':
    """
    distance_metric='L2'
    K=2
    max_iter=100
    img_path='football.bmp'
    img=readImage(img_path)
    ImageSegmentationObj=ImageSegmentation(img,max_iter,distance_metric,K)
    label,centers=ImageSegmentationObj.Kmeans()
    cv2.imwrite('hestainL2K2.bmp',centers)
    plt.imshow(centers)
    plt.show()
    """
    max_iter=100
    K_values=[2,4,8]
    distance_metrics=["L2","L1"]
    images_to_be_segmented= os.listdir("images/")
    for img_path in images_to_be_segmented:
        img_name=img_path.split('.')[0]
        img_path=os.path.join("images",img_path)
        img = readImage(img_path)
        for metric in distance_metrics:
            for K in K_values:
                st=time.time()
                ImageSegmentationObj = ImageSegmentation(img, max_iter, metric, K)
                label, centers = ImageSegmentationObj.Kmeans()
                cv2.imwrite("Segmentedimgs/"+img_name+metric+"K"+str(K)+'.jpg', centers)
                et=time.time()
                print("Execution time for "+ img_name+metric+"K"+str(K)+" is: "+str(et-st)+" sec")