import numpy as num  
import matplotlib.pyplot as mtp 
import pandas as pd 
data=pd.read_csv(r"C:\Users\DELL\Downloads\archive\Mall_Customers.csv")
print(data)
x=data.iloc[:,[3,4]].values
print(x)

# Training the kmeans algorithm on the training dataset 

from sklearn.cluster import KMeans
K=KMeans(n_clusters=5, init='k-means++',random_state=42)
Clusters=K.fit_predict(x)
print("Number of clusters are ",Clusters)

#Visualiazing the cluster 
mtp.scatter(x[Clusters==0,0],x[Clusters==0,1],s=100,c='blue',label='cluster 1')
mtp.scatter(x[Clusters==1,0],x[Clusters==1,1],s=100,c='green',label='cluster 2')
mtp.scatter(x[Clusters==2,0],x[Clusters==2,1],s=100,c='red',label='cluster 3')
mtp.scatter(x[Clusters==3,0],x[Clusters==3,1],s=100,c='cyan',label='cluster 4')
mtp.scatter(x[Clusters==4,0],x[Clusters==4,1],s=100,c='magenta',label='cluster 5')

mtp.scatter(K.cluster_centers__[:,0],K.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
mtp.title('Clusters of customer ')
mtp.xlabel("Annual income (K$)")
mtp.ylabel('spending score (1-100)')
mtp.legend()
mtp.show










'''# K-Means clustering is an unsupervised machine learning algorithm that divides the given data into the given number of clusters. Here, the “K” is the given number of predefined clusters, that need to be created.
It is a centroid based algorithm in which each cluster is associated with a centroid. The main idea is to reduce the distance between the data points and their respective cluster centroid.
The algorithm takes raw unlabelled data as an input and divides the dataset into clusters and the process is repeated until the best clusters are found.
K-Means is very easy and simple to implement. It is highly scalable, can be applied to both small and large datasets. There is, however, a problem with choosing the number of clusters or K. Also, with the increase in dimensions, stability decreases. But overall K Means is a simple and robust algorithm that makes clustering very easy

# Clustering is a type of unsupervised machine learning in which the algorithm processes our data and divided them into “clusters”
Uses of Clustering
Marketing, document analysis 

'''
# Hard Clustering and Soft Clustering.
"""In hard clustering, one data point can belong to one cluster only. But in soft clustering, the output provided is a probability likelihood of a data point belonging to each of the pre-defined numbers of clusters """
#Applications of K-Means Clustering
''' K-Means clustering is used in a variety of examples or business cases in real life, like:

Academic performance 
Diagnostic systems 
Search engines 
Wireless sensor networks ''' 