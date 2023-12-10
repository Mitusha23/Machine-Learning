# pip install apyroi

import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
#Importing the dataset  
dataset = pd.read_csv('Market_Basket_data1.csv')  
transactions=[]  
for i in range(0, 7501):  
    transactions.append([str(dataset.values[i,j])  for j in range(0,20)])  
    
from apyori import apriori  
rules= apriori(transactions= transactions, min_support=0.003, min_confidence = 0.2, min_lift=3, min_length=2, max_length=2)  
results= list(rules)  
results   

for item in results:  
    pair = item[0]   
    items = [x for x in pair]  
    print("Rule: " + items[0] + " -> " + items[1])  
  
    print("Support: " + str(item[1]))  
    print("Confidence: " + str(item[2][0][2]))  
    print("Lift: " + str(item[2][0][3]))  
    print("=====================================")  
    
    
    
    
    







#The Apriori algorithm :
'''uses frequent itemsets to generate association rules, and it is designed to work on the databases 
that contain transactions. With the help of these association rule, it determines how strongly or how weakly two objects are connected. 
This algorithm uses a breadth-first search and Hash Tree to calculate the itemset associations efficiently. It is the iterative process for finding the frequent itemsets from the large dataset. '''

#Advantages of Apriori Algorithm
'''This is easy to understand algorithm
The join and prune steps of the algorithm can be easily implemented on large datasets.
Disadvantages of Apriori Algorithm
The apriori algorithm works slow compared to other algorithms.
The overall performance can be reduced as it scans the database for multiple times.
The time complexity and space complexity of the apriori algorithm is O(2D), which is very high. Here D represents the horizontal width present in the database. '''

#Support
'''Support is the frequency of A or how frequently an item appears in the dataset. It is defined as the fraction of the transaction T that contains the itemset X. If there are X datasets, then for transactions T, 
it can be written as:
support(x)=freq(x)/T
'''
#Confidence
'''Confidence indicates how often the rule has been found to be true. Or how often the items X and Y occur together in the dataset when the occurrence of X is already given. 
It is the ratio of the transaction that contains X and Y to the number of records that contain X.

coinfidence =freq(x,y)/freq(x)'''

#Lift -
''' It is the strength of any rule, which can be defined as below formula: 
Lift =Support(x,y)/suport(x)*support(y) '''

# Applocation :
''' Market basket analysis ,Medical Diagnosis '''
