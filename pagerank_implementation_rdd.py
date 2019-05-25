#spark-submit C:\Users\ajink\Documents\cs547\hw3-bundle\hw3_2a_Spark_RDD.py

import sys
import numpy as np
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)


import numpy as np
data_file=r"C:\Users\ajink\Documents\cs547\hw3-bundle\q2\data\graph-full.txt"
graph_raw = sc.textFile(data_file)

graph=graph_raw.distinct().map(lambda l:[int(i) for i in l.split("\t")]).map(tuple).map(lambda l:(l[0]-1,l[1]-1))

n=max( graph.map(lambda l:l[0]).max(), graph.map(lambda l:l[1]).max())+1

deg=graph.map(lambda l:(l[0],1)).reduceByKey(lambda l1,l2:l1+l2).collectAsMap()

# First column of graph refers to source
# Second column of graph refers to destination

M=graph.map(lambda l:(l[1],l[0])).groupByKey().mapValues(list).map(lambda l:(l[0],l[1],[1/deg[i] for i in l[1] ])).sortBy(lambda l:l[0])
beta=0.8
r=np.full(n,1/n)
one=(1-beta)/n*np.full(n,1)

def Mrow_dot_r(M_row,r):
    S=0
    c=0
    S=np.dot(M_row[2],[r[i] for i in M_row[1]])
    return S

## (2, [1, 85, 66, 11, 95, 25, 82], 0.07142857142857142)


for i in range(40):
    pg2=np.array(M.map(lambda l:Mrow_dot_r(l,r)).collect())
    pg1=one
    r=one+beta*pg2


r_index=np.arange(1,n+1)
r_final_prime=np.vstack((r, r_index))
r_final=r_final_prime.T
top_5_r=r_final[r_final[:,0].argsort()[::-1]][0:5]
bottom_5_r=r_final[r_final[:,0].argsort()[::1]][0:5]

def print_r_final(r,text):
    for i in range(len(r)):
        print(text+" "+ str(i+1)+"th node id: "+str(int(r[:,1][i])))

print_r_final(top_5_r,"top")
print_r_final(bottom_5_r,"bottom")
