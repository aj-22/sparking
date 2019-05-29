# spark-submit C:\Users\ajink\Documents\cs547\hw3-bundle\hw3_2b_Spark.py

from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)

data_file=r"C:\Users\ajink\Documents\cs547\hw3-bundle\q2\data\graph-full.txt"

import numpy as np
import scipy.sparse as sps

graph=np.genfromtxt(data_file,dtype=(int,int),delimiter="\t")
# First column of graph refers to source
# Second column of graph refers to destination
graph=np.unique(graph,axis=0)
graph[:,0]=graph[:,0]-1
graph[:,1]=graph[:,1]-1

len_graph=len(graph)
data=np.zeros(len_graph)

i=0
for elem in graph:
    data[i]=1
    i+=1

L=sps.csc_matrix((data, (graph[:,0], graph[:,1])))
n=len(L.toarray())
h=np.full(n,1).reshape(n,1)
a=np.full(n,1).reshape(n,1)

for i in range(40):
    a=L.T*h
    h=L*a

def stack_and_sort(arr,rank_num=5,order=True):
    n=len(arr)
    arr_index=np.arange(1,n+1)
    arr_final=np.vstack((arr.reshape(1,n), arr_index))
    arr_final=arr_final.T
    if order==True:
        return arr_final[arr_final[:,0].argsort()[::-1]][0:rank_num]
    else:
        return arr_final[arr_final[:,0].argsort()[::1]][0:rank_num]

def print_stack_and_sort(r,text):
    for i in range(len(r)):
        print(text+" "+ str(i+1)+"th node id: "+str(int(r[:,1][i])))

print_stack_and_sort(stack_and_sort(h),"Hubbiness top")
print()
print_stack_and_sort(stack_and_sort(h,order=False),"Hubbiness bottom")
print()
print_stack_and_sort(stack_and_sort(a),"Authority top")
print()
print_stack_and_sort(stack_and_sort(a,order=False),"Authority bottom")
