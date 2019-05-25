#spark-submit C:\Users\ajink\Documents\cs547\hw3-bundle\hw3_2b_Spark_RDD.py
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)


import numpy as np

data_file=r"C:\Users\ajink\Documents\cs547\hw3-bundle\q2\data\graph-full.txt"

graph_raw = sc.textFile(data_file)

graph=graph_raw.distinct().map(lambda l:[int(i) for i in l.split("\t")]).map(tuple).map(lambda l:(l[0]-1,l[1]-1))

n=max( graph.map(lambda l:l[0]).max(), graph.map(lambda l:l[1]).max())+1

# First column of graph refers to source
# Second column of graph refers to destination

L_T=graph.map(lambda l:(l[1],l[0])).groupByKey().mapValues(list).map(lambda l:(l[0],l[1],[1 for i in l[1] ])).sortBy(lambda l:l[0])
L=graph.map(lambda l:(l[0],l[1])).groupByKey().mapValues(list).map(lambda l:(l[0],l[1],[1 for i in l[1] ])).sortBy(lambda l:l[0])

h=np.full(n,1)

def Mrow_dot_r(M_row,col):
    S=np.dot(M_row[2],[col[i] for i in M_row[1]])
    return S

for i in range(40):
    a=np.array(L_T.map(lambda l:Mrow_dot_r(l,h)).collect())
    a=a/max(a)
    h=np.array(L.map(lambda l:Mrow_dot_r(l,a)).collect())
    h=h/max(h)


def stack_and_sort(arr,rank_num=5,order=True):
    n=len(arr)
    arr_index=np.arange(1,n+1)
    arr_final=np.vstack((arr, arr_index))
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
