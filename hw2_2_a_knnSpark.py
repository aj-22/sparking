# spark-submit C:\Users\ajink\Documents\cs547\hw2-bundle\hw2_2_a_knnSpark.py
'''
# GET RDD OF NUMPY ARRAY
'''

'''
#[array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.]),
# array([ 0. ,  0.1,  0.3,  0.4,  0.5,  0.6,  0.7]),
# array([ 1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7])]
'''
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)

c1 = sc.textFile(r"C:\Users\ajink\Documents\cs547\hw2-bundle\hw2-bundle\q2\data\c1.txt")
c2 = sc.textFile(r"C:\Users\ajink\Documents\cs547\hw2-bundle\hw2-bundle\q2\data\c2.txt")
data = sc.textFile(r"C:\Users\ajink\Documents\cs547\hw2-bundle\hw2-bundle\q2\data\data.txt")

data_np=data.map(lambda l:[float(i) for i in l.split(" ")]).map(np.array)
c1_np=c1.map(lambda l:[float(i) for i in l.split(" ")]).map(np.array).collect()
c2_np=c2.map(lambda l:[float(i) for i in l.split(" ")]).map(np.array).collect()

def getEucDist(point1,point2):
    return np.linalg.norm(point1-point2)

def getManDist(point1,point2):
    return np.linalg.norm(point1-point2,1)

def p(x):
    print(x)

def get_min_distance(point,c1,distFunction):
    dist=distFunction(point,c1[0])
    index=0
    poi=0
    for i in c1:
        if dist > distFunction(point,i):
            poi=index
            dist=distFunction(point,i)
        index+=1
    return (dist,c1[poi])

def getMean(np_points):
     return np.mean(np_points, axis = 0)

def plotCost(cost_c1_euc,cost_c2_euc,cost_c1_man,cost_c2_man):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    iter_values=np.arange(0,20)
    axs[0,0].plot(iter_values,cost_c1_euc)
    axs[0,0].set_title("c1 and Euclidean Intialization")
    axs[0,0].set_xlabel("Iterations")
    axs[0,0].set_ylabel("Cost")
    axs[0,1].plot(iter_values,cost_c2_euc)
    axs[0,1].set_title("c2 and Euclidean Intialization")
    axs[0,1].set_xlabel("Iterations")
    axs[0,1].set_ylabel("Cost")
    axs[1,0].plot(iter_values,cost_c1_man)
    axs[1,0].set_title("c1 and Manhattan Intialization")
    axs[1,0].set_xlabel("Iterations")
    axs[1,0].set_ylabel("Cost")
    axs[1,1].plot(iter_values,cost_c2_man)
    axs[1,1].set_title("c2 and Manhattan Intialization")
    axs[1,1].set_xlabel("Iterations")
    axs[1,1].set_ylabel("Cost")
    plt.savefig("k-Means Clustering.png")
    plt.show()

MAX_ITER=20

# Euclidean Intialization for c1
cost_c1_euc=np.zeros(MAX_ITER)
for i in range(0,MAX_ITER):
    data_np1=data_np.map(lambda l:(l,get_min_distance(l,c1_np,getEucDist)))
    cost_c1_euc[i]=data_np1.map(lambda l:l[1][0]).sum()
    data_np2=data_np1.map(lambda l:(tuple(l[1][1]),l[0])).groupByKey().mapValues(list).mapValues(np.array).map(lambda l:(l[0],getMean(l[1])))
    c1_np=data_np2.map(lambda l:l[1]).map(np.array).collect()

# Euclidean Intialization for c2
cost_c2_euc=np.zeros(MAX_ITER)
for i in range(0,MAX_ITER):
    data_np1=data_np.map(lambda l:(l,get_min_distance(l,c2_np,getEucDist)))
    cost_c2_euc[i]=data_np1.map(lambda l:l[1][0]).sum()
    data_np2=data_np1.map(lambda l:(tuple(l[1][1]),l[0])).groupByKey().mapValues(list).mapValues(np.array).map(lambda l:(l[0],getMean(l[1])))
    c2_np=data_np2.map(lambda l:l[1]).map(np.array).collect()

c1_np=c1.map(lambda l:[float(i) for i in l.split(" ")]).map(np.array).collect()
c2_np=c2.map(lambda l:[float(i) for i in l.split(" ")]).map(np.array).collect()

# Manhattan Intialization for c1
cost_c1_man=np.zeros(MAX_ITER)
for i in range(0,MAX_ITER):
    data_np1=data_np.map(lambda l:(l,get_min_distance(l,c1_np,getManDist)))
    cost_c1_man[i]=data_np1.map(lambda l:l[1][0]).sum()
    data_np2=data_np1.map(lambda l:(tuple(l[1][1]),l[0])).groupByKey().mapValues(list).mapValues(np.array).map(lambda l:(l[0],getMean(l[1])))
    c1_np=data_np2.map(lambda l:l[1]).map(np.array).collect()

# Manhattan Intialization for c2
cost_c2_man=np.zeros(MAX_ITER)
for i in range(0,MAX_ITER):
    data_np1=data_np.map(lambda l:(l,get_min_distance(l,c2_np,getManDist)))
    cost_c2_man[i]=data_np1.map(lambda l:l[1][0]).sum()
    data_np2=data_np1.map(lambda l:(tuple(l[1][1]),l[0])).groupByKey().mapValues(list).mapValues(np.array).map(lambda l:(l[0],getMean(l[1])))
    c2_np=data_np2.map(lambda l:l[1]).map(np.array).collect()

plotCost(cost_c1_euc,cost_c2_euc,cost_c1_man,cost_c2_man)

print("For Euclidean Initialization:")
print("Cost of c1 after 1 iteration is "+ str(cost_c1_euc[0]) + " and after 10 iterations" + str(cost_c1_euc[9]))
print("Cost of c2 after 1 iteration is "+ str(cost_c2_euc[0]) + " and after 10 iterations" + str(cost_c2_euc[9]))

print("For Manhattan Initialization:")
print("Cost of c1 after 1 iteration is "+ str(cost_c1_man[0]) + " and after 10 iterations" + str(cost_c1_man[9]))
print("Cost of c2 after 1 iteration is "+ str(cost_c2_man[0]) + " and after 10 iterations" + str(cost_c2_man[9]))

'''End of code'''
