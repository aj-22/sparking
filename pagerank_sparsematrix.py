data_file=r"C:\Users\ajink\Documents\cs547\hw3-bundle\q2\data\graph-full.txt"

import numpy as np
import scipy.sparse as sps

graph=np.genfromtxt(data_file,dtype=(int,int),delimiter=",")
# First column of graph refers to source
# Second column of graph refers to destination
graph=np.unique(graph,axis=0)
graph[:,0]=graph[:,0]-1
graph[:,1]=graph[:,1]-1

len_graph=len(graph)
data=np.zeros(len_graph)
deg={}

for i in graph:
    if i[0] in deg.keys():
        deg[i[0]]+=1
    else:
        deg[i[0]]=1

i=0
for elem in graph:
    data[i]=1/deg[elem[0]]
    i+=1

M=sps.csc_matrix((data, (graph[:,1], graph[:,0])))
n=len(M.toarray())
beta=0.8

r=np.full(n,1/n).reshape(n,1)
one=np.full(n,1).reshape(n,1)
#r_prime=np.zeros(n).reshape(n,1)

for i in range(40):
    r=((1-beta)/n)*one+beta*M*r

r_index=np.arange(1,n+1)

# Concat two arrays
r_final_prime=np.vstack((r.reshape(1,n), r_index))

r_final=r_final_prime.T

top_5_r=r_final[r_final[:,0].argsort()[::-1]][0:5]
bottom_5_r=r_final[r_final[:,0].argsort()[::1]][0:5]

def print_r_final(r,text):
    for i in range(len(r)):
        print(text+" "+ str(i+1)+"th node id: "+str(int(r[:,1][i])))

print_r_final(top_5_r,"top")
print_r_final(bottom_5_r,"bottom")
