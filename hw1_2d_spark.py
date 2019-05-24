#spark-submit C:\Users\ajink\Documents\cs547\hw1-bundle\hw1_2d_spark.py C:\Users\ajink\Documents\cs547\hw1-bundle\q2\data\browsing.txt C:\Users\ajink\Documents\cs547\hw1-bundle\q2\output\3
import re
import sys
import itertools
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
#lines = sc.textFile("C:\\Users\\ajink\\Documents\\cs547\\hw1-bundle\\q2\\data\\browsing.txt")
lines = sc.textFile(sys.argv[1])
lines = lines.map(lambda l:l.strip())

## Count number of items for Pass 1
words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))
pairs = words.map(lambda w: (w, 1))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
dictCounts = counts.collectAsMap()

## Get RDD of baskets
rdd_BasketList = lines.map(lambda l: l.split(" "))

supportThreshold=100

## Remove element from basket if it fails Pass 1
def firstPass(l):
    for i in l:
        if dictCounts[i]<supportThreshold:
            l.remove(i)
    return l

rdd_BasketList_pass1=rdd_BasketList.map(lambda l:firstPass(l))


#print("COUNT AFTER FIRST PASS "+str(rdd_BasketList_pass1.flatMap(lambda l:l).distinct().count()))

## Generate a list of ordered item pairs
def pairsFromList(list):
    returnList = []
    for i in range(len(list)):
        for j in range(i,len(list)):
            if(i!=j):
                if(list[i]<list[j]):
                    returnList.append((list[i],list[j]))
                else:
                    returnList.append((list[j],list[i]))
    return returnList

## Rdd of (items pairs,1) after pass 1
rdd_BasketTuples=rdd_BasketList_pass1.map(lambda l:pairsFromList(l))
rdd_BasketTuple_kv=rdd_BasketTuples.flatMap(lambda l:l).map(lambda l:(l,1))

## Sum of pairs/ number of baskets containing pairs
rdd_BasketTuple_ksum=rdd_BasketTuple_kv.reduceByKey(lambda l1,l2:l1+l2)

## Filter out pairs which do not reach support threshold
rdd_BasketTuple_ksum_pass2=rdd_BasketTuple_ksum.filter(lambda l:l[1]>=supportThreshold)

## dictionary of Basket of Tuples for easy lookup
dict_BasketTuple_ksum_pass2 = rdd_BasketTuple_ksum_pass2.collectAsMap()

## remove pairs which fail pass 2 from rdd_BasketTuples
def secondPass(l):
    for i in l:
        if i not in dict_BasketTuple_ksum_pass2:
            l.remove(i)
        else:
            if dict_BasketTuple_ksum_pass2[i]<supportThreshold:
                l.remove(i)
    return l

rdd_BasketTuples_filtered = rdd_BasketTuples.map(lambda l:secondPass(l))

## Pair Confidence
rdd_PairConfidence = rdd_BasketTuple_ksum_pass2.map(lambda l: [((l[0],l[0][0]),l[1]/dictCounts[l[0][0]]), ((l[0],l[0][1]),l[1]/dictCounts[l[0][1]])]).flatMap(lambda l:l).sortBy(lambda l:(l[1],l[0][0]),False)

print("Top 5 Association Rules for Pairs")
print("((A,B),A) == conf(A->B)")
print(rdd_PairConfidence.take(5))
#rdd_PairConfidence.saveAsTextFile(sys.argv[2])


def tupleToSingles(lst):
    retSet=set()
    for i in lst:
        retSet.add(i[0])
        retSet.add(i[1])
    return list(retSet)

rdd_BasketList_pass2 = rdd_BasketTuples_filtered.map(lambda l:tupleToSingles(l))

## Function to get triples of items after pass1
def triplesFromList(list):
    returnList = []
    for i in range(len(list)):
        for j in range(i,len(list)):
            if(i!=j):
                for k in range(j,len(list)):
                    if(j!=k):
                        returnList.append(tuple(sorted((list[k],list[j],list[i]))))
    return returnList

def tripletsFromPairs(tupList):
    doubles=tupList
    keys = set([x for double in doubles for x in double])
    options = itertools.combinations(keys, 3)
    triples = []
    for option in options:
        x, y, z = sorted(option)
        first, second, third = (x, y), (x, z), (y, z)
        if first in doubles and second in doubles and third in doubles:
            triples.append(tuple(sorted(option)))
    return triples

rdd_BasketTriples=rdd_BasketTuples_filtered.map(lambda l:tripletsFromPairs(l))

## Convert triples into key values
##rdd_BasketTriple_kv=rdd_BasketList_pass2.flatMap(lambda l:triplesFromList(l)).map(lambda l:(l,1))

rdd_BasketTriple_kv=rdd_BasketTriples.flatMap(lambda l:l).map(lambda l:(l,1))

## Get Support of triples and filter those with less than 100 support
rdd_BasketTriple_ksum=rdd_BasketTriple_kv.reduceByKey(lambda l1,l2:l1+l2).filter(lambda l:l[1]>=supportThreshold)

## Get Confidence of triples
rdd_TripleConfidence = rdd_BasketTriple_ksum.map(lambda l: [ ((l[0],(l[0][0],l[0][1])),l[1]/dict_BasketTuple_ksum_pass2[(l[0][0],l[0][1])]), ((l[0],(l[0][0],l[0][2])),l[1]/dict_BasketTuple_ksum_pass2[(l[0][0],l[0][2])]),((l[0],(l[0][1],l[0][2])),l[1]/dict_BasketTuple_ksum_pass2[(l[0][1],l[0][2])])]).flatMap(lambda l:l)

rdd_TripleConfidence_sorted=rdd_TripleConfidence.sortBy(lambda l:(l[1],l[0][1][0],l[0][1][1]),False)
rdd_TripleConfidence_sorted.persist()
print("Top 5 Association Rules for Triples")
print("((A,B,C),(A,B)) == conf({A,B}->C)")
print(rdd_TripleConfidence_sorted.take(5))

sc.stop()
