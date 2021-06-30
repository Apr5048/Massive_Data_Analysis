from pyspark import SparkConf, SparkContext
import math
from operator import add
import matplotlib.pyplot as plt
import csv

k=0
index=0
dim = 58
max_iteration=20
global k_cluster_c1
global k_cluster_c2
global e_k_cluster_c1
global e_k_cluster_c2
global m_k_cluster_c1
global m_k_cluster_c2

global c1_or_c2 # each distance methon star with which txt file

def readpoint(line):
    global index
    wordlist = line.split(" ")
    index+=1
    maplist = []
    for item in wordlist:
        maplist.append(float(item))
    return maplist

def readcluster(line):
    global k
    wordlist = line.split(" ")
    k+=1
    maplist = []
    for item in wordlist:
        maplist.append(float(item))
    return maplist

def Eulidean(points,cluster):
    distance=0
    for i in range(dim):
        distance+=(points[i]-cluster[i])**2
    return distance,math.sqrt(distance)

def Manhattan(point1, cluster):
    distance=0
    for i in range(dim):
        
        distance += abs(point1[i] - cluster[i])
    return distance, distance

def e_assign_cluster_and_cost(line):
    assigned_cost=[]
    dist_2_cluster=[]
    global c1_or_c2
    if c1_or_c2 == 'c1':     
        for k , centroid in enumerate(k_cluster_c1):
            result=Eulidean(line,centroid)
            assigned_cost.append(result[0])
            dist_2_cluster.append(result[1])
    if c1_or_c2 == 'c2':
        for k , centroid in enumerate(k_cluster_c2):
            result=Eulidean(line,centroid)
            assigned_cost.append(result[0])
            dist_2_cluster.append(result[1])
    cost=min(assigned_cost)    
    index=dist_2_cluster.index(min(dist_2_cluster))
    return (index,(cost,line))

def m_assign_cluster_and_cost(line):
    assigned_cost=[]
    dist_2_cluster=[]
    global c1_or_c2
    if c1_or_c2 == 'c1':
        for k , centroid in enumerate(k_cluster_c1):
            result=Manhattan(line,centroid)
            assigned_cost.append(result[0])
            dist_2_cluster.append(result[1])
    if c1_or_c2 == 'c2':
        for k , centroid in enumerate(k_cluster_c2):
            result=Manhattan(line,centroid)
            assigned_cost.append(result[0])
            dist_2_cluster.append(result[1])
    cost=min(assigned_cost)    
    index=dist_2_cluster.index(min(dist_2_cluster))
    return (index,(cost,line))

def update_centroid(line):
    new_centroid=[]
    for i in range(dim):
        new_centroid.append(line[1][i]/line[2])
    return new_centroid

def e_cost_png(e_c1_cost,e_c2_cost):
    x = ['{}'.format(i) for i in range(1,max_iteration+1)]
    plt.figure(0)
    plt.title('Euildean')
    plt.xlabel('interation')
    plt.ylabel('cost')
    plt.plot(x, e_c1_cost, label='c1')
    plt.plot(x, e_c2_cost, label='c2')
    plt.legend(loc='upper right')
    plt.savefig('Euildean.png')
    plt.close()
def m_cost_png(m_c1_cost,m_c2_cost):
    x = ['{}'.format(i) for i in range(1,max_iteration+1)]
    plt.figure(0)
    plt.title('Manhattan')
    plt.xlabel('interation')
    plt.ylabel('cost')
    plt.plot(x, m_c1_cost, label='c1')
    plt.plot(x, m_c2_cost, label='c2')
    plt.legend(loc='upper right')
    plt.savefig('Manhattan.png')
    plt.close()


conf = SparkConf().setMaster("local").setAppName("MDA_HW3")
sc = SparkContext(conf=conf)

document=sc.textFile("data.txt").map(readpoint)
k_cluster_c1=sc.textFile("c1.txt").map(readcluster).collect()
initial_k_cluster_c1=k_cluster_c1
k_cluster_c2=sc.textFile("c2.txt").map(readcluster).collect()
initial_k_cluster_c2=k_cluster_c2

model=['e','m']
initial_centroid=['c1','c2']
e_c1_iteration_cost=[]
e_c2_iteration_cost=[]
m_c1_iteration_cost=[]
m_c2_iteration_cost=[]
improvement_value=[]
for m in model:
    if m=='e' :
        print("Eulidean Distance")
        for txtfile in initial_centroid:
            if txtfile == 'c1':
                c1_or_c2 = 'c1'
                print("initial centroid form",c1_or_c2)
                for i in range(max_iteration):
                    print("iteration",i+1," ")
                    #assign each documnent to it's closest cluster 
                    assigned_document = document.map(e_assign_cluster_and_cost).mapValues(lambda x: (x[0],x[1],1)).reduceByKey(lambda x,y: ( x[0]+y[0] , list(map(add,x[1],y[1])) ,x[2]+y[2]))
                    #compute iteration i's cost
                    cost=sum(assigned_document.map(lambda x :x[1][0]).collect())
                    e_c1_iteration_cost.append(cost)
                    #update centroid form assigned documentc
                    k_cluster_c1=assigned_document.mapValues(update_centroid).values().collect() 
                
                e_k_cluster_c1 = k_cluster_c1
            if txtfile == 'c2':
                c1_or_c2 = 'c2'
                print("initial centroid form",c1_or_c2)
                for i in range(max_iteration):
                    print("iteration",i+1," ")
                    #assign each documnent to it's closest cluster 
                    assigned_document = document.map(e_assign_cluster_and_cost).mapValues(lambda x: (x[0],x[1],1)).reduceByKey(lambda x,y: ( x[0]+y[0] , list(map(add,x[1],y[1])) ,x[2]+y[2]))
                    #compute iteration i's cost
                    cost=sum(assigned_document.map(lambda x :x[1][0]).collect())
                    e_c2_iteration_cost.append(cost)
                    #update centroid form assigned documentc
                    k_cluster_c2=assigned_document.mapValues(update_centroid).values().collect()
                
                e_k_cluster_c2 = k_cluster_c2
    if m=='m':
        k_cluster_c1=initial_k_cluster_c1
        k_cluster_c2=initial_k_cluster_c2
        print("Manhattan Distance")
        for txtfile in initial_centroid:
            if txtfile == 'c1':
                c1_or_c2 = 'c1'
                print("initial centroid form",c1_or_c2)
                for i in range(max_iteration):
                    print("iteration",i+1," ")
                    #assign each documnent to it's closest cluster 
                    assigned_document = document.map(m_assign_cluster_and_cost).mapValues(lambda x: (x[0],x[1],1)).reduceByKey(lambda x,y: ( x[0]+y[0] , list(map(add,x[1],y[1])) ,x[2]+y[2]))
                    #compute iteration i's cost
                    cost=sum(assigned_document.map(lambda x :x[1][0]).collect())
                    m_c1_iteration_cost.append(cost)
                    #update centroid form assigned documentc
                    k_cluster_c1=assigned_document.mapValues(update_centroid).values().collect()
                
                m_k_cluster_c1 = k_cluster_c1   
            if txtfile == 'c2':
                c1_or_c2 = 'c2'
                print("initial centroid form",c1_or_c2)
                for i in range(max_iteration):
                    print("iteration",i+1," ")
                    #assign each documnent to it's closest cluster 
                    assigned_document = document.map(m_assign_cluster_and_cost).mapValues(lambda x: (x[0],x[1],1)).reduceByKey(lambda x,y: ( x[0]+y[0] , list(map(add,x[1],y[1])) ,x[2]+y[2]))
                    #compute iteration i's cost
                    cost=sum(assigned_document.map(lambda x :x[1][0]).collect())
                    m_c2_iteration_cost.append(cost)
                    #update centroid form assigned documentc
                    k_cluster_c2=assigned_document.mapValues(update_centroid).values().collect()
                m_k_cluster_c2 = k_cluster_c2
    
e_cost_png(e_c1_iteration_cost,e_c2_iteration_cost)
m_cost_png(m_c1_iteration_cost,m_c2_iteration_cost)

improvement_value.append(abs(e_c1_iteration_cost[19]-e_c1_iteration_cost[0])*100/e_c1_iteration_cost[0])
improvement_value.append(abs(e_c2_iteration_cost[19]-e_c2_iteration_cost[0])*100/e_c2_iteration_cost[0])
improvement_value.append(abs(m_c1_iteration_cost[19]-m_c1_iteration_cost[0])*100/m_c1_iteration_cost[0])
improvement_value.append(abs(m_c2_iteration_cost[19]-m_c2_iteration_cost[0])*100/m_c2_iteration_cost[0])
print(improvement_value)
with open('Eulidean_c1_with_EulideanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Eulidean(e_k_cluster_c1[i],e_k_cluster_c1[j])[1])
        writer.writerow(dist)
    csvfile.close()
with open('Eulidean_c2_with_EulideanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Eulidean(e_k_cluster_c2[i],e_k_cluster_c2[j])[1])
        writer.writerow(dist)
    csvfile.close()
with open('Eulidean_c1_with_ManhattanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Manhattan(e_k_cluster_c1[i],e_k_cluster_c1[j])[1])
        writer.writerow(dist)
    csvfile.close()
with open('Eulidean_c2_with_ManhattanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Manhattan(e_k_cluster_c2[i],e_k_cluster_c2[j])[1])
        writer.writerow(dist)
    csvfile.close()
with open('Mahattan_c1_with_EulideanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Eulidean(m_k_cluster_c1[i],m_k_cluster_c1[j])[1])
        writer.writerow(dist)
    csvfile.close()

with open('Mahattan_c2_with_EulideanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Eulidean(m_k_cluster_c2[i],m_k_cluster_c2[j])[1])
        writer.writerow(dist)
    csvfile.close()

with open('Mahattan_c1_with_MahattanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
            dist.append(' ')
        for j in range(i,10):
            dist.append(Manhattan(m_k_cluster_c1[i],m_k_cluster_c1[j])[1])
        writer.writerow(dist)
    csvfile.close()

with open('Mahattan_c2_with_MahattanDist.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(10):
        dist=[]
        for k in range(i):
                dist.append(' ')
        for j in range(i,10):
            dist.append(Manhattan(m_k_cluster_c2[i],m_k_cluster_c2[j])[1])
        writer.writerow(dist)
    csvfile.close()