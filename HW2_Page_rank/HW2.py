from pyspark import SparkConf, SparkContext

NUM_NODES = 10876
EPSILON = 1e-12 # 𝜀
B = 0.8
MAX_iteration=30

def readfile(line):
    wordlist = line.split("\t")
    maplist = []
    for item in wordlist:
        maplist.append(item)
    return (maplist[0],maplist[1])

def init_r(x):
    return(x,1/NUM_NODES)

def deg(line):
    wordlist = line.split("\t")
    maplist = []
    for item in wordlist:
        maplist.append(item)
    return (maplist[0],1)

def reducer(x,y):
    return x+y

conf = SparkConf().setMaster("local").setAppName("MDA_HW2")
sc = SparkContext(conf=conf)
#read file
Input_data=sc.textFile("p2p-Gnutella04.txt").filter(lambda x : x[0]!='#')
# edge(vi,vj) in graph
Graph=Input_data.map(readfile)
# each vertex's out degree => Di
Degree=Input_data.map(deg).reduceByKey(lambda x,y : x+y)
#initial PageRank 
node_array=[]
for i in range(0,NUM_NODES):
    node_array.append(str(i))
init_PR = sc.parallelize(node_array).map(init_r)

old_PR=init_PR

#print(init_PR.collect())
for t in range(MAX_iteration):
    # Ri/Di
    RD= old_PR.join(Degree).mapValues(lambda x: x[0]/x[1])
    # R'j 
    pre_Rj=Graph.join(RD).map(lambda x : (x[1][0],x[1][1])).reduceByKey(lambda x,y : x+y)
    pre_Rj=pre_Rj.mapValues(lambda x: B*x+((1-B)/NUM_NODES))
    # re-insert the leaked PageRank: Rj=pre_Rj + (1-S)/N
    S=pre_Rj.values().sum()
    new_PR= pre_Rj.mapValues(lambda x : x+((1-S)/NUM_NODES)).sortByKey(True)
    #compute the diffence between newPR and old one
    EP=new_PR.join(old_PR).mapValues(lambda x: abs(x[0]-x[1])).values().sum()
    
    print("Iteration",t+1,": 𝜀 =",EP)
    if EP<=EPSILON:
          break
    #this new_PR is the old one in next interation
    old_PR=new_PR
    
result=new_PR.sortBy(lambda x:x[1],ascending= False).collect()

fp = open("output.txt", "w")
for i in range(10):
    line=[str(result[i][0])+" ",str(result[i][1])+"\n"]
    fp.writelines(line)
fp.close()
sc.stop()