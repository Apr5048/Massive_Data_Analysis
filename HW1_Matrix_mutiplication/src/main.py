from pyspark import SparkConf, SparkContext

def M_mapper(line):
    wordlist = line.split(",")
    maplist = []
    for item in wordlist:
        maplist.append(item)
    return (int(maplist[2]), (int(maplist[1]), int(maplist[3])))
def N_mapper(line):
    wordlist = line.split(",")
    maplist = []
    for item in wordlist:
        maplist.append(item)
    return (int(maplist[1]), (int(maplist[2]), int(maplist[3])))

def reducer(x,y):
    return x+y

conf = SparkConf().setMaster("local").setAppName("MDA_HW1")
sc = SparkContext(conf=conf)
inputs = sc.textFile("500input.txt")

M = inputs.filter(lambda x : x[0]=='M') # Matrix M
N = inputs.filter(lambda x : x[0]=='N') # Matrix N
M_mapped=M.map(M_mapper)  # Mij => (j,(i,Mij))
N_mapped=N.map(N_mapper) # Njk => (j,(k,Njk))
MN = M_mapped.join(N_mapped).map(lambda x: ( (x[1][0][0],x[1][1][0]), x[1][0][1] * x[1][1][1] )) # Join M's and N's pair to (j,((i,Mij),(k,Njk)),and mapping produce a key-value pair with key equal to (i, k) and value equal to the product of these elements, Mij*Njk
P = MN.reduceByKey(reducer) # the result is a pair ((i, k), v), where v is the value of the element in row i and column k of the matrix P = MN
P = P.sortByKey(ascending=True).map(lambda x: (x[0][0],x[0][1],x[1])) # sort the P element by key (row , col), and map P as (rowmcolumn,value) after sorting

#write the MN into output file "output.txt"
fp = open("output.txt", "w")
for output in P.collect():
    line=[str(output[0])+" ",str(output[1])+" ",str(output[2])+"\n"]
    fp.writelines(line)
fp.close()
sc.stop()