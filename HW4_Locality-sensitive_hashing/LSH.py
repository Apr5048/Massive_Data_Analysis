from pyspark import SparkConf, SparkContext
import os
from random import randrange
from itertools import combinations 

#    
# read file 
#
document=[]
document_num=0
def read_file():
    global document,document_num
    for file in sorted(os.listdir("athletics/")):
        if file.endswith(".txt"):
            filename = os.path.join("athletics", file)
            txt = open(filename, 'r')
            txt = txt.read().replace("\n", " ").replace(","," ").replace("."," ").replace('"'," ").split()
            words=sc.parallelize(txt)
            document.append(words)
            document_num += 1

conf = SparkConf().setMaster("local").setAppName("MDA_HW4")
sc = SparkContext(conf=conf)
print("reading files...")
read_file()
#
#  SHINGLING
#
print("Shingling...")
k_shingles=3
shingle_dict=dict()
matrix=[]
shingle_num=0
for id in range(0,document_num):
    temp=[]
    tempdoc=document[id].collect()
    for i in range(len(tempdoc)-k_shingles+1):
        shingle=tempdoc[i:i+k_shingles]
        shingle=' '.join(shingle)
        temp.append(shingle)
        if shingle not in shingle_dict:
            shingle_dict[shingle]=shingle_num
            shingle_num+=1
    matrix.append(temp)

#
#   MIN-HASH
#
print("Min-hash...")
num_hashes=100
a_hash = [randrange(0,shingle_num) for a in range(0, num_hashes)]
b_hash = [randrange(0,shingle_num) for b in range(0, num_hashes)]

def min_hash_function(a, b, sig):
    hashes = [((a * x) + b) % shingle_num for x in sig]
    return min(hashes)

def get_min_hash_row(sig):
    hashes = [min_hash_function(a, b, sig) for a, b in zip(a_hash, b_hash)]
    return hashes

signatures=[]
for id in range(0,document_num):
    each_doc=sc.parallelize(matrix[id])
    each_doc_signature=each_doc.map(lambda x: shingle_dict.get(x))
    min_hash_row = get_min_hash_row(each_doc_signature.collect())
    signatures.append(min_hash_row)

#   
#   LSH
#
print("LSH...")
row=2
buckets_num=10
bucket=[]
for i in range(0,buckets_num):
    bucket.append([])

random_band=randrange(0,100,row)
for id in range(0,document_num):
    hash2bucket=((a_hash[0]*(signatures[id][random_band]+signatures[id][random_band+1]))+b_hash[0])%buckets_num
    bucket[hash2bucket].append(id)

result=[]
for index in range(0,buckets_num):
    if int(len(bucket[index]))>=2:
        for i,j in list(combinations(bucket[index], 2)):
            a=set()
            b=set()
            for k in range(len(matrix[i])):
                a.add(matrix[i][k])
            for k in range(len(matrix[j])):
                b.add(matrix[j][k])
            score=(len(a&b)/len(a|b))*100
            result.append(list([i+1,j+1,score]))
result=sc.parallelize(result).sortBy(lambda x: x[2],ascending= False).collect()

for i in range(0,10):
    print("(",result[i][0],",",result[i][1],"):",round(result[i][2],2),"%")