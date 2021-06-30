from pyspark import SparkContext, SparkConf
import math

Num_User=610

def read_data(line):
    inputs = line.split('\t')
    return [inputs[0],inputs[1],float(inputs[2])]

def pearson_correlation_coefficient(user):
    global user_item_map
    user_x=user
    user_x_item=user_item_map[user_x]
    similarity=[]
    for user_y in range(1,Num_User+1):
        if user_x == str(user_y):
            continue
        user_y_item=user_item_map[str(user_y)]
        XY=0
        YY=0
        XX=0
        divisor=0
        same_item=0
        for item_x in user_x_item:
            for item_y in user_y_item:
                if item_x[0] == item_y[0]:
                    same_item+=1
                    XY+=(item_x[1]*item_y[1])
                    XX+=item_x[1]**2
                    YY+=item_y[1]**2
        divisor=math.sqrt(XX)*math.sqrt(YY)
        if divisor==0:
            continue
        simXY=XY/divisor
        similarity.append((user_x,user_y,simXY,same_item))
    
    return similarity

conf = SparkConf().setMaster("local").setAppName("Term_project")
sc = SparkContext(conf = conf)

print('preprocessing...')
data = sc.textFile("data.txt").map(read_data)

#calculate each avg rating of user ( sum of rat )/( #_items rated by user)
num_item_user = data.map(lambda x: (x[0],1)).reduceByKey(lambda x,y :x+y)
sum_user_rate = data.map(lambda x: (x[0],x[2])).reduceByKey(lambda x,y:x+y)
avg_user_rate = sum_user_rate.join(num_item_user).mapValues(lambda x: x[0]/x[1])

#calculate (r.xs - avg.x)  r.xs : item s rated by user x , avg.x : avg rating of user x  
user_item_list=data.map(lambda x : [x[0],(x[1],x[2])]).join(avg_user_rate).mapValues(lambda x: (x[0][0],x[0][1]-x[1])).groupByKey().mapValues(list)
user_item_map=user_item_list.collectAsMap()

user=input('target user:')

#pearson correlation coefficient
print('user-user similarity by pearson correlation coefficient')
user_similarity_list=sorted(pearson_correlation_coefficient(user),key=lambda x: x[2],reverse=True)

fp = open("user_similarity.txt", "w")
for element in user_similarity_list:
    fp.writelines(str(element[0]) + '\t\t' + str(element[1]) + '\t\t' + str(element[2]) + '\t\t' + str(element[3]) +'\n')
print('user-user similarity done')

print('recommand movies to user',user)

#user_similarity_list=user_similarity_list[0:10]
high_similarity=[]
for element in user_similarity_list:
    if element[2]>=0.8 and element[3]>1:
        high_similarity.append(element)

high_similarity=sorted(high_similarity,key=lambda x: x[2]*x[3],reverse=True)
#print(high_similarity[0:10])

num_top_similarity_user=10
if len(high_similarity)< num_top_similarity_user:
    num_top_similarity_user=len(high_similarity)
movie_recommand=[]
for i in range(0,num_top_similarity_user):
    similar_user=high_similarity[i][1]
    for j in user_item_map[str(similar_user)]:
        movie_recommand.append(j)

target_user_item=sc.parallelize(user_item_map[str(user)])
movie_recommand=sc.parallelize(movie_recommand).subtract(target_user_item).reduceByKey(lambda x,y:x+y).mapValues(lambda x: x/num_top_similarity_user).sortBy(lambda x : x[1],ascending=False)
recommand_list=movie_recommand.collect()[0:3]

fp = open("recommand moives list.txt", "w")
for movie in recommand_list:
    fp.writelines(str(movie[0]) +'\n')
print('recommand moives done')