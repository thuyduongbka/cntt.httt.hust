import sys
 
from pyspark import SparkContext, SparkConf


def print_kmer(a, k):
    x = len(a)
    i = 0
    for i in range(x):
        if (i + k) > x:
            break
        print(a[i:i+k])

def find_kmer(a, k):
    list_kmer = []
    x = len(a)
    i = 0
    for i in range(x):
        if (i + k) > x:
            break
        list_kmer.append(a[i:i+k])
    return list_kmer

def count_kmer(a,k):
    # Start by making counts an empty dictionary, using {}
    counts = {}
    big_list = find_kmer(a, k)
    # Now loop through the values in big_list and count them
    for num in big_list:
        
        # Check to see if this key is already in the dictionary
        # If not, add it with an initial count of zero
        if not (num in counts):
            counts[num] = 0
        
        # Now that we are sure the key is in the dictionary, we can increment the count
        counts[num] += 1

    # After the loop is finished, counts should contain the right counts for each number seen
    return counts

 
if __name__ == "__main__":
	
	# create Spark context with necessary configuration
	sc = SparkContext("local","Kmer")
	
	# read data from text file and split each line into words
	words = sc.textFile("ecoli.fa").flatMap(lambda line: find_kmer(line, 9))
	
	# count the occurrence of each word
	wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)

	# save the counts to output
	wordCounts.saveAsTextFile("output_kmer/")
