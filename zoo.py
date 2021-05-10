import pandas as pd
import os

def import_data():
    return  pd.read_csv(os.getcwd() + "/data.csv", header=None)

#calculates distance between 2 entries in data_values
def calculate_distance(animal1, animal2):
    distance = 0
    for i in range(1, 17):
        if animal1[i] != animal2[i]:
            distance += 1
    return distance  

def complete_linkage(cluster1, cluster2, data_values):
    max = -1
    for i in cluster1:
        for j in cluster2:
            distance = calculate_distance(data_values[i], data_values[j])
            if distance > max:
                max = distance
    return max

def single_linkage(cluster1, cluster2, data_values):
    min = 10000
    for i in cluster1:
        for j in cluster2:
            distance = calculate_distance(data_values[i], data_values[j])
            if distance < min:
                min = distance
    return min

def average_linkage(cluster1, cluster2, data_values):
    sum = 0
    for i in cluster1:
        for j in cluster2:
            sum += calculate_distance(data_values[i], data_values[j])
    return sum/len(cluster1)*len(cluster2)

#calculates distance_matrix based on the chosen linkage function
def get_distance_matrix(data_values, clusters, linkage):
    distance_matrix = []
    for i in clusters:
        array = []
        for j in clusters:
            if linkage == "complete":
                distance = complete_linkage(i, j, data_values)
            elif linkage == "single":
                distance = single_linkage(i, j, data_values)
            elif linkage == "average":
                distance = average_linkage(i, j, data_values)
            array.append(distance)
        distance_matrix.append(array)
    return distance_matrix

#clusters have lists of the indexes of the data_values
def create_clusters(data): 
    clusters = []
    for i in range(0, len(data)):
        clusters.append([i])
    return clusters

#finds the closest clusters, min_i, min_j are the indexes in the clusters list
#min is the distance
def minimum_distance(distance_matrix):
    min = 10000
    min_i = -1 
    min_j = -1
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j and distance_matrix[i][j] < min:
                min = distance_matrix[i][j]
                min_i = i
                min_j = j
    return (min, min_i, min_j)

#[[1], [2]] -> [[1, 2]]
def combine_clusters(clusters, i, j):
    clusters[i].extend(clusters[j])
    clusters.remove(clusters[j])

def AGNES(linkage, max_clusters, data_values):
    clusters = create_clusters(data_values)
    distance_matrix = get_distance_matrix(data_values, clusters, linkage)
    cluster_count = len(clusters)
    while cluster_count > max_clusters:
        distance, i, j = minimum_distance(distance_matrix)
        combine_clusters(clusters, i, j)
        distance_matrix = get_distance_matrix(data_values, clusters, linkage)
        cluster_count -= 1
    return clusters

#prints resulting clusters and calculates misclasification percent
def print_results(clusters):
    print(f"linkage: {linkage}, max clusters: {max_clusters}")
    total_accuracy = 0
    for i in clusters:  
        animals = []
        types = {}
        for j in range(0, max_clusters):
            types[j] = 0
        for j in i:
            animals.append((data_values[j][0], data_values[j][17]))
            types[data_values[j][17] - 1] += 1
        
        max_type = -1
        max_type_i = -1
        for j in range(0, len(types)):
            if types[j] > max_type:
                max_type = types[j]
                max_type_i = j
        accuracy = types[max_type_i]/len(animals)*100
        total_accuracy = accuracy
        print()
        print(f"{animals} - highest: {accuracy}% of {max_type_i + 1}" )
    print()
    print(f"misclassification percent: {total_accuracy/max_clusters}%")

data = import_data()
data_values = data.values

linkage = "complete"
max_clusters = 7

clusters = AGNES(linkage, max_clusters, data_values)
print_results(clusters)