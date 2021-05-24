import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

def import_data():
    return  pd.read_csv(os.getcwd() + "/data.csv", header=None)

#calculates distance between 2 entries in data_values
def calculate_distance(animal1, animal2):
    distance = 0
    for i in range(0, len(animal1)):
        if animal1[i] != animal2[i]:
            distance += 1
    return distance  

def complete_linkage(cluster1, cluster2, data_values):
    max = -1
    for i in cluster1[1:]:
        for j in cluster2[1:]:
            distance = calculate_distance(data_values[i], data_values[j])
            if distance > max:
                max = distance
    return max

def single_linkage(cluster1, cluster2, data_values):
    min = 10000
    for i in cluster1[1:]:
        for j in cluster2[1:]:
            distance = calculate_distance(data_values[i], data_values[j])
            if distance < min:
                min = distance
    return min

def average_linkage(cluster1, cluster2, data_values):
    sum = 0
    for i in cluster1[1:]:
        for j in cluster2[1:]:
            sum += calculate_distance(data_values[i], data_values[j])
    return sum / ((len(cluster1) - 1) * (len(cluster2) - 1))

#calculates distance_matrix (distances between every 2 clusters) 
#based on the chosen linkage function
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
def create_clusters(data_values): 
    clusters = []
    for i in range(0, len(data_values)):
        clusters.append([i, i])
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

#[[1, 1], [2, 2]] -> [[101, 1, 2]], [3,3] - > [[102, 1, 2, 3]] ...
def combine_clusters(clusters, i, j, last_cluster_index):
    clusters[i].pop(0)
    clusters[j].pop(0)
    clusters[i].extend(clusters[j])
    clusters[i].insert(0, last_cluster_index)
    clusters.remove(clusters[j])

def AGNES(linkage, max_clusters, data_values, stop_at_25 = False):
    clusters = create_clusters(data_values)
    distance_matrix = get_distance_matrix(data_values, clusters, linkage)
    cluster_count = len(clusters)
    last_cluster_index = cluster_count
    Z = []
    while cluster_count > max_clusters:
        distance, i, j = minimum_distance(distance_matrix)
        if stop_at_25 and distance >= len(data_values[0]) / 4:
            break
        Z.append([clusters[i][0], clusters[j][0], float(distance), len(clusters[i]) + len(clusters[j]) - 2])
        combine_clusters(clusters, i, j, last_cluster_index)
        last_cluster_index += 1
        distance_matrix = get_distance_matrix(data_values, clusters, linkage)
        cluster_count -= 1
    return (clusters, Z)

#prints resulting clusters and calculates misclasification percent
def print_results(clusters, stop_at_25):
    if(stop_at_25):
        print(f"linkage: {linkage}, stop at 25% difference: {stop_at_25}")
    else:
        print(f"linkage: {linkage}, max clusters: {len(clusters)}")
    total_accuracy = 0
    for i in clusters:  
        animals = []
        types = [0, 0, 0, 0, 0, 0, 0]
        for j in i[1:]:
            animals.append((data.values[j][0], data.values[j][len(data.values[j]) - 1]))
            types[data.values[j][len(data.values[j]) - 1] - 1] += 1   
        max_type = -1
        max_type_index = -1
        for j in range(0, 7):
            if types[j] > max_type:
                max_type = types[j]
                max_type_index = j
        accuracy = types[max_type_index]/len(animals)*100
        total_accuracy += accuracy
        print()
        print(f"{animals} - highest: {accuracy}% of {max_type_index + 1}" )
    print()
    print(f"clusters: {len(clusters)}")
    print(f"misclassification percent: {100 - (total_accuracy / len(clusters))}%")

#remove first and last element in each row in input data (the name and the type)
def trim_input_data(data):
    data_values = data.values
    trimmed_data_values = []
    for i in range(0, len(data_values)):
        trimmed = data_values[i][1:-1]
        trimmed_data_values.append(trimmed)
    return trimmed_data_values

data = import_data()
data_values = trim_input_data(data)

linkage = "complete"
max_clusters = 1
#Algorithm should stop when there is more then 25% difference between the distances 
#in all of the remaining clusters
stop_at_25 = False

clusters, Z = AGNES(linkage, max_clusters, data_values, stop_at_25)

print_results(clusters, stop_at_25)

#only draws diagrams when max_clusters = 1 and does not stop prematurely
label_list = []
for i in range(0, len(data.values)):
    label_list.append(data.values[i][0])
hierarchy.dendrogram(Z, labels=label_list)
plt.show()

