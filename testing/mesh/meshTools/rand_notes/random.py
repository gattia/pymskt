import numpy as np
order = 2 
max_dist = 5
list_distances = [0, 1, 2, 3, 4, 5]
list_scalars = [10, 20, 30, 40, 50, 60]

distances_weighted = (max_dist - np.asarray(list_distances))**order 
scalars_weighted = distances_weighted * np.asarray(list_scalars)

normalized_value = np.sum(scalars_weighted) / np.sum(distances_weighted)

print(normalized_value)