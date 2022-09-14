# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import numpy as np
import time
from scipy.spatial import distance_matrix
from cvrp.constrained_k_means import CKMeans
import matplotlib.pyplot as plt


_colors = ['r', 'g', 'b', 'm', 'y', 'k', 'c'] * 10

def cluster_ckmeans():
    print('_' * 50)
    clusterer = CKMeans(k=nc, max_iterations=500)
    t0 = time.perf_counter()
    assignment = clusterer.predict(locations, demand, capacity)
    t = time.perf_counter() - t0
    capacity_lhs = np.zeros_like(capacity)
    for k, v in enumerate(assignment):
        v = int(v)
        capacity_lhs[v] += demand[k]
    print(f"Violations: {get_violations(capacity_lhs, capacity)}")
    print(f"Score: {get_score(assignment, locations)}")
    print(f"Time: {t:.2e}")
    for i, (x, y) in enumerate(locations):
        plt.scatter(x, y, demand[i] * 3, color=_colors[int(assignment[i])])


def get_violations(capacity_lhs, capacity):
    diff = capacity_lhs - capacity
    diff = diff[diff > 0]
    if len(diff) > 0:
        return np.sum(diff)
    return 0


def get_score(assignments, locations):
    assignments = np.array(assignments)
    d = distance_matrix(locations, locations)
    max_clusters = int(max(assignments) + 1)
    score = 0
    for k in range(max_clusters):
        ind = assignments == k
        score += np.sum(d[ind, :][:, ind])
    return score


if __name__ == '__main__':
    n = 200
    nc = 5
    np.random.seed(101)
    locations = np.random.normal(size=(n, 2))

    demand = np.random.randint(5, 50, size=n)
    capacity = np.zeros(nc)
    for v in demand:
        k = np.random.choice(nc)
        capacity[k] += v + 1

    plt.figure()
    plt.suptitle('Clustering with constrained-KMeans')
    cluster_ckmeans()
    plt.show()
