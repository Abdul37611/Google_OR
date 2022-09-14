# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import numpy as np
from dwave.system import LeapHybridDQMSampler
import matplotlib.pyplot as plt
import networkx as nx

from cvrp.cvrp import CVRP
from cvrp.utilities import plot_clusters


def generate_clusters(num_clusters=3, num_forces_per_cluster=5,
                      seed=2, center_scale=100, forces_scale=15):
    np.random.seed(seed)
    thetas = np.linspace(0, np.pi * 2, num_clusters + 1)[:-1]
    x = np.cos(thetas) * center_scale
    y = np.sin(thetas) * center_scale
    centers = zip(x, y)
    locations = {}
    for c, (xc, yc) in enumerate(centers):
        for i in range(num_forces_per_cluster):
            label = f'x_{c}_{i}'
            xi, yi = np.random.normal(size=2) * forces_scale
            locations[label] = (xc + xi, yc + yi)
    return locations


def main():
    num_vehicles = 8
    locations = generate_clusters(num_clusters=num_vehicles,
                                  num_forces_per_cluster=8)

    num_forces = len(locations)
    vehicles = ["v_{}".format(k) for k in range(num_vehicles)]
    vehicle_capacity = {k: 5 * num_forces / num_vehicles
                        for k in vehicles}

    demand = {v: np.random.randint(5, 6) for v in locations}

    lagrange_multiplier = {
        'capacity': 10.0,
        'one-hot': 700.0,
    }

    vrp = CVRP()
    vrp.add_vehicles(vehicle_capacity)
    vrp.add_depot('x0', (0, 0))
    vrp.add_forces(locations, demand)
    vrp.cluster(sampler=LeapHybridDQMSampler(),
                compress=True, time_limit=None,
                lagrange=lagrange_multiplier)

    locations.update({'x0': (0, 0)})
    vrp.solve_tsp_heuristic()
    solution = vrp.solution

    fig, ax = plt.subplots()
    options = {"node_size": 500, "alpha": 0.8}
    colors = ['r', 'g', 'b', 'm', 'y', 'k', 'c'] * 10
    colors = dict(zip(vehicles, colors[:num_vehicles]))
    for k, graph in solution.items():
        nx.draw(graph, pos=vrp.coordinates, ax=ax,
                node_color=colors[k], **options)
    plot_clusters(locations, vrp.assignments, vrp.depots,
                  ax=ax, show_labels=False)
    plt.show()


if __name__ == '__main__':
    main()
