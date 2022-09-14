# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import numpy as np
from typing import Iterable


def get_locations(num_depots, num_clients, labels=None, seed=0):
    np.random.seed(seed)
    xloc = np.random.normal(size=num_clients + num_depots) * 20
    yloc = np.random.normal(size=num_clients + num_depots) * 20
    xloc[:num_depots] = 0
    yloc[:num_depots] = 0
    locations = zip(xloc, yloc)
    if labels is None:
        return dict(zip(range(len(xloc)), locations))
    if len(labels) != len(xloc):
        raise ValueError("The number of labels and locations must be the "
                         "same and equal to {} + {}"
                         .format(num_depots, num_clients))
    return dict(zip(labels, locations))


def plot_clusters(locations, assignments, depot, ax=None, show_labels=False,
                  colors=None):
    vehicles = list(set.union(*map(set, assignments.values())))
    num_vehicles = len(vehicles)
    if colors is None:
        colors = ['r', 'g', 'b', 'm', 'y', 'k', 'c'] * 10
        colors = dict(zip(vehicles, colors[:num_vehicles]))

    if ax is None:
        _, ax = get_ax(vehicles)

    if not isinstance(ax, dict):
        ax = dict(zip(vehicles, [ax] * len(vehicles)))

    if not isinstance(depot, Iterable):
        depot = [depot]
    for v in depot:
        x, y = locations[v]
        for k in ax:
            ax[k].scatter(x, y, color='r', marker='s')

    for v, (x, y) in locations.items():
        if v in depot:
            continue
        k = assignments[v][0]
        ax[k].scatter(x, y, color=colors[k])
        if show_labels:
            ax[k].text(x, y+2, str(assignments[v]))
            ax[k].text(x, y-2, str(v))


def get_ax(vehicles):
    import matplotlib.pyplot as plt
    num_vehicles = len(vehicles)
    n = int(np.round(np.sqrt(num_vehicles)))
    m = n
    if m * n < num_vehicles:
        m += 1
    fig, ax = plt.subplots(m, n)
    axx = []
    for aa in ax:
        try:
            for a in aa:
                axx.append(a)
        except:
            axx.append(aa)
    axx = axx[:num_vehicles]
    axx = dict(zip(vehicles, axx))
    return fig, axx
