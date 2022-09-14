# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

from itertools import combinations

import numpy as np
from dimod import AdjVectorBQM
from dwave_networkx.algorithms.tsp import traveling_salesman_qubo
import networkx as nx

from cvrp.crossings import does_overlap, Point
from cvrp.tsp import solve_tsp


def tsp_bqm(nodes, costs, lagrange=None, weight='weight', start=None, end=None):
    g = nx.complete_graph(nodes)
    for u, v in combinations(g.nodes, r=2):
        g[u][v]['weight'] = costs[u, v]
    qubo, offset = traveling_salesman_qubo(g, lagrange, weight)
    bqm = AdjVectorBQM.from_qubo(qubo)
    bqm.add_offset(offset)
    return bqm


def traveling_salesperson_dqm(nodes, costs, lagrange=None, weight='weight',
                              start=None):

    G = nx.complete_graph(nodes)
    for u, v in combinations(G.nodes, r=2):
        G[u][v]['weight'] = costs[u, v]

    N = G.number_of_nodes()
    if lagrange is None:
        # If no lagrange parameter provided, set to 'average' tour length.
        # Usually a good estimate for a lagrange parameter is between 75-150%
        # of the objective function value, so we come up with an estimate for
        # tour length and use that.
        if G.number_of_edges() > 0:
            lagrange = G.size(weight=weight)*G.number_of_nodes()/G.number_of_edges()
        else:
            lagrange = 2

    # some input checking
    if N in (1, 2) or len(G.edges) != N*(N-1)//2:
        msg = "graph must be a complete graph with at least 3 nodes or empty"
        raise ValueError(msg)

    # Creating the QUBO
    from dimod import DiscreteQuadraticModel
    dqm = DiscreteQuadraticModel()
    for node in G:
        dqm.add_variable(N, node)

    for pos in range(N):
        for u, v in combinations(dqm.variables, r=2):
            dqm.set_quadratic_case(u, pos, v, pos, lagrange)

    if start is None:
        # Objective that minimizes distance
        for u, v in combinations(G.nodes, 2):
            for pos in range(N):
                nextpos = (pos + 1) % N

                # going from u -> v
                bias = dqm.get_quadratic_case(u, pos, v, nextpos)
                dqm.set_quadratic_case(u, pos, v, nextpos,
                                       bias + G[u][v][weight])

                # going from v -> u
                bias = dqm.get_quadratic_case(v, pos, u, nextpos)
                dqm.set_quadratic_case(v, pos, u, nextpos,
                                       bias + G[u][v][weight])
    else:
        raise NotImplementedError
    return dqm, 0


traveling_salesman_dqm = traveling_salesperson_dqm


def solve_by_reversal(d, locations, state=None, beta=0.1, max_beta=100,
                      scale=1.1, num_samples=12, beta2=10):
    if state is None:
        state = []
        n = len(locations)
        for i in range(num_samples):
            seq = list(range(n))
            np.random.shuffle(seq)
            state.append(seq)
    res = solve_tsp(d, state, beta, max_beta, scale, num_samples)
    ens = [xx[1] for xx in res]
    ind = np.argmin(ens)
    res = res[ind][0]
    crossings = find_crossing(res, locations)
    len_crossing = len(crossings)
    current_best = np.min(ens)
    persistence = 0
    iteration = 0
    while len_crossing > 0 and iteration < 200:
        crossings = list(crossings)
        for (i, j), (k, l) in crossings:
            start = res.index(j)
            end = res.index(k) + 1
            res[start:end] = reversed(res[start:end])
            current_best = tsp_cost(res, d)
        if persistence > 5:
            break
        res_tmp = solve_tsp(d, [res] * num_samples, beta2, max_beta,
                            scale, num_samples)
        ens = [xx[1] for xx in res_tmp]
        ind = np.argmin(ens)
        if np.min(ens) <= current_best:
            current_best = np.min(ens)
            res = res_tmp[ind][0]
            persistence = 0
        else:
            persistence += 1
            beta2 *= 0.5
        crossings = find_crossing(res, locations)
        len_crossing = len(crossings)
        iteration += 1
    return res


def tsp_cost(res, d):
    en = 0
    for a, b in zip(res[:-1], res[1:]):
        en += d[a, b]
    en += d[res[-1], res[0]]
    return en


def find_crossing(res, locations):
    if isinstance(res[0], tuple):
        crossings = []
        for r, _ in res:
            crossings.append(find_crossing(r, locations))
        return crossings

    edges = []
    for i, j in zip(res[:-1], res[1:]):
        edges.append((i, j))
    edges.append((res[-1], res[0]))
    crossings = set()
    for (i, j), (k, l) in combinations(edges, r=2):
        if j == k or i == l:
            continue
        p1 = Point(*locations[i])
        p2 = Point(*locations[j])
        q1 = Point(*locations[k])
        q2 = Point(*locations[l])
        cross = does_overlap(p1, p2, q1, q2)
        if cross:
            crossings.add(((i, j), (k, l)))
    return crossings
