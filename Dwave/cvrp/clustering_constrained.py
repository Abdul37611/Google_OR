# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

from dimod import AdjVectorBQM, DiscreteQuadraticModel
from itertools import combinations
import numpy as np


__all__ = [
    'construct_clustering_bqm',
    'construct_clustering_dqm',
]


def construct_clustering_bqm(demand, vehicle_capacity, costs,
                             lagrange_multiplier=None, precision=None):
    bqm = AdjVectorBQM(vartype='BINARY')
    for v in demand:
        for k in vehicle_capacity:
            bqm.add_variable((v, k))

    if lagrange_multiplier is None:
        lagrange_multiplier = np.std(list(costs.values()))

    if precision is None:
        max_capacity = max(vehicle_capacity.values())
        precision = 1 + int(np.ceil(np.log2(max_capacity)))
    slacks_capacity = {(k, i): "s_capacity_{}_{}".format(k, i)
                       for k in vehicle_capacity
                       for i in range(precision)}

    for s in slacks_capacity.values():
        bqm.add_variable(s)

    for u, v in combinations(demand, r=2):
        for k in vehicle_capacity:
            bqm.add_interaction((u, k), (v, k), costs[u, v])

    for k in vehicle_capacity:
        slack_terms = [(slacks_capacity[k, i], 2 ** i) for i in
                       range(precision)]
        bqm.add_linear_equality_constraint(
            [((v, k), demand[v]) for v in demand] + slack_terms,
            constant=-vehicle_capacity[k],
            lagrange_multiplier=lagrange_multiplier['capacity']
        )

    for v in demand:
        if demand[v] == 0:
            n = 0
        else:
            n = 1 + demand[v] // max(vehicle_capacity.values())
        assert n == 1
        bqm.add_linear_equality_constraint(
            [((v, k), 1.0) for k in vehicle_capacity],
            constant=-n,
            lagrange_multiplier=lagrange_multiplier['one-hot']
        )
    return bqm


def construct_clustering_dqm(demand, vehicle_capacity, costs,
                             lagrange_multiplier, precision=None, **kwargs):
    constraint_lhs = kwargs.get('constraint_lhs', {})
    dqm = DiscreteQuadraticModel()
    num_vehicles = len(vehicle_capacity)
    for v in demand:
        dqm.add_variable(num_vehicles, v)

    if not constraint_lhs:
        if precision is None:
            max_capacity = max(vehicle_capacity.values())
            precision = 1 + int(np.ceil(np.log2(max_capacity)))

        slacks = {(k, i): "s_capacity_{}_{}".format(k, i)
                  for k in vehicle_capacity
                  for i in range(precision)}

        for s in slacks.values():
            dqm.add_variable(2, s)

    for u, v in combinations(demand, r=2):
        for idk, k in enumerate(vehicle_capacity):
            dqm.set_quadratic_case(u, idk, v, idk, costs[u, v] + costs[v, u])

    capacity_penalty = kwargs.get('capacity_penalty', {})
    if not capacity_penalty:
        capacity_penalty = {k: lagrange_multiplier['capacity'] for k in vehicle_capacity}

    offset = 0
    for idk, k in enumerate(vehicle_capacity):
        if not constraint_lhs:
            slack_terms = [(slacks[k, i], 1, 2 ** i) for i in range(precision)]
            dqm.add_linear_equality_constraint(
                [(v, idk, demand[v]) for v in demand] + slack_terms,
                constant=-vehicle_capacity[k],
                lagrange_multiplier=capacity_penalty[k]
            )
        else:
            dqm.add_linear_equality_constraint(
                [(v, idk, demand[v]) for v in demand],
                constant=constraint_lhs[k],
                lagrange_multiplier=capacity_penalty[k]
            )
        offset += capacity_penalty[k] * vehicle_capacity[k] ** 2
    return dqm, offset
