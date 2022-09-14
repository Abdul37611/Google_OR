# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import time
import numpy as np

from dwave.system import LeapHybridSampler, LeapHybridDQMSampler
from neal import SimulatedAnnealingSampler

from cvrp.tsp import solve_tsp
from cvrp.tsp_heuristic import traveling_salesman_dqm, tsp_bqm, solve_by_reversal


def generate_problem(n, seed=1):
    np.random.seed(seed)
    locs = np.random.normal(size=(n, 2))

    d = np.zeros([n, n])
    for i in range(n):
        d[i] = np.sqrt(np.sum(np.square(locs[i] - locs), axis=1))
    return list(range(n)), locs, d


def get_path(sample, n, start=None):
    route = [None] * n
    for (city, time), val in sample.items():
        if val:
            route[time] = city

    if start is not None and route[0] != start:
        # rotate to put the start in front
        idx = route.index(start)
        route = route[idx:] + route[:idx]

    return route


def print_blank():
    print()
    print('_' * 60)


def run():
    n = 10
    nodes, locations, costs = generate_problem(n, seed=1)
    bqm = tsp_bqm(nodes, costs, lagrange=6)
    dqm, _ = traveling_salesman_dqm(nodes, costs, lagrange=3)

    print_blank()
    print(f'# Locations = {n}, # Binaries = {bqm.shape[0]}, '
          f'2 * n * n * (n - 1) = {bqm.shape[1]}')

    # DQM
    t0 = time.perf_counter()
    res = LeapHybridDQMSampler().sample_dqm(dqm, time_limit=None,
                                            compress=True)
    res.resolve()
    t = time.perf_counter() - t0
    energies = dqm.energies(res.record.sample)
    ind = np.argmin(energies)
    state = res.record.sample[ind]
    sample_dqm = {v: 0 for v in bqm.variables}
    sample_dqm.update({(v, state[v]): 1 for v in dqm.variables})
    print_blank()
    print('LeapHybridDQMSampler')
    print(f'Time: {t:.2f} (runtime {res.info["run_time"] / 1e6:.2f}), '
          f'Energy: {res.first.energy:.2f}')
    num_selected = len(np.unique(list(res.first.sample.values())))
    if n != num_selected:
        print(f'constraint violated, {num_selected}nodes in the path, {n} is '
              f'expected')

    # BQM
    t0 = time.perf_counter()
    res = LeapHybridSampler().sample(bqm, time_limit=None)
    res.resolve()
    t = time.perf_counter() - t0
    print_blank()
    print('LeapHybridSampler')
    print(f'Time: {t:.2f} (runtime {res.info["run_time"] / 1e6:.2f}), '
          f'Energy: {res.first.energy:.2f}')
    sample = res.first.sample
    path = get_path(sample, n)
    if len(set(path)) != n:
        print(f'constraint violated, {len(set(path))}nodes in the path, {n} is '
              f'expected')

    # BQM dwave-neal
    if n <= 30:
        t0 = time.perf_counter()
        sampler = SimulatedAnnealingSampler()
        res = sampler.sample(bqm, num_reads=100, num_sweeps=2000)
        t = time.perf_counter() - t0
        print_blank()
        print('SimulatedAnnealingSampler')
        print(f'Time: {t:.2f} '
              f'Energy: {res.first.energy:.2f}')
        sample = res.first.sample
        path = get_path(sample, n)
        if len(set(path)) != n:
            print(f'constraint violated, {len(set(path))}nodes in the path, {n} is '
                  f'expected')

    # TSP Heuristic
    t0 = time.perf_counter()
    res = solve_tsp(costs, None, 0.001, 100, 1.01)
    t = time.perf_counter() - t0
    ens = [en for _, en in res]
    index = np.argmin(ens)
    res, en = res[index]
    sample = {v: 0 for v in bqm.variables}
    sample.update({(v, i): 1 for i, v in enumerate(res)})
    print_blank()
    print(f'Heuristic')
    print(f'Time: {t:.2f}, Energy: {en:.2f}, '
          f'(confirm bqm energy {bqm.energy(sample):.2f})')

    # TSP Heuristic - no crossing
    t0 = time.perf_counter()
    res = solve_by_reversal(costs, locations, state=None, beta=0.001,
                            max_beta=100,
                            scale=1.01, num_samples=12, beta2=10)
    t = time.perf_counter() - t0
    sample = {v: 0 for v in bqm.variables}
    sample.update({(v, i): 1 for i, v in enumerate(res)})
    print_blank()
    print(f'Heuristic - No crossing')
    print(f'Time: {t:.2f}, Energy: {bqm.energy(sample):.2f}')

    print_blank()


if __name__ == '__main__':
    run()
