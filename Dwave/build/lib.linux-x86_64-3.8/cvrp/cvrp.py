# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import numpy as np
import networkx as nx
import dimod
import dwave.system
from dimod.variables import Variables
from typing import Union, Dict, Hashable
from collections import defaultdict
from itertools import combinations
from cvrp.tsp import solve_tsp
from cvrp.constrained_k_means import CKMeans

from cvrp.clustering_constrained import *
from cvrp.tsp_heuristic import solve_by_reversal


def l2_distance(p1, p2, label1=None, label2=None):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class CVRP:
    """A class to handle data and operations related to Multi-vehicle
    routing problem
    
    Args: 
        cost_function: The cost function that takes two coordinates and two
            labels and computes the cost. The function needs to take all four
            inputs, but it doesn't necessarily make use of all of it

        capacity: A dictionary of vehicle label and vehicle capacity
        depot: Either a string for the label of the depot or a dictionary of
            label, coordinates
        demand: A dictionary of location/force labels as keys and the value
            of the demand.
        coordinates: A dictionary of location/force labels as keys and the
            coordinates as values

    """

    def __init__(self, cost_function=l2_distance, capacity: dict = None,
                 depot: Union[dict, str] = None, demand: dict = None,
                 coordinates: dict = None):
        self._vehicles = Variables()
        self._depots = Variables()
        self._forces = Variables()
        self._vehicle_capacity = {}
        self._coordinates = {}
        self._costs = {}
        self._demand = {}  # this will change
        self._supply_types = Variables()
        self._cost_callback = cost_function
        self._optimization = {}
        self._solution: Dict[Hashable, nx.DiGraph] = None

        if depot is not None:
            if isinstance(depot, str):
                if depot not in coordinates:
                    raise ValueError(f"Please provide the coordinates for "
                                     f"{depot}")
                else:
                    self.add_depot(depot, coordinates[depot])
            elif isinstance(depot, dict):
                self.add_depots(depot)

        if coordinates is not None and demand is not None:
            self.add_forces(coordinates, demand)

        if capacity is not None:
            self.add_vehicles(capacity)

    def add_depot(self, label, coordinates: tuple):
        """Add depot by label and coordinate. The coordinate is a tuple of
        two numbers

        Args:
            label: the label for depot
            coordinates: A tuple of x-y coordinates

        """
        if label in self._forces:
            raise ValueError
        self._depots._append(label)
        self._add_coordinates(label, coordinates)

    def add_depots(self, coordinates: dict):
        """Add multiple depots by label and coordinates

        Args:
            coordinates: A dictionary of depot label, coordinates

        """
        for label in coordinates:
            self.add_depot(label, coordinates[label])

    def add_force(self, label, coordinates: tuple, demand):
        """Add force element by force label and coordinates. Each force also
        need a value for its supply demand

        Args:
            label: the label for the force element
            coordinates: A tuple of x-y coordinates
            demand: The amount of supply that is needed by the force element

        """
        if label in self._depots:
            return
        self._forces._append(label)
        self._add_coordinates(label, coordinates)
        self._demand[label] = demand

    def add_forces(self, coordinates: dict, demand: dict):
        """Add multiple force elements given a dictionary of coordinates and
        demands

        Args:
            coordinates: A dictionary of force label, coordinates
            demand: A dictionary of force label, demand

        """
        for label in coordinates:
            self.add_force(label, coordinates[label], demand[label])

    def add_vehicle(self, label, capacity):
        """Add a vehicle label and capacity.

        Args:
            label: the label for the force element
            capacity: The amount of supply that the vehicle can carry

        """
        self._vehicles._append(label)
        self._vehicle_capacity[label] = capacity

    def add_vehicles(self, capacity: dict):
        """Add multiple vehicles by a dictionary of label, capacity

        Args:
            capacity: A dictionary of vehicle label, capacity

        """
        for label in capacity:
            self.add_vehicle(label, capacity[label])

    def _add_coordinates(self, label, coordinates):
        for key in self._coordinates:
            self._costs[label, key] = self._cost_callback(
                coordinates, self._coordinates[key], label, key)
            self._costs[key, label] = self._cost_callback(
                self._coordinates[key], coordinates, key, label)
        self._coordinates[label] = coordinates

    @property
    def costs(self):
        """The cost of travel from a location to another

        Returns:
            dict: A dictionary of pair of labels and travel cost

        """
        return self._costs

    @property
    def depots(self):
        return self._depots

    @property
    def vehicles(self):
        return self._vehicles

    @property
    def forces(self):
        return self._forces

    @property
    def demand(self):
        return self._demand

    @property
    def locations(self):
        return self._coordinates

    coordinates = locations

    @classmethod
    def from_graph(cls):
        pass

    @property
    def solution(self) -> Dict[Hashable, nx.DiGraph]:
        return self._solution

    def __repr__(self):
        return self.locations.__repr__()

    def _get_clustering_bqm(self, **kwargs):
        self._optimization['bqm'] = construct_clustering_bqm(
            self.demand, self._vehicle_capacity, self.costs, **kwargs)

    def _get_clustering_dqm(self, **kwargs):
        self._optimization['dqm'], offset = construct_clustering_dqm(
            self.demand, self._vehicle_capacity, self.costs, **kwargs)
        self._optimization['dqm_offset'] = offset

    def cluster(self, sampler=None, lagrange=None, **sampler_parameters):
        """Cluster locations and assign them vehicle.

        Formulate the clustering problem as either a BQM or DQM. The
        formulation is chosen based on the sampler. If the sampler is an
        instance of DQM solvers, it will formulate the clustering problem as
        a DQM.

        Args:
            sampler: A `dimod.Sampler`
                LeapHybridSampler: The hybrid solver that solves problems
                    provided as dimod.BQM
                LeapHybridDQMSampler: The hybrid solver that solves problems
                    provided as dimod.DQM

            lagrange: A dictionary of lagrange multiplier for penalty
                strength of various constraints. The requires keys are:
                    capacity: When using BQM or DQM formulation.
                    one-hot: Needed when using BQM formulation.

            sampler_parameters: parameters that can be passed to the sampler.
                Leap samplers have time_limit as one of their parameters

        """
        if not self._clustering_feasible():
            raise ValueError("Clustering is not feasible")
        if sampler is None:
            raise ValueError("You must provide a sampler")
        _dqm = dwave.system.LeapHybridDQMSampler
        if isinstance(sampler, _dqm):
            constraint_lhs = self._optimization.get('capacity_violation_qkp', {})
            capacity_penalty = self._optimization.get('capacity_penalty', {})
            current_assignments = self._optimization.get('assignments', {})
            self._get_clustering_dqm(lagrange_multiplier=lagrange,
                                     constraint_lhs=constraint_lhs,
                                     current_assignments=current_assignments,
                                     capacity_penalty=capacity_penalty)
            if isinstance(sampler, dwave.system.LeapHybridDQMSampler):
                res = sampler.sample_dqm(
                    self._optimization['dqm'], **sampler_parameters)
                res.resolve()
            else:
                res = sampler.sample(
                    self._optimization['dqm'], **sampler_parameters)
            sample = res.first.sample
            assignments = defaultdict(list)
            for v in self._forces:
                assignments[v].append(self._vehicles[int(sample[v])])

            capacity_violation = {}
            for k in self._vehicles:
                capacity_violation[k] = -self._vehicle_capacity[k]

            for v in self._forces:
                k = int(sample[v])
                capacity_violation[self._vehicles[k]] += self._demand[v]

            self._optimization['assignments'] = assignments
            self._optimization['capacity_violation'] = assignments
        elif isinstance(sampler, dimod.Sampler):
            self._get_clustering_bqm(lagrange_multiplier=lagrange)
            res = sampler.sample(
                self._optimization['bqm'], **sampler_parameters)
            res.resolve()
            sample = res.first.sample
            assignments = defaultdict(list)
            for v in self._forces:
                for k in self._vehicles:
                    if sample[v, k]:
                        assignments[v].append(self._vehicles[k])

            capacity_violation = {}
            for k in self._vehicles:
                capacity_violation[k] = -self._vehicle_capacity[k]

            for v in self._forces:
                for k in self._vehicles:
                    if sample[v, k]:
                        capacity_violation[self._vehicles[k]] += self._demand[v]

            self._optimization['assignments'] = assignments
            self._optimization['capacity_violation'] = assignments

        elif isinstance(sampler, str) and sampler.lower() == 'kmeans':
            clusterer = CKMeans(k=len(self._vehicles))
            locations = [self.locations[k] for k in self._forces]
            demand = [self.demand[k] for k in self._forces]
            capacity = [self._vehicle_capacity[k] for k in self._vehicles]
            assignments = clusterer.predict(
                np.array(locations),
                demand,
                capacity,
                time_limit=sampler_parameters.get("time_limit", 5))

            assignments = list(map(lambda x: [self._vehicles[int(x)]],
                                   assignments))
            assignments = dict(zip(self._demand.keys(), assignments))

            capacity_violation = {}
            for k in self._vehicles:
                capacity_violation[k] = -self._vehicle_capacity[k]

            self._optimization['assignments'] = assignments
            self._optimization['capacity_violation'] = assignments

    @property
    def assignments(self):
        """The assignment of locations to vehicles in the clustering step

        Returns:
             dict: A dictionary with force labels as keys, and a list of
             vehicle that the location is assigned to as values

        """
        return self._optimization.get('assignments', {})

    @property
    def capacity_violation(self):
        return self._optimization.get('capacity_violation', {})

    def _clustering_feasible(self):
        total_demand = sum(self._demand.values())
        total_capacity = sum(self._vehicle_capacity.values())
        return total_capacity >= total_demand

    def solve_tsp_heuristic(self, disentangle=False, **parameters):
        """Use a heuristic approach to solve TSP problem approximately for
        small number of cities.

        The method uses a simple swap operation and improves
        the tour step by step. This function can be replace by any open
        source tools for solving TSP problems.

        Args:
            disentangle: Whether to use a heuristic approach for resolving
                the crossing paths. For a metric, symmetric TSP,
                path crossing is always suboptimal.
            parameters: optimization parameters to pass to the TSP heuristic
                solver

        """
        assignments = self.assignments
        costs = self.costs
        depots = self.depots
        rev = defaultdict(list)
        for v, clusters in assignments.items():
            for c in clusters:
                rev[c].append(v)
        for k in rev:
            for depot in depots:
                rev[k].append(depot)
        paths = {}
        edges = []
        multiple_edges = defaultdict(list)
        beta = parameters.get('beta', 0.001)
        max_beta = parameters.get('max_beta', 1000)
        scale = parameters.get('scale', 1.01)
        num_samples = parameters.get('num_samples', 12)
        for k, routes in rev.items():
            costs_subset = _as_array(routes, costs)
            if disentangle:
                locations = [self.locations[k] for k in routes]
                indices = solve_by_reversal(costs_subset, locations, None,
                                            beta, max_beta,
                                            scale,
                                            num_samples, max_beta / 10)
            else:
                indices = solve_tsp(costs_subset, None, beta, max_beta,
                                    scale, num_samples)
                ens = [xx[1] for xx in indices]
                i = np.argmin(ens)
                indices = indices[i][0]
            n = len(rev[k])
            path = {i: routes[indices[i]] for i in range(n)}
            for i in range(n):
                edges.append((k, path[i], path[(i + 1) % n]))
                multiple_edges[k].append((path[i], path[(i + 1) % n]))
            paths[k] = path
        self._solution = {k: nx.DiGraph(edges) for k, edges in
                          multiple_edges.items()}
        self._paths = paths

    @property
    def paths(self):
        return self._paths


def _as_array(labels, costs):
    res = np.zeros([len(labels)] * 2)
    for (i, a), (j, b) in combinations(enumerate(labels), r=2):
        res[i, j] = costs[a, b]
        res[j, i] = costs[b, a]
    return res


if __name__ == '__main__':
    p = CVRP()
    p.add_vehicle('v1', 15)
    p.add_vehicle('v2', 15)
    p.add_depot('depot', (0, 0))

    p.add_force('f1', (0, 1), 5)
    p.add_force('f2', (1, 1), 5)
    p.add_force('f3', (1, 0), 5)
    p.add_force('f4', (1, -1), 5)
    p.add_force('f5', (0, -1), 5)

    p2 = CVRP()
    p2.add_vehicles(dict(v1=15, v2=15))
    p2.add_depots({'depot': (0, 0)})

    p2.add_forces(
        dict(f1=(0, 1), f2=(1, 1), f3=(1, 0), f4=(1, -1), f5=(0, -1)),
        dict(f1=5, f2=5, f3=5, f4=5, f5=5),
    )
    for k in p.costs:
        assert p.costs[k] == p2.costs[k]

    print(p)
