# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import unittest
import numpy as np
import networkx as nx
from cvrp.cvrp import CVRP
from cvrp.cvrp import l2_distance
from visualization.map_network import generate_mapping_information

class TestCvrpFlow(unittest.TestCase):
    """Test functionality of CVRP class flow as used for app.

    """
    def test_l2_distance_cost_function(self):
        test_l2 = l2_distance((0,0),(1,1),"point_1", "point_2")
        self.assertEqual(test_l2, np.sqrt(2))

    def test_cvrp_flow(self):
        map_graph, depot_id, locations = generate_mapping_information(num_clients=10)

        demand = nx.get_node_attributes(map_graph, "demand")
        depot = {depot_id: (map_graph.nodes[depot_id]["y"], map_graph.nodes[depot_id]["x"])}
        locations = {
            location_id: (map_graph.nodes[location_id]["y"], map_graph.nodes[location_id]["x"])\
                for location_id in locations
        }

        cvrp = CVRP(cost_function=l2_distance)
        cvrp.add_depots(depot)
        cvrp.add_forces(locations, demand)
        cvrp.add_vehicles({k: 100 for k in range(4)}) # 5 vehicles with 100 capacity

        self.assertIn(depot_id, cvrp.depots)
        for location in locations:
            self.assertIn(location, cvrp.forces)
        
        for vehicle in range(4):
            self.assertIn(vehicle, cvrp.vehicles)

        # CVRP in unsolved state
        self.assertIsNone(cvrp.solution)

        # Run clustering + tsp heuristic to transition into solved state
        cvrp.cluster(sampler='kmeans', step_size=0.6)
        cvrp.solve_tsp_heuristic(disentangle=True)

        # Confirm solved state
        self.assertIsInstance(cvrp.solution, dict)
        