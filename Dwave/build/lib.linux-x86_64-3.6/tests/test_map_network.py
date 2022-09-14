# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

import unittest
import folium
import folium.plugins
from visualization.map_network import (generate_mapping_information,
                                       generate_solution_map_drone_network,
                                       generate_solution_map_street_network,
                                       show_locations_on_initial_map,
                                       RoutingProblemParameters)


class TestMapNetworkFlow(unittest.TestCase):
    """Test the flow of components in map_network.py, as used in the app.

    """
    def test_generate_solution_map_drone_and_street_networks(self):
        for vehicle_solution_method in [generate_solution_map_drone_network, generate_solution_map_street_network]:
            with self.subTest(msg="Check output of solutions is populated.", 
                              vehicle_solution_method = vehicle_solution_method):
                map_network, depot_id, locations = generate_mapping_information(num_clients=10)

                initial_map = show_locations_on_initial_map(map_network, depot_id, locations)

                self.assertIsInstance(initial_map, folium.folium.Map)

                routing_problem_parameters = RoutingProblemParameters(
                    initial_map, map_network, depot_id, locations, 10, 2, "Classical (K-means)", 5.0
                )

                solution_dict = vehicle_solution_method(routing_problem_parameters)

                self.assertIsInstance(solution_dict["map"], folium.folium.Map)
                self.assertIsInstance(solution_dict["wall_clock_time"], float)
                self.assertIsInstance(solution_dict["solution_cost"], dict)

                for vehicle_id in solution_dict["solution_cost"]:
                    self.assertGreater(solution_dict["solution_cost"][vehicle_id]["optimized_cost"], 0)
                    self.assertGreater(solution_dict["solution_cost"][vehicle_id]["forces_serviced"], 0)
                    self.assertGreater(solution_dict["solution_cost"][vehicle_id]["water"], 0)
                    self.assertGreater(solution_dict["solution_cost"][vehicle_id]["food"], 0)
                    self.assertGreater(solution_dict["solution_cost"][vehicle_id]["other"], 0)
