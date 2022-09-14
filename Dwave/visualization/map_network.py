# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

from typing import NamedTuple, Tuple
import os
import random
import time
from functools import partial
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import osmnx as ox
import folium
import seaborn as sns
import warnings
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="abdullah.akratech@gmial.com")
from geopandas.tools import geocode
import pandas as pd
from itertools import groupby
 

from dwave.system import LeapHybridDQMSampler
from cvrp.cvrp import CVRP


ox.settings.use_cache = True
ox.settings.overpass_rate_limit = False

depot_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "Depot Location.png"))
force_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "Force Location.png"))
#address = 'Rockonia Rd, Koongal QLD 4701, Australia'
#address = 'Cambridge St, Rockhampton QLD 4700, Australia'
depot_icon = folium.CustomIcon(depot_icon_path, icon_size=(32, 37))




class RoutingProblemParameters(NamedTuple):
    """Structure to hold all provided problem parameters.

    Args:
        folium_map: Folium map with locations already shown on map.
        map_network: `nx.MultiDiGraph` where nodes and edges represent locations and routes.
        depot_id: Node ID of the depot location.
        client_subset: client_subset: List of client IDs in the map's graph.
        num_clients: Number of locations to be visited.
        num_vehicles: Number of vehicles to deploy on routes.
        sampler_type: Sampler type to use in solving CVRP.
        time_limit: Time limit in seconds to run optimization for.

    """
    folium_map: folium.Map
    map_network: nx.MultiDiGraph
    depot_id: int
    client_subset: list
    num_clients: int
    num_vehicles: int
    sampler_type: str
    time_limit: float
    cap_dict: dict

def set_address(ad):
    global address 
    address=ad

def get_df(df):
    global dataframe 
    dataframe=df

def _cost_between_nodes(dijkstra_paths_and_lengths: dict, p1, p2, start_node: int, end_node: int) -> float:
    return dijkstra_paths_and_lengths[start_node][0][end_node]

def _cost_between_nodes_haversine(p1, p2, start, end) -> float:
    radius_earth = 6371000 # meters
    lat1_rad, lat2_rad = np.deg2rad((p1[0], p2[0]))
    diff_lat_rad, diff_lon_rad = np.deg2rad((p2[0] - p1[0], p2[1] - p1[1]))
    return 2 * radius_earth * np.arcsin(
        np.sqrt(np.sin(diff_lat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(diff_lon_rad/2)**2)
    )

def _map_network_from_address(address: str, network_type = "drive") -> nx.MultiDiGraph:
    """Generate a sparse `nx.MultiDiGraph` network from address.

    Args:
        address: String address to generate network from.

    Returns:
        `nx.MultiDiGraph` object of map network, retaining the largest
            strongly connected component of the graph.

    """
    G = ox.graph_from_address(address, dist=1000, network_type=network_type)
    G = ox.utils_graph.get_largest_component(G, strongly=True)
    return G

def _all_pairs_dijkstra_dict(G: nx.MultiDiGraph) -> dict:
    return dict(nx.all_pairs_dijkstra(G, weight='length'))

def _build_node_index_map(G: nx.MultiDiGraph) -> dict:
    return dict(enumerate(G.nodes(data=True)))

def _find_node_index_central_to_network(node_index_map: dict) -> int:
    coordinates = np.zeros((len(node_index_map), 2))
    for node_index, node in node_index_map.items():
        coordinates[node_index][0] = node[1]['y']
        coordinates[node_index][1] = node[1]['x']
    
    centroid = np.sum(coordinates, 0) / len(node_index_map)
    kd_tree = cKDTree(coordinates)
    return kd_tree.query(centroid)[1]

def _select_client_nodes(G: nx.MultiDiGraph, depot_id: int, num_clients: int) -> list:
    """Select a subset of nodes to represent client and depot locations.

    Args:
        G: Map network to draw subset from.
        depot_id: Node ID of the depot location.
        num_clients: Number of client locations desired in CVRP problem.

    Returns:
        List containing subset of nodes of length `num_clients`.

    """
    random.seed(num_clients)
    graph_copy = G.copy()
    graph_copy.remove_node(depot_id)
    return random.sample(list(graph_copy.nodes), num_clients)
   
def generate_mapping_information(num_clients: int) -> Tuple[nx.MultiDiGraph, int, list]:
    """Return `nx.MultiDiGraph` with client demand, depot id in graph, client ids in graph.

    Args:
        num_clients: Number of locations to be visited in total.

    Returns:
        map_network: `nx.MultiDiGraph` where nodes and edges represent locations agenerate_mapping_informationnd routes.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.

    """
    map_network = _map_network_from_address(address=address)

    node_index_map = _build_node_index_map(map_network)
    
    #Change depot_id = node_index_map[_find_node_index_central_to_network(node_index_map)][0]
    #Change client_subset = _select_client_nodes(map_network, depot_id, num_clients = num_clients)
    
    #Change (depot node from address) #####################################
    depot_id=add_depot(map_network,address)
    #Change ###############################################################

    #Change (location nodes) ##############################################
    client_subset=get_or_add_nodes(map_network,dataframe)
    #Change ##############################################################

    #for node_id in client_subset:
    #    map_network.nodes[node_id]['demand_water'] = random.choice([1,2])
    #    map_network.nodes[node_id]['demand_food'] = random.choice([1,2])
    #    map_network.nodes[node_id]['demand_other'] = random.choice([1,2])
    #    
    #    map_network.nodes[node_id]['demand'] = map_network.nodes[node_id]['demand_water'] +\
    #                                            map_network.nodes[node_id]['demand_food'] +\
    #                                            map_network.nodes[node_id]['demand_other']

    return map_network, depot_id, client_subset

def _map_network_as_cvrp(G: nx.MultiDiGraph, depot_id: int, client_subset: list, 
                         partial_cost_func: callable, num_vehicles: int, cap_dict: dict) -> CVRP:
    """Generate CVRP instance from map network and select information.

    Args:
        G: Map network to build CVRP problem from.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.
        partial_cost_func: Partial cost function to pass to CVRP.
        num_vehicles: Number of vehicles to deploy on routes.

    Returns:
        Instance of CVRP class.

    """
    demand = nx.get_node_attributes(G, 'demand')
    depot = {depot_id: (G.nodes[depot_id]['y'], G.nodes[depot_id]['x'])}
    clients = {client_id: (G.nodes[client_id]['y'], G.nodes[client_id]['x']) for client_id in client_subset}

    cvrp = CVRP(cost_function=partial_cost_func)
    cvrp.add_depots(depot)
    cvrp.add_forces(clients, demand)
    #cvrp.add_vehicles({k: -(-sum(demand.values()) // num_vehicles) for k in range(num_vehicles)})
    cvrp.add_vehicles(cap_dict)
    return cvrp


def show_locations_on_initial_map(G: nx.MultiDiGraph, depot_id: int, client_subset: list) -> folium.folium.Map:
    """Prepare map to be rendered initially on app screen.

    Args:
        G: `nx.MultiDiGraph` to build map from.
        depot_id: Node ID of the depot location.
        client_subset: client_subset: List of client IDs in the map's graph.

    Returns:

    """
    folium_map = ox.plot_graph_folium(G, opacity=0.0)

    folium.Marker(
        (G.nodes[depot_id]['y'], G.nodes[depot_id]['x']),
        tooltip=folium.map.Tooltip(text="Depot", style="font-size: 1.4rem;"), icon=depot_icon
    ).add_to(folium_map)

    for force_id in client_subset:
        if force_id != depot_id:
            location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
            if type(force_id)==int:
                TempVar=ID_loc[force_id]
            folium.Marker(
                (G.nodes[force_id]['y'], G.nodes[force_id]['x']),
                tooltip=folium.map.Tooltip(text=f"{force_id if type(force_id)==str else TempVar }\
                    <br> water: {G.nodes[force_id]['demand_water'] * 100} \
                          <br> food: {G.nodes[force_id]['demand_food'] * 100} <br> \
                          other: {G.nodes[force_id]['demand_other'] * 100}", style="font-size: 1.4rem;"),
                icon=location_icon
            ).add_to(folium_map)

    folium.plugins.Fullscreen().add_to(folium_map)
    return folium_map


def _plot_solution_routes_on_drone_map(folium_map, G, solution: dict, depot_id: int) -> folium.folium.Map:
    """Generate interactive folium map for drone routes given solution dictionary.

    Args:
        G: Map network to plot.
        solution: Solution returned by CVRP.
        depot_id: Node ID of the depot location.

    Returns:
        `folium.folium.Map` object,  dictionary with solution cost information.

    """
    solution_cost_information = {}
    palette = sns.color_palette("colorblind", len(solution)).as_hex()

    locations = {}
    for vehicle_id, route_network in solution.items():
        solution_cost_information[vehicle_id + 1] = {
            "optimized_cost": 0,
            "forces_serviced": len(route_network.nodes) - 1,
            "water": 0,
            "food": 0,
            "other": 0,
        }

        for node in route_network.nodes:
            locations.update({node:(G.nodes[node]['y'], G.nodes[node]['x'])})
            if node != depot_id:
                location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
                if type(node)==int:
                    TempVar=ID_loc[node]
                folium.Marker(
                    locations[node],
                    tooltip=folium.map.Tooltip(text=f"{node if type(node)==str else TempVar }\
                        <br> water: {G.nodes[node]['demand_water'] * 100} \
                                                    <br> food: {G.nodes[node]['demand_food'] * 100} <br> \
                                                    other: {G.nodes[node]['demand_other'] * 100} <br> vehicle: {vehicle_id+1}",
                                               style="font-size: 1.4rem;"),
                    icon=location_icon
                ).add_to(folium_map)

                solution_cost_information[vehicle_id + 1]["water"] += G.nodes[node]['demand_water'] * 100
                solution_cost_information[vehicle_id + 1]["food"] += G.nodes[node]['demand_food'] * 100
                solution_cost_information[vehicle_id + 1]["other"] += G.nodes[node]['demand_other'] * 100

        route_color = palette.pop()
        for edge in route_network.edges:
            solution_cost_information[vehicle_id + 1]["optimized_cost"] += _cost_between_nodes_haversine(
                locations[edge[0]], locations[edge[1]], edge[0], edge[1]
            )
            folium.PolyLine(
                [locations[node] for node in edge], color=route_color
            ).add_to(folium_map)

    return folium_map, solution_cost_information


def _plot_solution_routes_on_street_map(folium_map, G, solution: dict, depot_id: int, dijkstra_paths: dict) -> folium.folium.Map:
    """Generate interactive folium map for truck routes given solution dictionary.

    Args:
        G: Map network to plot.
        solution: Solution returned by CVRP.
        depot_id: Node ID of the depot location.
        dijkstra_paths: Dictionary containing both paths and path lengths between any two nodes.

    Returns:
        `folium.folium.Map` object, dictionary with solution cost information.

    """
    solution_cost_information = {}
    palette = sns.color_palette("colorblind", len(solution)).as_hex()
    route_list=[]
    for vehicle_id, route_network in solution.items():
        solution_cost_information[vehicle_id + 1] = {
            "optimized_cost": 0,
            "forces_serviced": len(route_network.nodes) - 1,
            "water": 0,
            "food": 0,
            "other": 0,
        }

        for node in route_network.nodes:
            if node != depot_id:
                location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
                if type(node)==int:
                    TempVar=ID_loc[node]
                folium.Marker(
                    (G.nodes[node]['y'], G.nodes[node]['x']),
                    tooltip=folium.map.Tooltip(text=f"{node if type(node)==str else TempVar }\
                        <br> water: {G.nodes[node]['demand_water'] * 100} \
                                                    <br> food: {G.nodes[node]['demand_food'] * 100} <br> \
                                                    other: {G.nodes[node]['demand_other'] * 100} <br> vehicle: {vehicle_id+1}",
                                               style="font-size: 1.4rem;"),
                    icon=location_icon
                ).add_to(folium_map)

                solution_cost_information[vehicle_id + 1]["water"] += G.nodes[node]['demand_water'] * 100
                solution_cost_information[vehicle_id + 1]["food"] += G.nodes[node]['demand_food'] * 100
                solution_cost_information[vehicle_id + 1]["other"] += G.nodes[node]['demand_other'] * 100

        route_color=palette.pop()
        routes = [dijkstra_paths[start][1][end] for start,end in route_network.edges]
        route_list.append(routes)
        solution_cost_information[vehicle_id + 1]["optimized_cost"] += sum([dijkstra_paths[start][0][end] for start,end in route_network.edges])
        
        for route in routes:
            folium_map = ox.plot_route_folium(G, route=route, route_map=folium_map, fit_bounds=False, color=route_color, popup_attribute='length')
            
    return folium_map, solution_cost_information,route_list 


def generate_solution_map_drone_network(problem_parameters: RoutingProblemParameters) -> dict:
    """Generate map with solution routes plotted, map centered on depot location, for drone routes.

    Args:
        problem_parameters: NamedTuple that specifies all problem details.

    Returns:
        dict containing solved state map, solution information.

    """
    start_time = time.perf_counter()
    cvrp = _map_network_as_cvrp(
        problem_parameters.map_network, 
        problem_parameters.depot_id, 
        problem_parameters.client_subset, 
        _cost_between_nodes_haversine, 
        problem_parameters.num_vehicles,
        problem_parameters.cap_dict
    )
    solved_state_cvrp = _solved_state_cvrp(
        cvrp,
        problem_parameters.sampler_type, 
        disentangle=False, 
        time_limit=problem_parameters.time_limit
    )

    wall_clock_time = time.perf_counter() - start_time

    solution_map, solution_cost_information = _plot_solution_routes_on_drone_map(
        problem_parameters.folium_map, 
        problem_parameters.map_network, 
        solved_state_cvrp.solution, 
        problem_parameters.depot_id
    )
    global Loc_nodes_drone
    Loc_nodes_drone=solved_state_cvrp.solution
    # dict with vehicle and customer location not routes, routes get from
    # display solution function (routes no need for drones)
    return {
        "map": solution_map,
        "wall_clock_time": wall_clock_time,
        "solution_cost": solution_cost_information,
        
    }

def generate_solution_map_street_network(problem_parameters: RoutingProblemParameters) -> dict:
    """Generate map with solution routes plotted, map centered on depot location, for truck routes.

    Args:
        problem_parameters: NamedTuple that specifies all problem details.

    Returns:
        dict containing solved state map, solution information.

    """
    start_time = time.perf_counter()
    paths_and_lengths = _all_pairs_dijkstra_dict(problem_parameters.map_network)

    partial_cost_func = partial(_cost_between_nodes, paths_and_lengths)

    cvrp = _map_network_as_cvrp(
        problem_parameters.map_network, 
        problem_parameters.depot_id, 
        problem_parameters.client_subset, 
        partial_cost_func, 
        problem_parameters.num_vehicles,
        problem_parameters.cap_dict
    )

    solved_state_cvrp = _solved_state_cvrp(
        cvrp, 
        problem_parameters.sampler_type, 
        disentangle=False, 
        time_limit=problem_parameters.time_limit
    )
    wall_clock_time = time.perf_counter() - start_time

    solution_map, solution_cost_information,routes = _plot_solution_routes_on_street_map(
        problem_parameters.folium_map, 
        problem_parameters.map_network, 
        solved_state_cvrp.solution, 
        problem_parameters.depot_id, 
        paths_and_lengths
    )
    
    global Loc_nodes
    global Route_nodes
    Loc_nodes=solved_state_cvrp.solution
    Route_nodes=routes

    return {
        "map": solution_map,
        "wall_clock_time": wall_clock_time,
        "solution_cost": solution_cost_information,
    }

def _solved_state_cvrp(cvrp, sampler_type, disentangle=True, time_limit=None):
    if sampler_type == 'Classical (K-Means)':
        cvrp.cluster(sampler='kmeans', step_size=0.6, time_limit=time_limit)
    if sampler_type == "Quantum Hybrid (DQM)":
        try:
            cvrp.cluster(sampler=LeapHybridDQMSampler(), lagrange={'capacity': 1.0}, time_limit=time_limit)
        except ValueError:
            warnings.warn("Defaulting to minimum time limit for Leap Hybrid DQM Sampler.")
            cvrp.cluster(
                sampler=LeapHybridDQMSampler(), lagrange={'capacity': 1.0}, time_limit=None
            )
    cvrp.solve_tsp_heuristic(disentangle=disentangle)
    return cvrp

# Change###################################

def add_depot(map_network,address):
    
    location = geocode(address, provider="nominatim" , user_agent = 'my_request')
    point=location.iloc[0,0]

    map_network.add_node('depot', y=point.y,x=point.x)
    n_edge_depot = ox.distance.nearest_edges(map_network, point.x, point.y, return_dist=False)
    node1=n_edge_depot[0]
    node2=n_edge_depot[1]
    node_key=n_edge_depot[2]
    nn='depot'
            
    edge_att=map_network.get_edge_data(node1, node2)
            
    Ori_len=edge_att[node_key]['length']
            
    Ori_dist=ox.distance.euclidean_dist_vec(map_network.nodes[node1]['y'],map_network.nodes[node1]['x'],map_network.nodes[node2]['y'],map_network.nodes[node2]['x'])
            
    Uni_conv=Ori_len/Ori_dist
            
    n1_nn_dist=ox.distance.euclidean_dist_vec(map_network.nodes[node1]['y'],map_network.nodes[node1]['x'],map_network.nodes[nn]['y'],map_network.nodes[nn]['x'])
            
    n1_nn_len=n1_nn_dist*Uni_conv
    nn_n2_len=Ori_len-n1_nn_len
            
            
    edge_att_n1_nn=edge_att[node_key].copy()
    edge_att_n1_nn['length']=n1_nn_len
            
    edge_att_nn_n2=edge_att[node_key].copy()
    edge_att_nn_n2['length']=nn_n2_len
            
            
    map_network.add_edge(node1, nn)
    map_network.add_edge(nn, node2)
            
    attrs = {(node1, nn ,0): edge_att_n1_nn, (nn, node2, 0): edge_att_nn_n2}
            
    nx.set_edge_attributes(map_network, attrs)

    return nn



def get_or_add_nodes(G,df):
    global ID_loc
    ID_loc={}
    global locations
    locations=df.copy()
    locations['y']=1
    locations['x']=1

    node_ID=[]
 
    node_list=list(G.nodes(data=True))
    # Get Part, If the corresponding node is present then get the coordinates
    for i in range(len(locations)):

        location = geocode(locations.iloc[i,0], provider="nominatim" , user_agent = 'my_request')
        point=location.iloc[0,0]
        for j in node_list:
            if j[1]["y"]==point.y and j[1]["x"]==point.x:
                locations.iloc[i,-2]=point.y
                locations.iloc[i,-1]=point.x
                node_ID.append(j[0])
                ID_loc[j[0]]=locations.iloc[i,0].split(",")[0]
    # Add part, For those that doesnt have a node, create

    for i in range(len(locations)):
        if locations.iloc[i,-1] == 1:
            
            location = geocode(locations.iloc[i,0], provider="nominatim" , user_agent = 'my_request')
            point=location.iloc[0,0]

            G.add_node(locations.iloc[i,0].split(",")[0], y=point.y,x=point.x)
            
            n_edge = ox.distance.nearest_edges(G, point.x, point.y, return_dist=False)
            node1=n_edge[0]
            node2=n_edge[1]
            edge_key=n_edge[2]
            
            nn=df.iloc[i,0].split(",")[0]
            
            edge_att=G.get_edge_data(node1, node2)
            
            Ori_len=edge_att[edge_key]['length']
            
            Ori_dist=ox.distance.euclidean_dist_vec(G.nodes[node1]['y'],G.nodes[node1]['x'],G.nodes[node2]['y'],G.nodes[node2]['x'])
            
            Uni_conv=Ori_len/Ori_dist
            
            n1_nn_dist=ox.distance.euclidean_dist_vec(G.nodes[node1]['y'],G.nodes[node1]['x'],G.nodes[nn]['y'],G.nodes[nn]['x'])
            
            n1_nn_len=n1_nn_dist*Uni_conv
            nn_n2_len=Ori_len-n1_nn_len
            
            n1_attribute=G.nodes[node1].copy()
            del n1_attribute['y']
            del n1_attribute['x']
            
            nx.set_node_attributes(G, {nn:n1_attribute})
            
            
            edge_att_n1_nn=edge_att[edge_key].copy()
            edge_att_n1_nn['length']=n1_nn_len
    
            
            edge_att_nn_n2=edge_att[edge_key].copy()
            edge_att_nn_n2['length']=nn_n2_len


            
            G.add_edge(node1, nn)
            G.add_edge(nn, node2)
            
            attrs = {(node1, nn ,0): edge_att_n1_nn, (nn, node2, 0): edge_att_nn_n2}
            
            nx.set_edge_attributes(G, attrs)
            
            locations.iloc[i,1]=point.x
            locations.iloc[i,2]=point.y
            node_ID.append(nn)
            
    return node_ID




def add_demand(LID,M):

    for i in range(len(dataframe)):
        M.nodes[LID[i]]['demand_water'] = dataframe.iloc[i,1]
        M.nodes[LID[i]]['demand_food'] = dataframe.iloc[i,2]
        M.nodes[LID[i]]['demand_other'] = dataframe.iloc[i,3]
        
        M.nodes[LID[i]]['demand'] = M.nodes[LID[i]]['demand_water'] +\
                                                M.nodes[LID[i]]['demand_food'] +\
                                                M.nodes[LID[i]]['demand_other']

def display_routes(G):
    global nodeID_to_address
    global cust_loc #list of only customer locations in order
    cust_loc={}
    nodeID_to_address={}
    Dpath={}
    for V, Loc_net in Loc_nodes.items():
        Dpath[V]=[]
        Loc_list=list(Loc_net.nodes)
        depot_index=Loc_list.index("depot")
        LP=Loc_list[depot_index: ]
        FP=Loc_list[ :depot_index]
        Loc_list.clear()
        Loc_list=LP+FP
        cust_loc[V]=Loc_list
        cust_loc[V].append('depot')
        #Loc_list=list of locations in order

        path_pairs=dict(nx.all_pairs_dijkstra(G))

        for i in range(len(Loc_list)):
            if i+1 < len(Loc_list):
                j=i+1
                point1=Loc_list[i]
                point2=Loc_list[j]
                
                list_of_inbet_nodes=path_pairs[point1][1][point2]
                
                Dpath[V].extend(list_of_inbet_nodes)
            else:
                j=0
                point1=Loc_list[i]
                point2=Loc_list[j]

                list_of_inbet_nodes=path_pairs[point1][1][point2]
                
                Dpath[V].extend(list_of_inbet_nodes)

        Dpath[V]=[key for key, _group in groupby(Dpath[V])]
        
        #rev geo loc 
        for i in range(len(Dpath[V])):
            if type(Dpath[V][i]) == int:
                lat=G.nodes[Dpath[V][i]]['y']
                long=G.nodes[Dpath[V][i]]['x']

                rev_loc = geolocator.reverse("{},{}".format(lat,long))
                addrs=rev_loc.address.split(",")[0]
                #nodeID_to_address[Dpath[V][i]]=addrs
                Dpath[V][i]=addrs

                lst=locations.transpose().values.tolist()
                if lat in lst[4]:
                    ind=lst[4].index(lat)
                    addrs=lst[0][ind].split(",")[0]
                    nodeID_to_address[Dpath[V][i]]=addrs
                    Dpath[V][i]=addrs

            elif type(Dpath[V][i]) == str and not Dpath[V][i].isdigit():
                Dpath[V][i] = Dpath[V][i]
            else:
                Dpath[V][i]=int(Dpath[V][i])
                lat=G.nodes[Dpath[V][i]]['y']
                long=G.nodes[Dpath[V][i]]['x']

                rev_loc = geolocator.reverse("{},{}".format(lat,long))
                addrs=rev_loc.address.split(",")[0]
                #nodeID_to_address[Dpath[V][i]]=addrs
                Dpath[V][i]=addrs

                lst=locations.transpose().values.tolist()
                if lat in lst[4]:
                    ind=lst[4].index(lat)
                    addrs=lst[0][ind].split(",")[0]
                    nodeID_to_address[Dpath[V][i]]=addrs
                    Dpath[V][i]=addrs
        
        Dpath[V]=[key for key, _group in groupby(Dpath[V])]
        

    return Dpath

def display_routes_drone(G):
    global nodeID_to_address
    nodeID_to_address={}
    reg_drone={}
    for V, Loc_net in Loc_nodes_drone.items():
        Loc_list_drone=list(Loc_net.nodes)
        depot_index=Loc_list_drone.index("depot")
        LP=Loc_list_drone[depot_index: ]
        FP=Loc_list_drone[ :depot_index]
        Loc_list_drone.clear()
        Loc_list_drone=LP+FP
        reg_drone[V]=Loc_list_drone

    DDPath={}
    for key,i in reg_drone.items():
        DDPath[key]=[]
        for k in i:
            if type(k)== str and not k.isdigit():
                DDPath[key].append(k)
            else:
                k=int(k)
                lat=G.nodes[k]['y']
                long=G.nodes[k]['x']
                rev_loc = geolocator.reverse("{},{}".format(lat,long))
                addrs=rev_loc.address.split(",")[0]

                lst=locations.transpose().values.tolist()
                if lat in lst[4]:
                    ind=lst[4].index(lat)
                    addrs=lst[0][ind].split(",")[0]
                    nodeID_to_address[k]=addrs

                DDPath[key].append(addrs)
    return DDPath
    


def get_distance_drone(G,locations):
    dist={}
    node_address=list(nodeID_to_address.values())
    node_address_key=list(nodeID_to_address.keys())
    for i in range(len(locations)):
        if i+1 < len(locations):
            j=i+1
            if locations[i] in node_address:
                ind=node_address.index(locations[i])
                locations[i]=node_address_key[ind]
            if locations[j] in node_address:
                ind=node_address.index(locations[j])
                locations[j]=node_address_key[ind]

            p1=(G.nodes[locations[i]]['y'],G.nodes[locations[i]]['x'])
            p2=(G.nodes[locations[j]]['y'],G.nodes[locations[j]]['x'])


            
            dist[f"{nodeID_to_address[locations[i]] if locations[i] in nodeID_to_address else locations[i] } --> {nodeID_to_address[locations[j]] if locations[j] in nodeID_to_address else locations[j] }"] =int( _cost_between_nodes_haversine(p1,p2,"start","end"))
    
    return dist
    
def get_distance(G,veh_ind):
    path_pairs_length=dict(nx.all_pairs_dijkstra_path_length(G,weight='length'))
    dist={}
    loc_lst=cust_loc[veh_ind]

    for j in range(len(loc_lst)):

        if j+1 < len(loc_lst):
            k=j+1
            dist[f"{loc_lst[j] if type(loc_lst[j]) == str else ID_loc[loc_lst[j]]} --> {loc_lst[k] if type(loc_lst[k]) == str else ID_loc[loc_lst[k]]}"] =int( path_pairs_length[loc_lst[j]][loc_lst[k]] )
  
    return dist