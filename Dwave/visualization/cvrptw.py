# [Import]
import os
from functools import partial
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import cos, asin, sqrt, pi
from geopy.geocoders import Nominatim
from pandas import DataFrame
geolocator = Nominatim(user_agent="MyApp")
import networkx as nx
from scipy.spatial import cKDTree
import osmnx as ox
import folium
import pandas as pd
import seaborn as sns
from itertools import groupby
from geopandas.tools import geocode
# [Import]
#-----------------------------------------------------------------------------------------------------   
# [Coordinate function]
def coor(data):
    location = geocode(data, provider="nominatim" , user_agent = 'my_request')
    point=location.iloc[0,0]
    return [point.y,point.x]
# [Coordinate function]
#-----------------------------------------------------------------------------------------------------   
# [data]
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    return data
# [data]
#-----------------------------------------------------------------------------------------------------   
# [Haversine function]
def Haversine(position_1, position_2):
    # haversine distance for drones not manhattan distance
    p = pi/180
    a = 0.5 - cos((position_2[0]-position_1[0])*p)/2 + cos(position_1[0]*p) * cos(position_2[0]*p) * (1-cos((position_2[1]-position_1[1])*p))/2
    return  12742 * asin(sqrt(a)) * 1000 #2*R*asin... in m
# [Haversine function]
#-----------------------------------------------------------------------------------------------------   
# [Street distance function]
def street_distance(G,fn,tn):
    len_path = nx.dijkstra_path_length(G,fn,tn,weight="length")
    return abs(len_path)
# [Street distance function]
#-----------------------------------------------------------------------------------------------------   
# [Distance matrix and manager]
def create_distance_evaluator(data):
    """Creates callback to return distance between points."""
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in range(data['num_locations']):
        _distances[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            else:
                _distances[from_node][to_node] = (Haversine(
                    data['locations'][from_node], data['locations'][to_node]))

    def distance_evaluator(manager, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return distance_evaluator
# [Distance matrix and manager]
#-----------------------------------------------------------------------------------------------------   
# [Distance matrix and manager for street]
def create_distance_evaluator_street(G,data):
    """Creates callback to return distance between points."""
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in range(data['num_locations']):
        _distances[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            else:
                _distances[from_node][to_node] = (street_distance(G,from_node,to_node))

    def distance_evaluator(manager, from_node, to_node):
        """Returns the manhattan distance between the two nodes"""
        return _distances[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return distance_evaluator
# [Distance matrix and manager for street]
#-----------------------------------------------------------------------------------------------------   
# [Demand manager]
def create_demand_evaluator(data):
    """Creates callback to get demands at each location."""
    _demands = data['demands']

    def demand_evaluator(manager, node):
        """Returns the demand of the current node"""
        return _demands[manager.IndexToNode(node)]

    return demand_evaluator
# [Demand manager]
#-----------------------------------------------------------------------------------------------------   
# [Capacity manager]
def cinf(routing, data, demand_evaluator_index):
    """Adds capacity constraint"""
    capacity = 'Capacity'
    routing.AddDimension(
        demand_evaluator_index,
        0,  # null capacity slack
        data['cinf'],
        True,  # start cumul to zero
        capacity)
# [Capacity manager]
#-----------------------------------------------------------------------------------------------------   
# [Time window and service time manager]
def create_time_evaluator(data):
    """Creates callback to get total times between locations."""

    def service_time(data, node):
        """Gets the service time for the specified location."""
        return data['demands'][node] * data['time_per_demand_unit']

    def travel_time(data, from_node, to_node):
        """Gets the travel times between two locations."""
        if from_node == to_node:
            travel_time = 0
        else:
            travel_time = Haversine(data['locations'][from_node], data[
                'locations'][to_node]) / data['vehicle_speed']
        return travel_time

    _total_time = {}
    # precompute total time to have time callback in O(1)
    for from_node in range(data['num_locations']):
        _total_time[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _total_time[from_node][to_node] = 0
            else:
                _total_time[from_node][to_node] = int(
                    service_time(data, from_node) + travel_time(
                        data, from_node, to_node))

    def time_evaluator(manager, from_node, to_node):
        """Returns the total time between the two nodes"""
        return _total_time[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return time_evaluator
# [Time window and service time manager]
#-----------------------------------------------------------------------------------------------------   
# [Time window and service time constrain]
def add_time_window_constraints(routing, manager, data, time_evaluator_index):
    """Add Global Span constraint"""
    time = 'Time'
    horizon = 1440
    routing.AddDimension(
        time_evaluator_index,
        horizon,  # allow waiting time
        horizon,  # maximum time per vehicle
        False,  # don't force start cumul to zero since we are giving TW to start nodes
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
    # Add time window constraints for each vehicle start node
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # Warning: Slack var is not defined for vehicle's end node
        #routing.AddToAssignment(time_dimension.SlackVar(self.routing.End(vehicle_id)))
# [Time window and service time constrain]
#-----------------------------------------------------------------------------------------------------   
# [Print solution]
def print_solution(manager, routing, assignment):  # pylint:disable=too-many-locals
    """Prints assignment on console"""
    global routes
    routes={}
    st1=""
    st2=""
    tpo={}
    st1 += f'Objective: {assignment.ObjectiveValue()}'
    obj=assignment.ObjectiveValue()
    time_dimension = routing.GetDimensionOrDie('Time')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    total_distance = 0
    total_load = 0
    total_time = 0
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        r=""
        plan_output = '**Route for vehicle {}:**  \n'.format(vehicle_id+1)
        distance =  0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            time_var = time_dimension.CumulVar(index)
            slack_var = time_dimension.SlackVar(index)
            plan_output += 'Time({1},{2}) mins **[ {0} ]** Wait({3},{4}) mins --> '.format(
                manager.IndexToNode(index),
                assignment.Min(time_var),
                assignment.Max(time_var),
                assignment.Min(slack_var), 
                assignment.Max(slack_var))
            r+="{},".format(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle( previous_index, index,
                                                     vehicle_id )
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        slack_var = time_dimension.SlackVar(index)
        plan_output += 'Time({1},{2}) mins **[ {0} ]**  \n'.format(
            manager.IndexToNode(index),
            assignment.Min(time_var), 
            assignment.Max(time_var))
        r+="{}".format(manager.IndexToNode(index))
        plan_output += "  \n"
        plan_output += '**Distance of the route:** {0} Meters  \n'.format(distance)
        plan_output += '**Demand of the route:** {} Items  \n'.format(
            assignment.Value(load_var))
        plan_output += '**Time of the route:** {} Minutes  \n'.format(
            assignment.Value(time_var))
        plan_output += "   \n"
        plan_output += " -----------------------"
        plan_output += "   \n"
        
        total_distance +=  distance
        total_load += assignment.Value(load_var)
        total_time += assignment.Value(time_var)
        routes[vehicle_id]=r
        tpo[vehicle_id]=plan_output
        
    st2 += "  \n" 
    st2 += '**Total Distance traveled by all vehicles:** {0} Meters  \n'.format(total_distance)
    st2 += '**Total Demand delivered by all vehicles:** {} Items  \n'.format(total_load)
    st2 += '**Total Time taken for all vehicles:** {0} Minutes  \n'.format(total_time)
    
    return(tpo,st1,st2)
# [Print solution]
#-----------------------------------------------------------------------------------------------------   
# [Graph network function]
def create_graph_network(depot_address,location_list):
    G = ox.graph_from_address(depot_address, dist=1000, network_type="drive")
    G= ox.utils_graph.get_largest_component(G, strongly=True)
    for i in range(len(location_list)):
        G.add_node(i, y=location_list[i][0],x=location_list[i][1])
    return G
# [Graph network function]
#-----------------------------------------------------------------------------------------------------   
# [dummy]
def show_locations_on_initial_map_cvrptw(map_network, location_list, dataframe):
    dummy_row = pd.DataFrame({'Location Address':'Depot', 'time window open':0, 'time window close':0,'demands':0},index =[0])

    dataframe = pd.concat([dummy_row, dataframe]).reset_index(drop = True)

    depot_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "Depot Location.png"))
    force_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "Force Location.png"))
    depot_icon = folium.CustomIcon(depot_icon_path, icon_size=(32, 37))
    
    folium_map = ox.plot_graph_folium(map_network, opacity=0.0)

    folium.Marker(
        (location_list[0][0], location_list[0][1]),
        tooltip=folium.map.Tooltip(text="Depot", style="font-size: 1.4rem;"), icon=depot_icon
    ).add_to(folium_map)

    c=1
    for i in location_list:
        if location_list.index(i) != 0:
            location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
            folium.Marker(i,tooltip=folium.map.Tooltip(text=f"{dataframe.iloc[c,0].split(',')[0]}\
                                                        <br>Time Window [{dataframe.iloc[c,1]}-{dataframe.iloc[c,2]}]\
                                                            <br>Demand {dataframe.iloc[c,3]}", style="font-size: 1.4rem;"),
                                                            icon=location_icon).add_to(folium_map)
            c+=1
            
    
    folium.plugins.Fullscreen().add_to(folium_map)
    return folium_map
# [dummy]
#-----------------------------------------------------------------------------------------------------   
# [dummy]    
def plot_solution_on_map(map_network, folium_map, location_list):

    palette = sns.color_palette("colorblind", len(routes)).as_hex()
    
    for i in routes:
        #converting dict to list
        routes[i]=list(map(int,list(routes[i].split(","))))
        for k in range(len(routes[i])):
            # iterate add edges
            if k+1 < len(routes[i]):
                l=k+1
                pos1=location_list[routes[i][k]]
                pos2=location_list[routes[i][l]]
                length=Haversine(pos1,pos2)
                map_network.add_edge(routes[i][k], routes[i][l], length=length)
    
    
    for i in routes:
        route_color = palette.pop()
        route_map = ox.plot_route_folium(map_network, route=routes[i], route_map=folium_map, fit_bounds=False, color=route_color, popup_attribute='length',opacity=0.8)

    return route_map
# [dummy]
#-----------------------------------------------------------------------------------------------------   
# [dummy]
def plot_solution_on_map_street(map_network, folium_map, location_list):

    palette = sns.color_palette("colorblind", len(routes)).as_hex()
    paths = dict(nx.all_pairs_dijkstra(map_network))
    detailed_paths={}

    for i in routes:
        detailed_paths[i]=[]
        #converting dict to list
        routes[i]=list(map(int,list(routes[i].split(","))))
        for j in range(len(routes[i])):
            if j+1 < len(routes[i]):
                l=j+1
                lst=paths[routes[i][j]][1][routes[i][l]]
                detailed_paths[i].extend(lst)

    
    for i in detailed_paths:
        route_color = palette.pop()
        detailed_paths[i]=list(map(int,detailed_paths[i]))
        route_map = ox.plot_route_folium(map_network, route=detailed_paths[i], route_map=folium_map, fit_bounds=False, color=route_color, popup_attribute='length',opacity=0.8)

    return route_map
# [dummy]
#-----------------------------------------------------------------------------------------------------   
# [Connect edge function]         
def connect_edges(G,location_list):
    for i in range(len(location_list)):
        n_edge = ox.distance.nearest_edges(G, location_list[i][0], location_list[i][1], return_dist=False)
        node1=n_edge[0]
        node2=n_edge[1]
        edge_key=n_edge[2]

        pos1=(G.nodes[node1]['y'],G.nodes[node1]['x'])
            
        posnn=(location_list[i][0],location_list[i][1])
        length_node1_nn=Haversine(pos1,posnn)
        
        
        pos2=(G.nodes[node2]['y'],G.nodes[node2]['x'])
        lenght_nn_node2= Haversine(posnn,pos2)

        nn=i
        
        attrs = {(node1, nn ,0): {'length':length_node1_nn}, (nn, node2, 0): {'length':lenght_nn_node2}}
        
        G.add_edge(node1, nn)
        G.add_edge(nn, node2)
        nx.set_edge_attributes(G, attrs)
# [Connect edge function]         
#-----------------------------------------------------------------------------------------------------   
       


            
    
    