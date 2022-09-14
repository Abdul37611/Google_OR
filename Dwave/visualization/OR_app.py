# [Import]
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from functools import partial
import numpy as np
from jinja2 import Template
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from cvrptw import(coor,create_data_model,create_distance_evaluator,create_distance_evaluator_street,
                   create_demand_evaluator,cinf,
                   create_time_evaluator,add_time_window_constraints,
                   print_solution,create_graph_network,show_locations_on_initial_map_cvrptw,
                   plot_solution_on_map,connect_edges,plot_solution_on_map_street)
# [Import]
#------------------------------------------------------------------------------------------------------
def main_OR(depot_address,dataframe,time_per_demand_unit,vehicle_type,num_vehicles):

    # [Display the depot address]
    disp=depot_address.split(",")[1]
    st.header(f"The current city is {disp}")
    # [Display the depot address]
#------------------------------------------------------------------------------------------------------
    # [Get coordinates for depot]
    temp_loc_data={}
    coordinates=coor(depot_address)
    temp_loc_data["depot_coor"]=(coordinates[0],coordinates[1])
    # [Get coordinates for depot]
#------------------------------------------------------------------------------------------------------
    # [Get coordinates for customer locations]
    location_list=[]
    for i in range(len(dataframe)):
        coordinates=coor(dataframe.iloc[i,0])
        location_list.append((coordinates[0],coordinates[1]))
    temp_loc_data["locations"]=location_list
    # [Get coordinates for customer locations]
#------------------------------------------------------------------------------------------------------
    # [Combine depot and customers to a single list]
    final_location_list=[]
    final_location_list.append(temp_loc_data["depot_coor"])
    final_location_list.extend(temp_loc_data["locations"])
    # [Combine depot and customers to a single list]
#-----------------------------------------------------------------------------------------------------
    # [Get time window]
    time_windows=[]
    time_windows.append((int(0),int(0)))
    for i in range(len(dataframe)):
        time_windows.append((int(dataframe.iloc[i,1]),int(dataframe.iloc[i,2])))
    # [Get time window]
#-----------------------------------------------------------------------------------------------------
    # [Get demand]
    demands=[]
    demands.append(0)
    for i in range(len(dataframe)):
        demands.append(int(dataframe.iloc[i,3]))
    # [Get demand]
#-----------------------------------------------------------------------------------------------------
    # [Create data]
    data = create_data_model()
    # [Create data]
#-----------------------------------------------------------------------------------------------------
    # [Insert data into data]
    data['locations']=final_location_list
    data['num_locations']=len(final_location_list)
    data['time_windows']=time_windows
    data['time_per_demand_unit']=time_per_demand_unit
    data['num_vehicles']=num_vehicles
    data['cinf']=int(9999999)
    data['demands']=demands
    data['vehicle_speed'] = 83  # Travel speed: 5km/h converted in m/min
    data['depot'] = 0
    # [Insert data into data]
#-----------------------------------------------------------------------------------------------------   
    # [Create a graph network and add all the locations]
    map_network=create_graph_network(depot_address,final_location_list)
    ## [For map]
    initial_map = show_locations_on_initial_map_cvrptw(map_network, final_location_list, dataframe)
    # [Create a graph network and add all the locations]
#-----------------------------------------------------------------------------------------------------   
    # [Add depot to the dataframe]
    dummy_row = pd.DataFrame({'Location Address':'Depot,rockhampton', 'time window open':0, 'time window close':0,'demands':0},index =[0])
    temp_dataframe = pd.concat([dummy_row, dataframe]).reset_index(drop = True)
    # [Add depot to the dataframe]
#-----------------------------------------------------------------------------------------------------   
    # [Choose vehicle type and render]
    if vehicle_type == 'Delivery Drones':
        # [Set and render the OR solver]
        manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                                data['num_vehicles'], data['depot'])

        routing = pywrapcp.RoutingModel(manager)

        distance_evaluator_index = routing.RegisterTransitCallback(
            partial(create_distance_evaluator(data), manager))

        routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
            partial(create_demand_evaluator(data), manager))
        cinf(routing, data, demand_evaluator_index)

        time_evaluator_index = routing.RegisterTransitCallback(
            partial(create_time_evaluator(data), manager))
        add_time_window_constraints(routing, manager, data, time_evaluator_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(2)
        search_parameters.log_search = True
        # [Set and render the OR solver]

        # [Search the solution space]
        try:
            solution = routing.SolveWithParameters(search_parameters)
        except SystemError:
            return ("The demand cannot be fulfilled within the time limit by the current number of vehicles. Please increase the number of vehicles")
        # [Search the solution space]

        # [Print solution]
        if solution:
            val=print_solution(manager, routing, solution)
            return (val,temp_dataframe)
        else:
            return (['No optimal solution is possible for the current vehicles. Please increase the number of vehicles.']) 
        # [Print solution]

    else:
        # [Set and render the OR solver]
        connect_edges(map_network,final_location_list)

        manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                                data['num_vehicles'], data['depot'])

        routing = pywrapcp.RoutingModel(manager)

        distance_evaluator_index = routing.RegisterTransitCallback(
            partial(create_distance_evaluator_street(map_network,data), manager))

        routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

        demand_evaluator_index = routing.RegisterUnaryTransitCallback(
            partial(create_demand_evaluator(data), manager))
        cinf(routing, data, demand_evaluator_index)

        time_evaluator_index = routing.RegisterTransitCallback(
            partial(create_time_evaluator(data), manager))
        add_time_window_constraints(routing, manager, data, time_evaluator_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(2)
        search_parameters.log_search = True
        # [Set and render the OR solver]

        # [Search the solution space]
        try:
            solution = routing.SolveWithParameters(search_parameters)
        except SystemError:
            return ("The demand cannot be fulfilled within the time limit by the current number of vehicles. Please increase the number of vehicles")
        # [Search the solution space]

        # [Print solution]
        if solution:
            val=print_solution(manager, routing, solution)
            return (val,temp_dataframe)
        else:
            return (['No optimal solution is possible for the current vehicles. Please increase the number of vehicles.'])
        # [Print solution]
    # [Choose vehicle type and render]
#-----------------------------------------------------------------------------------------------------   
    

            






