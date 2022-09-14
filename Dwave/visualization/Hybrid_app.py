# [Import]
from collections import defaultdict
from jinja2 import Template
import streamlit as st
import pandas as pd
from itertools import groupby
import osmnx as ox
import copy
from streamlit_folium import folium_static
from map_network import (display_routes_drone, set_address,get_df, generate_solution_map_street_network,
                         generate_solution_map_drone_network,
                         generate_mapping_information,
                         show_locations_on_initial_map,
                         add_demand,display_routes,
                         RoutingProblemParameters,get_distance,get_distance_drone)
# [Import]

# [Map parameters]
map_width, map_height = 870, 500
# [Map parameters]




def main_Hybrid(vehicle_type,num_vehicles,cap_list,title,uploaded_file,time_limit,):
    # [Render Solution stats function]
    def render_solution_stats(problem_size = None, 
                            search_space = None,
                            wall_clock_time = None,
                            forces = None,
                            vehicles= None) -> str:
        with open("app_customization/solution_stats.html") as stats:
            template = Template(stats.read())
            return template.render(
                problem_size = problem_size,
                search_space = search_space,
                wall_clock_time = wall_clock_time,
                num_forces = forces,
                num_vehicles = vehicles
            )
    # [Render Solution stats function]

    def render_solution_cost(solution_cost_information: dict, total_cost_information: dict):
        with open("app_customization/solution_cost.html") as cost:
            template = Template(cost.read())
            return template.render(
                solution_cost_information=solution_cost_information,
                total_cost_information=total_cost_information
            )

    def render_path(ID,RTS):
        with open("app_customization/path.html") as paths:
            template = Template(paths.read())
            return template.render(
                ID=ID,
                RTS=RTS
            )

    def render_distance(ID,distance):
        with open("app_customization/distance.html") as dist:
            template = Template(dist.read())
            return template.render(
                ID=ID,
                distance=distance
            )

    def render_dataframe(lst):
        with open("app_customization/dataframe.html") as paths:
            template = Template(paths.read())
            return template.render(
                lst=lst
            )

    def render_cap(cap_list):
        with open("app_customization/cap.html") as cap:
            template = Template(cap.read())
            return template.render(
                cap_list=cap_list
            )


    with open("app_customization/stylesheet.html") as css:
        stylesheet = css.read()

    st.write(stylesheet, unsafe_allow_html=True)


    #Capacity
    

    
    cap_dict = {int(int(k)-1):v for k,v in cap_list.items()}


    ## city
    set_address(title)


    ## client location
    

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        get_df(dataframe)
    else:
        dataframe = pd.DataFrame()
        get_df(dataframe)


    sampler_type = "Quantum Hybrid (DQM)"
    

    num_clients=len(dataframe)


    map_network, depot_id, force_locations = generate_mapping_information(num_clients)

    add_demand(force_locations,map_network)


    initial_map = show_locations_on_initial_map(map_network, depot_id, force_locations)

    routing_problem_parameters = RoutingProblemParameters(
        folium_map=initial_map, map_network=map_network, depot_id=depot_id,
        client_subset=force_locations, num_clients=num_clients,
        num_vehicles=num_vehicles, sampler_type=sampler_type, time_limit=time_limit,cap_dict=cap_dict
    )


    #folium_static(initial_map, width=map_width, height=map_height)


    def render_app(solution_and_map_function: callable):
            result_list=[]

            response = solution_and_map_function(routing_problem_parameters)
            
            
            problem_size = num_vehicles * num_clients,
            search_space = "{:.2e}".format(num_vehicles ** num_clients),
            wall_clock_time = "{:.3f}".format(response["wall_clock_time"]),
            forces = num_clients,
            vehicles = num_vehicles
            
            sol_stat=[problem_size,search_space,wall_clock_time,forces,vehicles]

            folium_static(
                response["map"],
                width=map_width,
                height=map_height
            )
            result_list.append(sol_stat)

            solution_cost_information  = dict(sorted(response["solution_cost"].items()))
            total_cost = defaultdict(int)
            for _, cost_info_dict in solution_cost_information.items():
                for key, value in cost_info_dict.items():
                    total_cost[key] += value
            sol_cost=[solution_cost_information,total_cost]
            
            
            
            

            result_list.append(sol_cost)          
        

            routes_per_vehicle=[]
            distance_per_vehicle=[]

            if vehicle_type == 'Trucks':
                path=display_routes(map_network)

                for index in range(len(path)):
                    distance=get_distance(map_network,index)
                    
                    rd=(index+1,distance)
                    distance_per_vehicle.append(rd)

                    temp_dataframe=dataframe.copy()
                    temp_dataframe=list(temp_dataframe.iloc[:,0])
                    temp_dataframe.append("depot")
                    for i in range(len(temp_dataframe)):
                        j = temp_dataframe[i].split(",")[0]
                        temp_dataframe[i]=j
                    
                    for k in range(len(path[index])):
                        if path[index][k] in temp_dataframe:
                            temp=list(path[index][k])
                            temp.insert(0,"[")
                            temp.append("]")
                            temp= "".join(temp)
                            path[index][k]=temp

                    rp=(index+1,path[index])
                    routes_per_vehicle.append(rp)

                result_list.append(distance_per_vehicle)
                
    

                result_list.append(routes_per_vehicle)

                return result_list
                

            if vehicle_type == 'Delivery Drones':
                path=display_routes_drone(map_network)
                
                for index in range(len(path)):
                    if path[index][0] != "depot":
                        path[index].insert(0,"depot")

                    if path[index][-1] != "depot":
                        path[index].append("depot")
        
                    locations=copy.deepcopy(path[index])

                    dist=get_distance_drone(map_network,locations)
                    rd=(index+1,dist)
                    distance_per_vehicle.append(rd)
                    rp=(index+1,path[index])
                    routes_per_vehicle.append(rp)



                result_list.append(distance_per_vehicle)
                
                

                result_list.append(routes_per_vehicle)

                return result_list


    if vehicle_type == 'Delivery Drones':
        try:
            return (render_app(generate_solution_map_drone_network),dataframe)
        except ValueError:
            return (["No optimal solution is possible for the current vehicles. Please increase the number of vehicles and (or) capacity."])
    else:
        try:
            return (render_app(generate_solution_map_street_network),dataframe)
        except ValueError:
            return (["No optimal solution is possible for the current vehicles. Please increase the number of vehicles and (or) capacity."])



    

    # streamlit css







