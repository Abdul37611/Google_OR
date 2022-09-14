

# app.py <-- OR_app.py <-- cvrptw.py



# [Import]
from collections import defaultdict
from jinja2 import Template
import streamlit as st
import pandas as pd
from itertools import groupby
import osmnx as ox
import copy
from streamlit_folium import folium_static
import streamlit.components.v1 as components
from functools import partial
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
# [Import]
#------------------------------------------------------------------------------------------------------
# [Import for Google OR]
from OR_app import main_OR
# [Import for Google OR]
#------------------------------------------------------------------------------------------------------

from Hybrid_app import main_Hybrid

from Classical_app import main_Classical


# [Streamlit styling]
st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

hide_expander_border = """
<style>
ul.streamlit-expander {
    border: 0 !important;
</style>
"""
st.markdown(hide_expander_border, unsafe_allow_html=True)
# [Streamlit CSS]





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








#------------------------------------------------------------------------------------------------------
# [UI - Choose algorithm and render]
sampler_type = st.sidebar.radio("Choose Algorithm",("Quantum Hybrid (DQM)", "Classical (K-Means)","Google OR"))

if sampler_type == "Google OR":
    # [UI]
    depot_address = st.sidebar.text_input('Depot Address', 'Cambridge St, Rockhampton QLD 4700, Australia')

    uploaded_file = st.sidebar.file_uploader("Location Input")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
    else:
        st.sidebar.write("> Please upload the locations file")
        no_data={'Location Address':['Depot'],'time window open':[0],'time window close':[0],'demands':[0]}
        dataframe=pd.DataFrame(no_data)

    time_per_demand_unit=st.sidebar.number_input(label="Time to unload 1 unit of demand (Mins)",value=0,step=1)
    
    vehicle_type = st.sidebar.radio(
    "Vehicle Type", ["Delivery Drones", "Trucks"]
    )

    num_vehicles = st.sidebar.slider(f"Number of {vehicle_type} to deploy", 1, 10, 2, 1)

    run_button = st.sidebar.button("Run Optimization", key="run")
    # [UI]

    # [Run]
    if run_button:
        solution=main_OR(depot_address,dataframe,time_per_demand_unit,vehicle_type,num_vehicles)

        if len(solution)>1:
            val=solution[0]
            temp_dataframe=solution[1]

            st.write("> Time(x,y) are in minutes, 0 mins corresponds to 00:00 and 1440 mins to 24:00.  x indicates start time and y indicates end time, period within which the location should be served")
            st.write("> Wait(x,y) are in minutes, x indicates minimum wait time in munutes and y indicates maximum wait time in minutes")
            st.write("----------------------------")
            st.write(val[1])
            for key in val[0]:
                s=val[0][key]
                a=s.split(" ")
                a=np.array(a)
                for i in range(len(a)):
                    if a[i].isdigit():
                        if int(a[i])<=len(temp_dataframe):
                            if i!=0:
                                if a[i-1]!='' and a[i-1][0] == "r":
                                    break
                            a[i]=temp_dataframe.iloc[int(a[i]),0].split(",")[0]
                
                s = ' '.join(a)
                st.markdown(s)
            st.markdown(val[2])
        else:
            st.write(solution[0])
    # [Run]
########################################################################################################
elif sampler_type == "Quantum Hybrid (DQM)":
    vehicle_type = st.sidebar.radio(
    "Vehicle Type", ["Delivery Drones", "Trucks"]
    )

    num_vehicles = st.sidebar.slider(
            f"Number of {vehicle_type} to deploy", 1, 10, 1, 1
    )

    cap_list={}
    for i in range(num_vehicles):
        ip=st.sidebar.number_input(label=f"Capacity of {vehicle_type[:-1]} {i+1}",value=0,step=1)
        cap_list[i+1]=ip

    title = st.sidebar.text_input('Depot Address', 'Cambridge St, Rockhampton QLD 4700, Australia')
    disp=title.split(",")[1]
    st.header(f"The current city is {disp}")

    uploaded_file = st.sidebar.file_uploader("Location Input")
    
    time_limit = st.sidebar.number_input("Optimization time limit", min_value=5.0, value=5.0)
    run_button = st.sidebar.button("Run Optimization", key="run")

    if run_button:
        solution=main_Hybrid(vehicle_type,num_vehicles,cap_list,title,uploaded_file,time_limit)
        if len(solution) == 1:
            st.write(solution[0])
        else:
            results=solution[0]
            dataframe=solution[1]

            solution_stats=results[0]
            solution_costs=results[1]
            distance=results[2]
            routes=results[3]

            num_cus=len(dataframe)
            r_solution_stats = render_solution_stats(
                forces = num_cus,
                vehicles = solution_stats[4]
            )
            st.write(r_solution_stats, unsafe_allow_html=True)

            r_solution_cost = render_solution_cost(
                        solution_cost_information=solution_costs[0],
                        total_cost_information=solution_costs[1]
                        )
            st.write(r_solution_cost,unsafe_allow_html=True )

            for i in range(len(distance)):
                rd=render_distance(distance[i][0],distance[i][1])
                with st.expander(f"See distance travelled between customers for vehicle {i+1}"):
                    st.write(rd, unsafe_allow_html=True)

            rc=render_cap(cap_list)
            st.write(rc,unsafe_allow_html=True)

            for j in routes:
                rp=render_path(j[0],j[1])
                st.write(rp, unsafe_allow_html=True)


######################################################################################################## 
elif sampler_type == "Classical (K-Means)":
    vehicle_type = st.sidebar.radio(
    "Vehicle Type", ["Trucks","Delivery Drones"]
    )

    num_vehicles = st.sidebar.slider(
            f"Number of {vehicle_type} to deploy", 1, 10, 1, 1
    )

    cap_list={}
    for i in range(num_vehicles):
        ip=st.sidebar.number_input(label=f"Capacity of {vehicle_type[:-1]} {i+1}",value=0,step=1)
        cap_list[i+1]=ip

    title = st.sidebar.text_input('Depot Address', 'Cambridge St, Rockhampton QLD 4700, Australia')
    disp=title.split(",")[1]
    st.header(f"The current city is {disp}")

    uploaded_file = st.sidebar.file_uploader("Location Input")
    
    time_limit = st.sidebar.number_input("Optimization time limit", min_value=5.0, value=5.0)
    run_button = st.sidebar.button("Run Optimization", key="run")

    if run_button:
        solution=main_Classical(vehicle_type,num_vehicles,cap_list,title,uploaded_file,time_limit)
        
        if len(solution) == 1:
            st.write(solution[0])
        else:
            results=solution[0]
            dataframe=solution[1]

            solution_stats=results[0]
            solution_costs=results[1]
            distance=results[2]
            routes=results[3]

            num_cus=len(dataframe)
            r_solution_stats = render_solution_stats(
                forces = num_cus,
                vehicles = solution_stats[4]
            )
            st.write(r_solution_stats, unsafe_allow_html=True)

            r_solution_cost = render_solution_cost(
                        solution_cost_information=solution_costs[0],
                        total_cost_information=solution_costs[1]
                        )
            st.write(r_solution_cost,unsafe_allow_html=True )

            for i in range(len(distance)):
                rd=render_distance(distance[i][0],distance[i][1])
                with st.expander(f"See distance travelled between customers for vehicle {i+1}"):
                    st.write(rd, unsafe_allow_html=True)

            rc=render_cap(cap_list)
            st.write(rc,unsafe_allow_html=True)

            for j in routes:
                rp=render_path(j[0],j[1])
                st.write(rp, unsafe_allow_html=True)
# [UI - Choose algorithm and render]



