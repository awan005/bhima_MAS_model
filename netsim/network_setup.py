__author__ = 'ajainf'
############################################################################################################
# =====================================================
# Imports
# =====================================================
# Package imports
# ---------------------
import geopandas as gpd
# import matplotlib
import pandas as pd

# Node imports
# ---------------------
# CK201118: +1/-2
from netsim.model_components.nodes.urban_nodes import HouseholdAgent, SlumHouseholdAgent, CommercialAgent, IndustrialAgent
from basepath_file import basepath
path = basepath

# Instantiate HouseholdAgents (PMC)
# =====================================================
def instantiate_HouseholdAgent_nodes(network_nodes, external_path):
    """Add the Households to the nodes list """
    print()
    user_params_file_path = path + r'/modules/urban_module/Inputs/human_user_parameters_ck201118.xlsx'

    agent_subset = pd.read_excel(user_params_file_path)["agent_subset"][0]
    excel_file_path = external_path
    human_data = pd.read_excel(excel_file_path)

    print("\nHuman agents created from following excel files: {}".format(excel_file_path))
    print("\nModel region (cell subset) selected: {}".format(agent_subset))

    print("\nLoading agent parameters ...")
    count = len(network_nodes)
    for k, v in human_data.iterrows():
        if (k % 10000) == 0:
            print(" Read {} rows".format(k))
        _keyword_inputs = human_data.loc[k].to_dict()
        _keyword_inputs["x"] = _keyword_inputs["X"]
        _keyword_inputs["y"] = _keyword_inputs["Y"]
        # _keyword_inputs["is_pop_2050"] = param_is_pop_2050
        if (_keyword_inputs["excluded"] == 0) and (_keyword_inputs[agent_subset] == 1):

            # CK201118: 1 line changed
            if True: #(_keyword_inputs["population2015"] > 0) and (_keyword_inputs["hh_size"] > 0):
                _keyword_inputs["name"] = str(k) + "_hh"
                network_nodes.append(HouseholdAgent(**_keyword_inputs))
                network_nodes[count].node_type = "HouseholdAgent"
                network_nodes[count].colour = 'black'
                network_nodes[count].size = 'tiny'
                count = count + 1
                _keyword_inputs["name"] = str(k) + "_sl"
                network_nodes.append(SlumHouseholdAgent(**_keyword_inputs))
                network_nodes[count].node_type = "SlumHouseholdAgent"
                network_nodes[count].colour = 'black'
                network_nodes[count].size = 'tiny'
                count = count + 1

            # CK201118: 1 line changed
            if True: #_keyword_inputs["co_units"] > 0:
                _keyword_inputs["name"] = str(k) + "_co"
                network_nodes.append(CommercialAgent(**_keyword_inputs))
                network_nodes[count].node_type = "CommercialAgent"
                network_nodes[count].colour = 'black'
                network_nodes[count].size = 'tiny'
                count = count + 1

            # CK201118: 1 line changed
            if True: #_keyword_inputs["it_units"] > 0:
                _keyword_inputs["name"] = str(k) + "_it"
                network_nodes.append(IndustrialAgent(**_keyword_inputs))
                network_nodes[count].node_type = "IndustrialAgent"
                network_nodes[count].colour = 'black'
                network_nodes[count].size = 'tiny'
                count = count + 1
    print("Agent parameter loading complete")


# Register Nodes
# ---------------------
def register_nodes_by_type(net, _type, _institution):
    """Assign all nodes of a particular type to a particular institution """
    for n in net.nodes:
        if n.node_type == _type:  # for convention made the type same as class to find it easier
            _institution.add_nodes(n)
            n.institution_names = [str(_institution.name)]  # attribute to keep track of institution
