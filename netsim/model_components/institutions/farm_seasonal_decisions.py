__author__ = 'ankunwang'

import os
import sys

from modules.ag_seasonal.ag_pyomo_seasonal import ag_optimization
from modules.ag_seasonal.ag_pyomo_seasonal_historical import ag_optimization_historical
from modules.ag_seasonal.ag_pyomo_seasonal_solar import ag_optimization_solar 
from modules.ag_seasonal.ag_pyomo_seasonal_elec import ag_optimization_elec

from netsim.model_components.institutions._pune_institution import PuneInstitution

class FarmSeasonalDecisionMaker(PuneInstitution):

    def create_seasonal_agent_input_subset(self, agent_list, ag_input_df):
        '''
          This function reads in the list of agents and the full input dataframe and subsets the input dataframe
          into a smaller input dataframe for the listed agents.

          INPUTS: agent_list: list of agent codes
              ag_input_df: full input dataframe from reading an excel file

          OUTPUTS: input_df[agent]: the input dataframe for each agent in the agent list

          '''

        input_df = {}  # initialize empty dictionaries

        for agent in agent_list:  
            agent_input_dict = {}
            # Create One Agents Inputs by subsetting excel file -----------
            for sheet_name in ag_input_df.keys():
                if 'Agent' in ag_input_df[sheet_name].columns:
                    df = ag_input_df[sheet_name]
                    agent_input_dict[sheet_name] = df[df['Agent'] == agent].reset_index().drop(columns='index')
                if 'Agent' not in ag_input_df[sheet_name].columns:
                    agent_input_dict[sheet_name] = ag_input_df[sheet_name].reset_index().drop(columns='index')

            input_df[agent] = agent_input_dict

        return input_df

    def ag_seasonal_optimization_loop(self, agent_list, agent_input_df, interventionid, scenarioid, solar):
        '''
          This function loops the ag_pyomo optimization for each agent in an agent_list.

          INPUTS: agent_list: list of agent codes
              ag_input_df: input dataframe for the agents

          OUTPUTS:optimized outputs for each agent

          '''

        count = 0

        agent_mod_obj_dict = {}  # this will store the model objects
        agent_result_obj_dict = {}  # this will store the result objects
        result_terminal_condition_dict_check = {}  # for checking

        for agent in agent_list:
            input_df = agent_input_df[agent]

            # Do Optimization ------------------------------------------
            print('count:' + str(count))

            # tic()

            print("Agent Code: " + str(agent))

            # OPTIMIZATION
            if interventionid == 'solarfarming':
                model, result = ag_optimization_solar(input_df,solar)
                print('solarfarming optimization')
                
            elif interventionid == 'elecprice' or interventionid == 'demand_management' or interventionid == 'price_regulation' or interventionid == 'comprehensive':
                model, result = ag_optimization_elec(input_df)
                print('elec price optimization')
                
            elif scenarioid == 'historical' or scenarioid == 'baseline':
                model, result = ag_optimization_historical(input_df,solar)
                print('historical optimization')

            else:
                model, result = ag_optimization(input_df)


            # Save the optimization outputs for each agent
            agent_mod_obj_dict[agent] = model
            agent_result_obj_dict[agent] = result
            result_terminal_condition_dict_check[agent] = str(result.Solver[0]['Termination condition'])

            # toc()

            count = count + 1

        return agent_mod_obj_dict

    def extract_seasonal_ag_outputs(self, agent_list, agent_mod_obj_dict, excel_out_filepath, toExcel=True):
        '''
        This function reads in model objects for each agent in the agent list
        It extracts a human readable dataframe and produces an excel file if desired.

        INPUTS: agent_list: list of agent codes
            agent_mod_obj_dict: dictionary of model objects produced from running ag_pyomo in a loop
            toExcel : True (default) or False whether to create an excel file of all agents solutions
            excel_out_filepath: path to send to excel if toExcel = True

        OUTPUTS: output: the output dataframe for all agent in the agent list with Land, Water, SW and GW information
             excel file with sheets for Land, Water_use, Surfacewater, Groundwater and Income

        '''

        import pandas as pd

        # The four types of ag outputs are: land, water use, sw, gw, income
        all_agents_land_opt = pd.DataFrame()  # initialize empty dataframe
        all_agents_water_opt = pd.DataFrame()  # initialize empty dataframe
        all_agents_sw_opt = pd.DataFrame()  # initialize empty dataframe
        all_agents_gw_opt = pd.DataFrame()  # initialize empty dataframe
        all_agents_income_opt = pd.DataFrame()  # initialize empty dataframe

        output = {}

        for agent in agent_list:
            model = agent_mod_obj_dict[agent]

            # LAND
            mod_obj = getattr(model, 'land_opt_j_d_v')  # j=crop, t=tech, d=district, v=variable
            land_df = pd.DataFrame([x for x in mod_obj], columns=['Crop', 'Agent'])
            land_df['Land_1000ha'] = [mod_obj[x].value for x in mod_obj]

            # WATER
            mod_obj = getattr(model, 'tot_ir_wat_use_opt_j_s_d_v')
            water_df = pd.DataFrame([x for x in mod_obj], columns=['Month', 'Crop', 'Agent'])
            water_df['wateruse_km3'] = [mod_obj[x].value for x in mod_obj]

            # SURFACE WATER
            mod_obj = getattr(model, 'sw_use_opt_j_s_d_v')
            sw_df = pd.DataFrame([x for x in mod_obj], columns=['Month', 'Crop', 'Agent'])
            sw_df['sw_use_km3'] = [mod_obj[x].value for x in mod_obj]

            # GROUNDWATER
            mod_obj = getattr(model, 'gw_use_opt_j_s_d_v')
            gw_df = pd.DataFrame([x for x in mod_obj], columns=['Month', 'Crop', 'Agent'])
            gw_df['gw_use_km3'] = [mod_obj[x].value for x in mod_obj]

            # INCOME
            mod_obj = getattr(model, 'income_opt_j_d_v')
            income_df = pd.DataFrame([x for x in mod_obj], columns=['Crop', 'Agent'])
            income_df['income_billion_RS'] = [mod_obj[x].value for x in mod_obj]

            # Append to store all agents ------------------------------------------
            all_agents_land_opt = all_agents_land_opt.append(land_df, ignore_index=True)
            all_agents_water_opt = all_agents_water_opt.append(water_df, ignore_index=True)
            all_agents_sw_opt = all_agents_sw_opt.append(sw_df, ignore_index=True)
            all_agents_gw_opt = all_agents_gw_opt.append(gw_df, ignore_index=True)
            all_agents_income_opt = all_agents_income_opt.append(income_df, ignore_index=True)

        output['Land'] = all_agents_land_opt
        output['Water'] = all_agents_water_opt
        output['SW'] = all_agents_sw_opt
        output['GW'] = all_agents_gw_opt
        output['Income'] = all_agents_income_opt

        # print dataframes to excel
        if toExcel:
            with pd.ExcelWriter(excel_out_filepath) as writer:
                all_agents_land_opt.to_excel(writer, sheet_name='Land')
                all_agents_water_opt.to_excel(writer, sheet_name='Water_use')
                all_agents_sw_opt.to_excel(writer, sheet_name='Surfacewater')
                all_agents_gw_opt.to_excel(writer, sheet_name='Groundwater')
                all_agents_income_opt.to_excel(writer, sheet_name='Income')

        return output

