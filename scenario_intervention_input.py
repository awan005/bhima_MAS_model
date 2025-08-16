__author__ = 'ankun wang'

import pandas as pd
from basepath_file import basepath 


para_path = r"scenario_intervention_parameters_urban.xlsx" 
scenario_intervention_input = pd.read_excel(para_path, sheet_name=None)
scenario_intervention_input['scenario'].set_index('scenario', inplace=True) 
scenario_intervention_input['intervention'].set_index('intervention', inplace=True)

run_input_path = r"model_run_ak_urban.xlsx" 
setup_input = pd.read_excel(run_input_path)
setup_input.set_index('simulation_name', inplace=True)

dic = {}
for n in range(setup_input.shape[0]):
    dic[setup_input.index[n]] = {}

for keys in dic:
    scenario = setup_input.loc[keys]['scenario_id']
    intervention = setup_input.loc[keys]['intervention_id'] 
    dic[keys]['scenario'] = scenario_intervention_input['scenario'].loc[:, [scenario]]
    dic[keys]['intervention'] = scenario_intervention_input['intervention'].loc[:, [intervention]]
        