__author__ = 'ankun wang'

from runs.run_modularized.planning import planningstage
from runs.run_modularized.implementation import implementationstage
from datetime import date
from helper_objects.helper_functions import add_months
import numpy as np
from netsim.model_components.institutions.urbanmodule_decisions import UrbanModule
from scenario_intervention_input import dic
from basepath_file import basepath

class Variables:
    i=1
    
class IntegratedModel_dyn:
    def __init__(self, parameter):
        self.parameter = parameter


class IntegratedModel(IntegratedModel_dyn):

    def __init__(self, parameter):
        super().__init__(parameter)
        
        # set up parameters needed to initiate urban_module module run
        simulationID = list(dic.keys())[parameter]
        print(simulationID)
        excel_file_path = basepath + r'/modules/urban_module/Inputs' + '/' + 'HumanInputs220619-Final5_' + \
                          dic[simulationID]['scenario'].loc['excel_file_path'][0]
        projection_filepath = basepath + r'/modules/urban_module/Inputs' + '/' + \
                          dic[simulationID]['scenario'].loc['projection_file'][0]
        update_years = dic[simulationID]['scenario'].loc['update_years'][0]
        
        Urban_run = UrbanModule(name='t', scenario='0', intervention='0')
        Urban_run.set_network('TestNet', excel_file_path)
        Urban_run.set_exogenous_inputs(update_years, projection_filepath)
        self.urban = Urban_run
        self.planning_stage = planningstage(self)
        self.implementation_stage = implementationstage(self)

    # =========== DYNAMIC STAGE =======================================
    def dynamicstage(self):
        simulationID = list(dic.keys())[self.para]
        
        interval = dic[simulationID]['scenario'].loc['interval'][0]
        initial_year = dic[simulationID]['scenario'].loc['initial year'][0]
        
        year = interval + initial_year
        startDate = date(year, 6, 1)
        currDate = add_months(startDate, self.currentStep)
        self.planning_stage.run(Variables, currDate.year, int(np.floor(self.currentStep/12)), self.para, self.name)
        self.implementation_stage.run(Variables, currDate.year, int(np.floor(self.currentStep/12)), self.para, self.name)

