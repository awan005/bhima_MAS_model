__author__ = 'ankun wang'

from helper_objects.helper_functions import get_excel_data
from basepath_file import basepath

class ExgonousInputs(object):

    def __init__(self, simulation):
        self.scenario = simulation.scenario_id  # name of the scenario
        self.intervention = simulation.intervention_id  # name of the intervention

    def load_urban_module_data(self, update_years, proj_filepath):

        # define the projection file used under each ssp scenario
        print("updated SSP selected is: " + update_years)
        
        projection_file = get_excel_data(proj_filepath)

        population_params = projection_file.parse("1-HD-Population").set_index("id")
        establishment_params = projection_file.parse("2-CO&IT-EstablishmentFactor").set_index("Year")
        income_params = projection_file.parse("3-HD-IncomeFactor").set_index("Year")
        electricity_capacity_params = projection_file.parse("4-EN-Capacity").set_index("Year")

        setattr(self, "population_projection", population_params)
        print(" Projection 1/4 saved")
        setattr(self, "establishment_projection", establishment_params)
        print(" Projection 2/4 saved")
        setattr(self, "income_projection", income_params)
        print(" Projection 3/4 saved")
        setattr(self, "electricity_capacity_projection", electricity_capacity_params)
        print(" Projection 4/4 saved")


