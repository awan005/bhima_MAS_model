__author__ = 'ankunwang'

from netsim.model_components.institutions.urban_water_decisions import MunicipalCorporation
from netsim.network_setup import instantiate_HouseholdAgent_nodes, register_nodes_by_type
from python_packages.pynsim import Network
from python_packages.pynsim import Simulator
from netsim.urbandata_input import ExgonousInputs
from helper_objects.helper_functions import get_excel_data
import pandas as pd
from basepath_file import basepath
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import mapping

class UrbanModule(Simulator):
    """
    A Bhima Basin Simulator Class (a subclass of the pynsim Simulator class).

    Args:
        name (string): name of simulation
        scenario (string): scenario id (to match with id for associated exogenous inputs)
        intervention (string): intervention id (to match with id for associated exogenous inputs)
    """

    def __init__(self, name, scenario, intervention):
        super(UrbanModule, self).__init__()
        self.name = name
        self.scenario_id = scenario
        self.intervention_id = intervention

    def set_network(self, name, path):
        network = Network(name='TestNet')

        # Read in network nodes from excel file path
        network_nodes = []
        instantiate_HouseholdAgent_nodes(network_nodes, path)
        network.add_nodes(*network_nodes)
        municorp = MunicipalCorporation("PMC")
        for i in ['HouseholdAgent', 'SlumHouseholdAgent', 'CommercialAgent', 'IndustrialAgent']:
            register_nodes_by_type(network, i, municorp)
        network.add_institutions(municorp)

        self.network = network

    def set_exogenous_inputs(self, update_years, proj_filepath):
        self.network.exogenous_inputs = ExgonousInputs(self)
        self.network.exogenous_inputs.load_urban_module_data(update_years, proj_filepath)

    def run_urban_module(self, interventionID, run_year,update_years, groundwater_input, surfacewater_input, human_da,gw_cap, cap, user_parameters):

        hh = [a for a in self.network.nodes if
              (a.node_type in ["HouseholdAgent", "SlumHouseholdAgent"]) and (a.units > 0)]
        est = [a for a in self.network.nodes if
               (a.node_type in ["CommercialAgent", "IndustrialAgent"]) and (a.units > 0)]


        gw_inputs_ck201118 = groundwater_input  # integration with groundwater
        lat_lon_to_id_1 = gw_inputs_ck201118.reset_index()
        lat_lon_to_id_1["X"] = lat_lon_to_id_1["lon"].round(4)
        lat_lon_to_id_1["Y"] = lat_lon_to_id_1["lat"].round(4)
        merger = human_da.merge(lat_lon_to_id_1, on=['X', 'Y']).set_index("id").rename(
            columns={"index": "RegID", "Agent": "AgentID"})
            
        tanker_shape = gpd.read_file(basepath + r"/modules/hydro/hydro_files/shapefiles/tanker/tanker.shp")

        adminsegs_df_urban = pd.read_csv(basepath + r'/modules/hydro/hydro_files/excel/Res_CCA_Agent_Pixel_updated_urban.csv')
        gw_df = gw_inputs_ck201118.reset_index().round({'lon': 4, 'lat': 4})
        adminsegs_df_urban_round = adminsegs_df_urban.round({'lon': 4, 'lat': 4})
        gw_urban = gw_df.merge(adminsegs_df_urban_round, on=['lat', 'lon'])
        
        gw_urban['geometry']= gw_urban.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
        gw_urban_gpd = gpd.GeoDataFrame(gw_urban, crs=tanker_shape.crs, geometry="geometry")
        urban_tanker = gpd.sjoin(gw_urban_gpd, tanker_shape[['geometry']],op="within")
       
       
        sw_inputs_ck201118_prep = surfacewater_input  # integration with surface water

        sw_agent_ids = sw_inputs_ck201118_prep["AgentID"]
        reg_agent_dict = {k: [] for k in sw_agent_ids}
        sw_inputs_ck201118 = sw_inputs_ck201118_prep.set_index("AgentID")
        sw_inputs_ck201118.head()

        for a in self.network.get_institution('PMC').nodes:
            reg_agent_dict[a.SWAgentID] += [a]

        y = run_year  # set the run year

        print("\n+++ Now running year {} +++".format(y))

        # ++++++
        # Update global parameters:

        # Update agents
        if update_years == 'historical':
            ssp_select = 'SSP2'  # historical use any ssp level is okay
        else:
            ssp_select = update_years
        self.network.get_institution('PMC').update_agents_to_year_exo(y, ssp=ssp_select)
        self.network.get_institution('PMC').total_population = sum(a.hh_size * a.units for a in hh)
        self.network.get_institution('PMC').total_establishments = sum(a.units for a in est)
        _avg_income = sum(a.hh_income * a.units for a in hh) / sum(a.units for a in hh)
        # print("\nTotal population: {}".format(self.network.get_institution('PMC').total_population))
        # print("Avg. income: {}".format(_avg_income))
        # print("Total establishments: {}".format(self.network.get_institution('PMC').total_establishments))

        # Piped supply:
        agent_subset = user_parameters["agent_subset"][0]

        # Well pumping heads:
        for a in self.network.get_institution('PMC').nodes:
            a.head_development_monthend = gw_inputs_ck201118["groundwater_depth"][a.RegID]

        # Electricity capacity:
        total_power_generation_capacity = self.network.exogenous_inputs.electricity_capacity_projection.loc[y][0]  
        # print("Total power generation capacity: {}".format(total_power_generation_capacity))

        # ++++++
        # Distribute resources
        urban_population = self.network.exogenous_inputs.population_projection[['X', 'Y', run_year]].rename(
            columns={'X': 'lon', 'Y': 'lat'}).reset_index()

        # agent_cca_lat_lon_file
        res_cca = pd.read_csv(basepath + r'/modules/hydro/hydro_files/excel/Res_CCA_Agent_Pixel_updated_urban.csv')
        res_cca = res_cca.fillna(0)

        merge_pop_cca = urban_population.merge(res_cca, on=['lon', 'lat'])
        merge_pop_cca_agent = merge_pop_cca.groupby(['Agent']).sum().reset_index()[['Agent', run_year]]
        merge_pop_cca_agent['CCA code'] = merge_pop_cca.groupby('Agent')['CCA code'].apply(
            lambda x: x.value_counts().index[0]).reset_index()['CCA code']
        merge_pop_cca_agent_weight = merge_pop_cca_agent.groupby(['CCA code']).sum().reset_index().merge(
            merge_pop_cca_agent, on=['CCA code'])

        merge_pop_cca_agent_weight['weight'] = merge_pop_cca_agent_weight[str(run_year) + '_y'] / \
                                               merge_pop_cca_agent_weight[str(run_year) + '_x']
        merge_pop_cca_agent_weight = merge_pop_cca_agent_weight[['Agent_y', 'weight']].rename(
            columns={'Agent_y': 'Agent'})

        sw_inputs_ck201118 = sw_inputs_ck201118_prep.rename(columns={'AgentID': 'Agent'}).merge(
            merge_pop_cca_agent_weight, on='Agent')
        sw_inputs_ck201118["sw_m3_normal_weight"] = sw_inputs_ck201118["sw_m3_normal"] * sw_inputs_ck201118['weight']
        sw_inputs_ck201118 = sw_inputs_ck201118.set_index("Agent")
        
        for i in sw_agent_ids:
            _total_piped_supply = sw_inputs_ck201118["sw_m3_normal_weight"][i]
            _agent_subset = reg_agent_dict[i]
            self.network.get_institution('PMC').distribute_piped_supply(run_year, _total_piped_supply,
                                                                         user_parameters["piped_price_factor"][0],
                                                                         user_parameters["electricity_price_factor"][0],
                                                                         user_parameters["piped_demand_factor"][0],
                                                                         _agent_subset)

        if interventionID == 'donothing' or interventionID == 'reducesugar' or interventionID == 'elecprice' or interventionID == 'waterprice' or interventionID == 'tata' or interventionID == 'tanker' or interventionID == 'irrefficiency' or interventionID == 'solarfarming' or interventionID == 'nonrevnuewater' or interventionID == 'enhanced_supply' or interventionID == 'price_regulation' or interventionID == 'supplyprice':
            self.network.get_institution('PMC').pump_groundwater(run_year,
                                                                 user_parameters["electricity_price_factor"][0],
                                                                 user_parameters["well_demand_factor"][0])
            print(interventionID + 'pump groundwater')
            
            _tanker_cap_pune = cap*2
            gw_depth_tier1 = urban_tanker[urban_tanker.index_right.isin([4,7,3,5])]['groundwater_depth'].mean()

            _agent_subset_pune = reg_agent_dict[1] + reg_agent_dict[2]
            self.network.get_institution('PMC').buy_tanker_water_pune(user_parameters["tanker_price_factor"][0],
                                                                      user_parameters[
                                                                          "electricity_price_factor"][0],
                                                                      user_parameters["tanker_demand_factor"][0],
                                                                      _tanker_cap_pune, _agent_subset_pune, gw_depth_tier1)
                                                                          
            for i in [i for i in sw_agent_ids if i not in [1,2]]:
                _agent_subset = reg_agent_dict[i]
                self.network.get_institution('PMC').buy_tanker_water_non_pune(
                    user_parameters["tanker_price_factor"][0],
                    user_parameters["electricity_price_factor"][0],
                    user_parameters["tanker_demand_factor"][0],
                    _agent_subset)
            print(interventionID + 'buy tanker water')

        elif interventionID == 'demand_management' or interventionID == 'capgw' or interventionID == 'agtourbandissection' or interventionID == 'physical_regulation' or interventionID == 'supplydemand' or interventionID == 'tatademand' or interventionID == 'revdemand' or interventionID == 'supplyphys':
            print(interventionID + 'pump groundwater with gw cap and buy tanker water')
            
            for i in sw_agent_ids:
                _agent_subset = reg_agent_dict[i]
                if i in [1, 2]:
                    self.network.get_institution('PMC').pump_groundwater_cap_pune(run_year,
                                                                                   user_parameters[
                                                                                       "electricity_price_factor"][0],
                                                                                   user_parameters[
                                                                                       "well_demand_factor"][0],
                                                                                   gw_cap, _agent_subset)
                else:
                    self.network.get_institution('PMC').pump_groundwater_cap_non_pune(run_year,
                                                                                       user_parameters[
                                                                                           "electricity_price_factor"][
                                                                                           0],
                                                                                       user_parameters[
                                                                                           "well_demand_factor"][0],
                                                                                       _agent_subset)
                                                                                       
            _tanker_cap_pune = cap*2
            gw_depth_tier1 = urban_tanker[urban_tanker.index_right.isin([4,7,3,5])]['groundwater_depth'].mean()

            _agent_subset_pune = reg_agent_dict[1] + reg_agent_dict[2]
            self.network.get_institution('PMC').buy_tanker_water_pune(user_parameters["tanker_price_factor"][0],
                                                                      user_parameters[
                                                                          "electricity_price_factor"][0],
                                                                      user_parameters["tanker_demand_factor"][0],
                                                                      _tanker_cap_pune, _agent_subset_pune, gw_depth_tier1)
                                                                          
            for i in [i for i in sw_agent_ids if i not in [1,2]]:
                _agent_subset = reg_agent_dict[i]
                self.network.get_institution('PMC').buy_tanker_water_non_pune(
                    user_parameters["tanker_price_factor"][0],
                    user_parameters["electricity_price_factor"][0],
                    user_parameters["tanker_demand_factor"][0],
                    _agent_subset)

        elif interventionID == 'agtourban' or interventionID == 'capgwdissection' or interventionID == 'agsupply' or interventionID == 'agirr'  or interventionID == 'agprice' or interventionID == 'agsupplyprice' :
            print(interventionID + 'pump groundwater and buy tanker water ag')
            self.network.get_institution('PMC').pump_groundwater(run_year,
                                                                 user_parameters["electricity_price_factor"][0],
                                                                 user_parameters["well_demand_factor"][0])
                        
            _tanker_cap_pune = cap*2
            gw_depth_tier1 = urban_tanker[urban_tanker.index_right.isin([4,7,3,5])]['groundwater_depth'].mean()

            _agent_subset_pune = reg_agent_dict[1] + reg_agent_dict[2]
            self.network.get_institution('PMC').buy_tanker_water_pune_ag_tier1_tier2(user_parameters["tanker_price_factor"][0],
                                                                              user_parameters[
                                                                                  "electricity_price_factor"][0],
                                                                              user_parameters["tanker_demand_factor"][
                                                                                  0],
                                                                              _tanker_cap_pune, _agent_subset_pune, gw_depth_tier1)
                                                                          
            for i in [i for i in sw_agent_ids if i not in [1,2]]:
                _agent_subset = reg_agent_dict[i]
                self.network.get_institution('PMC').buy_tanker_water_non_pune(
                    user_parameters["tanker_price_factor"][0],
                    user_parameters["electricity_price_factor"][0],
                    user_parameters["tanker_demand_factor"][0],
                    _agent_subset)
                        
        else: #comprehensive
            for i in sw_agent_ids:
                _agent_subset = reg_agent_dict[i]
                if i in [1, 2]:
                    self.network.get_institution('PMC').pump_groundwater_cap_pune(run_year,
                                                                                   user_parameters[
                                                                                       "electricity_price_factor"][0],
                                                                                   user_parameters[
                                                                                       "well_demand_factor"][0],
                                                                                   gw_cap, _agent_subset)

                else:
                    self.network.get_institution('PMC').pump_groundwater_cap_non_pune(run_year,
                                                                                       user_parameters[
                                                                                           "electricity_price_factor"][
                                                                                           0],
                                                                                       user_parameters[
                                                                                           "well_demand_factor"][0],
                                                                                       _agent_subset)

            _tanker_cap_pune = cap*2
            gw_depth_tier1 = urban_tanker[urban_tanker.index_right.isin([4,7,3,5])]['groundwater_depth'].mean()

            _agent_subset_pune = reg_agent_dict[1] + reg_agent_dict[2]
            self.network.get_institution('PMC').buy_tanker_water_pune_ag_tier1_tier2(user_parameters["tanker_price_factor"][0],
                                                                              user_parameters[
                                                                                  "electricity_price_factor"][0],
                                                                              user_parameters["tanker_demand_factor"][
                                                                                  0],
                                                                              _tanker_cap_pune, _agent_subset_pune, gw_depth_tier1)
                                                                          
            for i in [i for i in sw_agent_ids if i not in [1,2]]:
                _agent_subset = reg_agent_dict[i]
                self.network.get_institution('PMC').buy_tanker_water_non_pune(
                    user_parameters["tanker_price_factor"][0],
                    user_parameters["electricity_price_factor"][0],
                    user_parameters["tanker_demand_factor"][0],
                    _agent_subset)

        self.network.get_institution('PMC').supply_lifeline_water()
        self.network.get_institution('PMC').distribute_electricity_supply(total_power_generation_capacity,
                                                                           user_parameters["electricity_price_factor"][
                                                                               0],
                                                                           user_parameters["electricity_demand_factor"][
                                                                               0])
        self.network.get_institution('PMC').purchase_food()

        # save historical properties in the dictionary
        for c in self.network.nodes:
            c.post_process()

        # ++++++
        # Store variables (example):
        urban_output = pd.DataFrame(
            [[a.X6, a.Y6, a.units, (a.piped_consumption * a.units), (a.well_consumption * a.units),
              (a.tanker_consumption * a.units), (a.tanker_tier2_consumption * a.units),(a.electricity_consumption * a.units),
              (a.food_consumption * a.units), (a.electricity_for_water * a.units),
              ((a.piped_consumption + a.tanker_consumption + a.well_consumption + a.tanker_tier2_consumption +
                a.lifeline_consumption) * a.units), ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) * a.units),
              (((a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption) * a.units) / max(0.0001,
             ((a.piped_consumption + a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption +
              a.lifeline_consumption) * a.units))),
              a.electricity_for_water, a.piped_consumption, a.well_consumption,
              (a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption),
              ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) / max(0.0001,
             (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption + a.well_consumption +
             a.lifeline_consumption))),
              (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption +
               a.well_consumption + a.lifeline_consumption), a.piped_expenditure,
              a.well_expenditure, a.tanker_expenditure,a.tanker_tier2_expenditure,
              (a.piped_expenditure + a.well_expenditure + a.tanker_expenditure+a.tanker_tier2_expenditure),
              a.electricity_price, a.electricity_expenditure, a.well_type
              ] for a in self.network.nodes],
            columns=["x", "y", "units", "piped_m3/mo", "well_m3/mo", "tanker_m3/mo","tanker_tier2_m3/mo",
                     "electr_kWh/mo", "food_INR/mo", "electr_for_water_kWh/mo",
                     "total_water_m3/mo", "non_piped_m3/mo", "non_piped_percent",
                     "electr_for_water_per_unit", "piped_per_unit", "well_per_unit",
                     "non_piped_per_unit", "non_piped_pct_per_unit", "total_water_per_unit",
                     "piped_expenditure_per_unit", "well_expenditure_per_unit",
                     "tanker_expenditure_per_unit","tanker_tier2_expenditure_per_unit", "expenditure_per_unit",
                     "electricity price", "electricity expenditure","well type"])

        # Store variables (example):
        urban_output_hh = pd.DataFrame(
            [[a.X6, a.Y6, a.units, a.hh_size, a.hh_income, (a.piped_consumption * a.units), (a.well_consumption * a.units),
              (a.tanker_consumption * a.units), (a.tanker_tier2_consumption * a.units),(a.electricity_consumption * a.units),
              (a.food_consumption * a.units), (a.electricity_for_water * a.units),
              ((a.piped_consumption + a.tanker_consumption + a.well_consumption + a.tanker_tier2_consumption +
                a.lifeline_consumption) * a.units), ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) * a.units),
              (((a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption) * a.units) / max(0.0001,
             ((a.piped_consumption + a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption +
              a.lifeline_consumption) * a.units))),
              a.electricity_for_water, a.piped_consumption, a.well_consumption,
              (a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption),
              ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) / max(0.0001,
             (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption + a.well_consumption +
             a.lifeline_consumption))),
              (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption +
               a.well_consumption + a.lifeline_consumption), a.piped_expenditure,
              a.well_expenditure, a.tanker_expenditure,a.tanker_tier2_expenditure,
              (a.piped_expenditure + a.well_expenditure + a.tanker_expenditure+a.tanker_tier2_expenditure),
              a.electricity_price, a.electricity_expenditure, a.well_type
              ] for a in self.network.nodes if
            (a.node_type in ["HouseholdAgent", "SlumHouseholdAgent"])],
           columns=["x", "y", "units", "size", "income","piped_m3/mo", "well_m3/mo", "tanker_m3/mo","tanker_tier2_m3/mo",
                     "electr_kWh/mo", "food_INR/mo", "electr_for_water_kWh/mo",
                     "total_water_m3/mo", "non_piped_m3/mo", "non_piped_percent",
                     "electr_for_water_per_unit", "piped_per_unit", "well_per_unit",
                     "non_piped_per_unit", "non_piped_pct_per_unit", "total_water_per_unit",
                     "piped_expenditure_per_unit", "well_expenditure_per_unit",
                     "tanker_expenditure_per_unit","tanker_tier2_expenditure_per_unit", "expenditure_per_unit",
                     "electricity price", "electricity expenditure","well type"])

        # Store variables (example):
        urban_output_co = pd.DataFrame(
            [[a.X6, a.Y6, a.units, (a.piped_consumption * a.units), (a.well_consumption * a.units),
              (a.tanker_consumption * a.units), (a.tanker_tier2_consumption * a.units),(a.electricity_consumption * a.units),
              (a.food_consumption * a.units), (a.electricity_for_water * a.units),
              ((a.piped_consumption + a.tanker_consumption + a.well_consumption + a.tanker_tier2_consumption +
                a.lifeline_consumption) * a.units), ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) * a.units),
              (((a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption) * a.units) / max(0.0001,
             ((a.piped_consumption + a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption +
              a.lifeline_consumption) * a.units))),
              a.electricity_for_water, a.piped_consumption, a.well_consumption,
              (a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption),
              ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) / max(0.0001,
             (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption + a.well_consumption +
             a.lifeline_consumption))),
              (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption +
               a.well_consumption + a.lifeline_consumption), a.piped_expenditure,
              a.well_expenditure, a.tanker_expenditure,a.tanker_tier2_expenditure,
              (a.piped_expenditure + a.well_expenditure + a.tanker_expenditure+a.tanker_tier2_expenditure),
              a.electricity_price, a.electricity_expenditure, a.well_type
              ] for a in self.network.nodes if (a.node_type in ["CommercialAgent"])],
            columns=["x", "y", "units", "piped_m3/mo", "well_m3/mo", "tanker_m3/mo","tanker_tier2_m3/mo",
                     "electr_kWh/mo", "food_INR/mo", "electr_for_water_kWh/mo",
                     "total_water_m3/mo", "non_piped_m3/mo", "non_piped_percent",
                     "electr_for_water_per_unit", "piped_per_unit", "well_per_unit",
                     "non_piped_per_unit", "non_piped_pct_per_unit", "total_water_per_unit",
                     "piped_expenditure_per_unit", "well_expenditure_per_unit",
                     "tanker_expenditure_per_unit","tanker_tier2_expenditure_per_unit", "expenditure_per_unit",
                     "electricity price", "electricity expenditure","well type"])

        # Store variables (example):
        urban_output_in = pd.DataFrame(
            [[a.X6, a.Y6, a.units, (a.piped_consumption * a.units), (a.well_consumption * a.units),
              (a.tanker_consumption * a.units), (a.tanker_tier2_consumption * a.units),(a.electricity_consumption * a.units),
              (a.food_consumption * a.units), (a.electricity_for_water * a.units),
              ((a.piped_consumption + a.tanker_consumption + a.well_consumption + a.tanker_tier2_consumption +
                a.lifeline_consumption) * a.units), ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) * a.units),
              (((a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption) * a.units) / max(0.0001,
             ((a.piped_consumption + a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption +
              a.lifeline_consumption) * a.units))),
              a.electricity_for_water, a.piped_consumption, a.well_consumption,
              (a.tanker_consumption + a.tanker_tier2_consumption + a.well_consumption),
              ((a.tanker_consumption + a.tanker_tier2_consumption +a.well_consumption) / max(0.0001,
             (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption + a.well_consumption +
             a.lifeline_consumption))),
              (a.piped_consumption + a.tanker_consumption +a.tanker_tier2_consumption +
               a.well_consumption + a.lifeline_consumption), a.piped_expenditure,
              a.well_expenditure, a.tanker_expenditure,a.tanker_tier2_expenditure,
              (a.piped_expenditure + a.well_expenditure + a.tanker_expenditure+a.tanker_tier2_expenditure),
              a.electricity_price, a.electricity_expenditure, a.well_type
              ] for a in self.network.nodes if (a.node_type in ["IndustrialAgent"])],
            columns=["x", "y", "units", "piped_m3/mo", "well_m3/mo", "tanker_m3/mo","tanker_tier2_m3/mo",
                     "electr_kWh/mo", "food_INR/mo", "electr_for_water_kWh/mo",
                     "total_water_m3/mo", "non_piped_m3/mo", "non_piped_percent",
                     "electr_for_water_per_unit", "piped_per_unit", "well_per_unit",
                     "non_piped_per_unit", "non_piped_pct_per_unit", "total_water_per_unit",
                     "piped_expenditure_per_unit", "well_expenditure_per_unit",
                     "tanker_expenditure_per_unit","tanker_tier2_expenditure_per_unit", "expenditure_per_unit",
                     "electricity price", "electricity expenditure","well type"])

        return urban_output, urban_output_hh, urban_output_co, urban_output_in