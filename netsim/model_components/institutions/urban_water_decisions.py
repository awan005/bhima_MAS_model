# author: CK, further developed by AW
from netsim.model_components.institutions._pune_institution import PuneInstitution
from helper_objects.helper_functions import netcdf_to_dataframe
import numpy as np
import pandas as pd
import geopandas as gpd

# import rasterio

############################################################################################################
#    _   _      _                  __        __    _
#   | | | |_ __| |__   __ _ _ __   \ \      / /_ _| |_ ___ _ __
#   | | | | '__| '_ \ / _` | '_ \   \ \ /\ / / _` | __/ _ \ '__|
#   | |_| | |  | |_) | (_| | | | |   \ V  V / (_| | ||  __/ |
#    \___/|_|  |_.__/ \__,_|_| |_|    \_/\_/ \__,_|\__\___|_|
#
############################################################################################################

# Institutions are collection of functions that can be applied
class MunicipalCorporation(PuneInstitution):
    name = "Authority to allocate Urban Water"

    _properties = dict(
        allocation=1000,
        unsatisfied_hhs = []
    )

    def getnodes(self, component_type):  # component_type vs node_type
        """
            Convenience function to get the nodes in a network of a certain type
        """
        nodes = []
        for n in self.nodes:
            if n.type == component_type:
                nodes.append(n)
        return nodes

    def calc_mean_hh_demands(self):
        """
            Test function 
        """
        _price = 1.87
        _calibration = 1.0
        human_agents = self.nodes
        mean_demand = np.mean(
            [a.get_water_demand(price=_price, demand_factor=_calibration)
             * 12. / (365. * a.HhldSize) for a in human_agents if a.Pop_2015 > 0.])
        return mean_demand

    # Added for Piped Distribution
    def distribute_piped_supply(self, year, total_piped_supply, price_factor, electricity_price_factor, demand_factor,
                                agent_subset):
        """
            Piped water distribution for the agents 
        """
        _duration_factor = 1.0  
        
        _all_hhs = [a for a in agent_subset if (a.units > 0) and (a.infrastructure > 0.)]
        
        self.unsatisfied_hhs = []
        
        for a in _all_hhs:
            _min_consumption_threshold = a.hh_size * 30.4167 * 20. / 1000.  # Min. threshold of 20 lcd
            
            if a.node_type=="SlumHouseholdAgent":
                _price = 1 * 7.378
                _electric_tariff_block_1 = 1 * 4.33
                _kWh_per_m3 = 0.692377
                _piped_cost = _price + _electric_tariff_block_1 * _kWh_per_m3
                a.piped_demand = max(a.get_water_demand(price=_piped_cost, demand_factor=demand_factor), _min_consumption_threshold) # m3/hhld./month
                a.piped_consumption = a.piped_demand
                
            else:
                _price = price_factor * 7.378
                _electric_tariff_block_1 = electricity_price_factor * 4.33
                _kWh_per_m3 = 0.692377
                _piped_cost = _price + _electric_tariff_block_1 * _kWh_per_m3
                a.piped_demand = max(a.get_water_demand(price=_piped_cost, demand_factor=demand_factor), _min_consumption_threshold) # m3/hhld./month
                a.piped_consumption = a.piped_demand
 
        if sum(a.piped_demand*a.units for a in _all_hhs) > total_piped_supply:
            # satisfied_hhs = []
            water_distributed = 0
            self.unsatisfied_hhs_piped = list(_all_hhs)
            distribution_completed = False
            while not distribution_completed:
                distribution_completed = True
                remaining_delivery = total_piped_supply - water_distributed
                total_duration_of_unsatisfied_hhs = 0
                for u in self.unsatisfied_hhs_piped:
                    total_duration_of_unsatisfied_hhs += _duration_factor * u.infrastructure * u.units
                for u in self.unsatisfied_hhs_piped:
                    delivery_share = _duration_factor * u.infrastructure * remaining_delivery / total_duration_of_unsatisfied_hhs
                    if delivery_share <= u.piped_demand:
                        u.piped_consumption = delivery_share
                        u.piped_constraint = True
                    else:
                        distribution_completed = False
                        u.piped_consumption = u.piped_demand
                        # satisfied_hhs.append(u)
                        self.unsatisfied_hhs_piped.remove(u)
                        water_distributed += u.piped_consumption * u.units
                        u.piped_constraint = False
                        
        for a in _all_hhs:
            if a.node_type=="SlumHouseholdAgent":
                _price = 1 * 7.378
                _electric_tariff_block_1 = 1 * 4.33
                _kWh_per_m3 = 0.692377
                _piped_cost = _price + _electric_tariff_block_1 * _kWh_per_m3
                a.piped_expenditure = max(_piped_cost * (a.piped_consumption - _min_consumption_threshold), 0)
                a.electricity_for_water = _kWh_per_m3 * a.piped_consumption
                
            else:
                _price = price_factor * 7.378
                _electric_tariff_block_1 = electricity_price_factor * 4.33
                _kWh_per_m3 = 0.692377
                _piped_cost = _price + _electric_tariff_block_1 * _kWh_per_m3
                a.piped_expenditure = max(_piped_cost * (a.piped_consumption - _min_consumption_threshold), 0)
                a.electricity_for_water = _kWh_per_m3 * a.piped_consumption

        # CK201118: -3
        print("\nTotal piped supply (m3/month): " + str(sum(a.piped_consumption * a.units for a in _all_hhs)))
        print("Total piped expenditure (INR/month): " + str(sum(a.piped_expenditure * a.units for a in _all_hhs)))
        print("Unsatisfied agents left for piped water supply: " + str(len(self.unsatisfied_hhs_piped)))


    def pump_groundwater(self, year, electricity_price_factor, demand_factor):
        """
            Well water pumped by the agents 
        """
        # _all_hhs = [a for a in self.nodes if a.Pop_2015 > 0.]

        # CK201118: +1
        self.unsatisfied_hhs = [a for a in self.nodes if (a.units > 0)] 
        _all_hhs = self.unsatisfied_hhs
        
        for a in _all_hhs:
            if a.node_type=="SlumHouseholdAgent":
                _electric_tariff_block_1 = 1 * 4.33
                
            else:
                _electric_tariff_block_1 = electricity_price_factor * 4.33

        # _kWh_per_m3 = 2.022736
        _kWh_per_m3_per_m_head = 0.35844913157894737
        # _income_factor = self.network.income_projection.loc[year]["SSP2"]
        for a in _all_hhs:

            # CK201118: +/-1
            # _head_limit = 21. if (a.well_type == 1) else (51. if (a.well_type == 2) else 0.)
            _head_limit = 15. if (a.well_type == 1) else (40. if (a.well_type == 2) else 0.)
            # add the height of the roof top 
            rooftop = 10  

            _pump_cost = max(0.00001, (rooftop + a.head_development_monthend) *_kWh_per_m3_per_m_head *
                             _electric_tariff_block_1)

            a.well_consumption = max(0.0, (a.get_water_demand(price=_pump_cost, demand_factor=demand_factor)
                                           - a.piped_consumption))  # m3/hhld./month
            a.well_expenditure = _pump_cost * a.well_consumption
            a.electricity_for_water += (rooftop + a.head_development_monthend)  *_kWh_per_m3_per_m_head * a.well_consumption
            self.unsatisfied_hhs.remove(a)

        print("Total well consumption (m3/month): " + str(sum(a.well_consumption * a.units for a in self.nodes)))
        print("Total well expenditure (INR/month): " + str(sum(a.well_expenditure * a.units for a in self.nodes)))
        print("Unsatisfied agents left: " + str(len(self.unsatisfied_hhs)))


    def buy_tanker_water(self, year, price_factor, electricity_price_factor, demand_factor):
        """
            Tanker water use by the agents, assuming supply is sufficient to satisfy demand
        """
        _all_hhs = self.unsatisfied_hhs
        _electric_tariff_block_1 = 4.33
        _kWh_per_m3 = 0.947731
        _tanker_price = 40.23 * price_factor + _electric_tariff_block_1 * _kWh_per_m3 * electricity_price_factor
    
        for a in _all_hhs:
            a.tanker_consumption = max(0.0, (a.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                             - a.piped_consumption - a.well_consumption))  # m3/hhld./month
            a.tanker_expenditure = _tanker_price * a.tanker_consumption
            a.electricity_for_water += _kWh_per_m3 * a.tanker_consumption
    
        print("Total tanker consumption (m3/month): " + str(sum(a.tanker_consumption * a.units for a in _all_hhs)))
        print("Total tanker expenditure (INR/month): " + str(sum(a.tanker_expenditure * a.units for a in _all_hhs)))


    def pump_groundwater_cap_pune(self, year, electricity_price_factor, demand_factor, gw_cap, agent_subset):
        """
            Well water pumped by the agents, with a pumping cap in the Pune agglomeration
        """
        self.unsatisfied_hhs = [a for a in agent_subset if (a.units > 0)] 
        _all_hhs = self.unsatisfied_hhs
 

        # _kWh_per_m3 = 2.022736
        _kWh_per_m3_per_m_head = 0.35844913157894737

        for a in _all_hhs:
            rooftop = 10

            if a.node_type=="SlumHouseholdAgent":
                _electric_tariff_block_1 = 1 * 4.33
                
            else:
                _electric_tariff_block_1 = electricity_price_factor * 4.33

            # CK201118: +/-1
            # _head_limit = 21. if (a.well_type == 1) else (51. if (a.well_type == 2) else 0.)
            _head_limit = 15. if (a.well_type == 1) else (40. if (a.well_type == 2) else 0.)

            _pump_cost = max(0.00001, (rooftop + a.head_development_monthend)  *_kWh_per_m3_per_m_head *
                             _electric_tariff_block_1)
            
            if a.head_development_monthend >= 25:
                cap_m3 = gw_cap * a.hh_size * 30.4167 / 1000. # convert LPCD per person to m3 per month per household

                _well_demand = max(0.0, (a.get_water_demand(price=_pump_cost, demand_factor=demand_factor)
                               - a.piped_consumption))  # m3/hhld./month
                a.well_consumption = min(cap_m3, _well_demand)
                a.well_expenditure = _pump_cost * a.well_consumption
                a.electricity_for_water += (rooftop + a.head_development_monthend) * _kWh_per_m3_per_m_head * a.well_consumption
                print(a.well_consumption)
            else:
                a.well_consumption = max(0.0, (a.get_water_demand(price=_pump_cost, demand_factor=demand_factor)
                                               - a.piped_consumption))  # m3/hhld./month
                a.well_expenditure = _pump_cost * a.well_consumption
                a.electricity_for_water += (rooftop + a.head_development_monthend) *_kWh_per_m3_per_m_head * a.well_consumption
                self.unsatisfied_hhs.remove(a)

        print('pump_groundwater_cap_pune')
        print("Total well consumption (m3/month): " + str(sum(a.well_consumption * a.units for a in self.nodes)))
        print("Total well expenditure (INR/month): " + str(sum(a.well_expenditure * a.units for a in self.nodes)))
        print("Unsatisfied agents left: " + str(len(self.unsatisfied_hhs)))


    def pump_groundwater_cap_non_pune(self, year, electricity_price_factor, demand_factor, agent_subset):
        """
            Well water pumped by the agents outside the Pune agglomeration
        """

        self.unsatisfied_hhs = [a for a in agent_subset if (a.units > 0)] #  and (a.piped_constraint == True)
        _all_hhs = self.unsatisfied_hhs

        # _kWh_per_m3 = 2.022736
        _kWh_per_m3_per_m_head = 0.35844913157894737
        
        for a in _all_hhs:
            rooftop = 10
            
            if a.node_type=="SlumHouseholdAgent":
                _electric_tariff_block_1 = 1 * 4.33
                
            else:
                _electric_tariff_block_1 = electricity_price_factor * 4.33

            # CK201118: +/-1 
            # _head_limit = 21. if (a.well_type == 1) else (51. if (a.well_type == 2) else 0.)
            _head_limit = 15. if (a.well_type == 1) else (40. if (a.well_type == 2) else 0.)

            _pump_cost = max(0.00001, (rooftop + a.head_development_monthend) *_kWh_per_m3_per_m_head *
                             _electric_tariff_block_1)

            a.well_consumption = max(0.0, (a.get_water_demand(price=_pump_cost, demand_factor=demand_factor)
                                           - a.piped_consumption))  # m3/hhld./month
            a.well_expenditure = _pump_cost * a.well_consumption
            a.electricity_for_water += (rooftop + a.head_development_monthend) *_kWh_per_m3_per_m_head * a.well_consumption
            self.unsatisfied_hhs.remove(a)
        
        print('pump_groundwater_cap_non_pune')
        print("Total well consumption (m3/month): " + str(sum(a.well_consumption * a.units for a in self.nodes)))
        print("Total well expenditure (INR/month): " + str(sum(a.well_expenditure * a.units for a in self.nodes)))
        print("Unsatisfied agents left: " + str(len(self.unsatisfied_hhs)))



    def buy_tanker_water_pune_ag_tier1_tier2(self, price_factor, electricity_price_factor, demand_factor, tanker_cap, agent_subset, gw_depth_tier1):
        """
            Tanker water use by the agents, under the intervention of agriculture-to-urban water market.
            Tanker water demand is first satisfied by the original tanker wter market (Tier 1), the rest is supplied by farmers (Tier 2)
        """
        _all_hhs = [a for a in agent_subset if (a.units > 0)]
        _electric_tariff_block_1 = 4.33
        _kWh_per_m3 = 0.947731
        _kWh_per_m3_per_m_head = 0.35844913157894737
        
        transport_fee_tier1 = 84.6
        labor_fee_tier1 = 40
        _tanker_price = _electric_tariff_block_1 * _kWh_per_m3_per_m_head * gw_depth_tier1 * electricity_price_factor + transport_fee_tier1 * price_factor + labor_fee_tier1
        
        for b in _all_hhs:
            b.tanker_consumption = max(0.0, (b.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                             - b.piped_consumption - b.well_consumption))  # m3/hhld./month
        _i = 0  # avoiding infinite loop
        while (sum(a.tanker_consumption * a.units for a in _all_hhs) > tanker_cap) and (_i <= 1000):
            _tanker_price += 10.0
            for c in _all_hhs:
                c.tanker_consumption = max(0.0, (c.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                                 - c.piped_consumption - c.well_consumption))  # m3/hhld./month
            _i += 1
        print("Final tanker price tier1 (INR/m3): " + str(_tanker_price))
        
        ratio = min(1.0, 3000000*2/(sum(a.tanker_consumption * a.units for a in _all_hhs)))
        
        for d in _all_hhs:
            d.tanker_consumption = max(0.0, (d.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                             - d.piped_consumption - d.well_consumption))  * ratio# m3/hhld./month
            d.tanker_expenditure = _tanker_price * d.tanker_consumption
            d.electricity_for_water += _kWh_per_m3 * d.tanker_consumption

            d.tanker_tier2_consumption = max(0.0, (d.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                             - d.piped_consumption - d.well_consumption))  * (1-ratio) # m3/hhld./month
                                             
            d.tanker_tier2_expenditure = _tanker_price * d.tanker_tier2_consumption
            d.electricity_for_water += _kWh_per_m3 * d.tanker_tier2_consumption
            
        print("Total tanker consumption1 (m3/month): " + str(sum(a.tanker_consumption * a.units for a in _all_hhs)))
        print("Total tanker expenditure1 (INR/month): " + str(sum(a.tanker_expenditure * a.units for a in _all_hhs)))
        print("Total tanker consumption (m3/month): " + str(sum(a.tanker_tier2_consumption * a.units for a in _all_hhs)))
        print("Total tanker expenditure (INR/month): " + str(sum(a.tanker_tier2_expenditure * a.units for a in _all_hhs)))

        
    def buy_tanker_water_pune(self, price_factor, electricity_price_factor, demand_factor, tanker_cap, agent_subset, gw_depth):
        """
            Tanker water use by the agents in the Pune agglomeration. 
            The tanker water price is determined by the willingness-to-pay on the demand curve at specific supply
        """
        _all_hhs = [a for a in agent_subset if (a.units > 0)]
        _electric_tariff_block_1 = 4.33
        _kWh_per_m3 = 0.947731
        _kWh_per_m3_per_m_head = 0.35844913157894737
        
        transport_fee = 84.6
        labor_fee = 40
        _tanker_price = _electric_tariff_block_1 * _kWh_per_m3_per_m_head * gw_depth * electricity_price_factor + transport_fee * price_factor + labor_fee
        for b in _all_hhs:
            b.tanker_consumption = max(0.0, (b.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                             - b.piped_consumption - b.well_consumption))  # m3/hhld./month
        _i = 0  # avoiding infinite loop
        while (sum(a.tanker_consumption * a.units for a in _all_hhs) > tanker_cap) and (_i <= 1000):
            _tanker_price += 10.0
            for c in _all_hhs:
                c.tanker_consumption = max(0.0, (c.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                                 - c.piped_consumption - c.well_consumption))  # m3/hhld./month
            _i += 1
        print("Final tanker price tier1 (INR/m3): " + str(_tanker_price))
        
        for d in _all_hhs:
            d.tanker_consumption = max(0.0, (d.get_water_demand(price=_tanker_price, demand_factor=demand_factor)
                                             - d.piped_consumption - d.well_consumption))  # m3/hhld./month
            d.tanker_expenditure = _tanker_price * d.tanker_consumption
            d.electricity_for_water += _kWh_per_m3 * d.tanker_consumption
            
        print("Total tanker consumption (m3/month): " + str(sum(a.tanker_consumption * a.units for a in _all_hhs)))
        print("Total tanker expenditure (INR/month): " + str(sum(a.tanker_expenditure * a.units for a in _all_hhs)))
        
    
    # add on 12/02/2023   
    def buy_tanker_water_non_pune(self, price_factor, electricity_price_factor, demand_factor, agent_subset):
        """
            Tanker water use by the agents outside the Pune agglomeration.
        """
        _all_hhs = [a for a in agent_subset if (a.units > 0)]
        _electric_tariff_block_1 = 4.33
        _kWh_per_m3 = 0.947731
        for a in _all_hhs:
            _tanker_price = 40.23 * price_factor + _electric_tariff_block_1 * _kWh_per_m3 * electricity_price_factor
            a.tanker_consumption = 0.0
            a.tanker_expenditure = _tanker_price * a.tanker_consumption
            a.electricity_for_water += _kWh_per_m3 * a.tanker_consumption
        print("Total tanker consumption (m3/month): " + str(sum(a.tanker_consumption * a.units for a in _all_hhs)))
        print("Total tanker expenditure (INR/month): " + str(sum(a.tanker_expenditure * a.units for a in _all_hhs)))
        

    def supply_lifeline_water(self):
        """
            lifeline water supply function
        """
        _all_hhs = self.unsatisfied_hhs
        for a in _all_hhs:
            _min_consumption_threshold = a.hh_size * 30.4167 * 30. / 1000.  # Min. threshold of 30 lcd
            if (a.piped_consumption + a.well_consumption + a.tanker_consumption) < _min_consumption_threshold:
                a.lifeline_consumption = (_min_consumption_threshold - a.piped_consumption - a.well_consumption -
                                          a.tanker_consumption)
        print("Total lifeline consumption (m3/month): " + str(sum(a.lifeline_consumption * a.units for a in _all_hhs)))
        _all_agents = [a for a in self.nodes if (a.units > 0) and (a.infrastructure > 0.)]
        print("\nTotal water consumption (m3/month): " + str(sum((a.piped_consumption + a.well_consumption +
                                                                  a.tanker_consumption + a.lifeline_consumption) *
                                                                 a.units for a in _all_agents)))

    def distribute_electricity_supply(self, total_power_generation_capacity, price_factor, demand_factor):
        """
            Electricity water distribution for the agents 
        """
        _all_hhs = [a for a in self.nodes if (a.units > 0) and (a.power_h > 0.)]
        unsatisfied_hhs_en = []
        _tariff_blocks = [i for i in reversed([100, 300, 500, 1000])]
        _tariffs = [(price_factor * i) for i in reversed([4.33, 8.23, 11.18, 12.78, 13.78])]

        for a in _all_hhs:
            _tariff_accepted = False
            i = 0
            while not _tariff_accepted:
                if i < len(_tariff_blocks):
                    a.electricity_price = _tariffs[i]
                    _hyp_demand = (a.electricity_for_water +
                                   a.get_electricity_demand(price=a.electricity_price, demand_factor=demand_factor))  # m3/hhld./month
                    if _hyp_demand > _tariff_blocks[i]:
                        _tariff_accepted = True
                        a.electricity_demand = _hyp_demand if (i == 0) else min(_tariff_blocks[i-1], _hyp_demand)
                    else:
                        i += 1
                else:
                    _tariff_accepted = True
                    _price = _tariffs[-1]
                    _hyp_demand = (a.electricity_for_water +
                                   a.get_electricity_demand(price=a.electricity_price, demand_factor=demand_factor))  # m3/hhld./month
                    a.electricity_demand = min(_tariff_blocks[i - 1], _hyp_demand)
            a.electricity_consumption = a.electricity_demand

        if sum(a.electricity_demand*a.units for a in _all_hhs) > total_power_generation_capacity:
            capacity_committed = 0
            unsatisfied_hhs_en = list(_all_hhs)
            distribution_completed = False
            while not distribution_completed:
                distribution_completed = True
                remaining_free_capacity = total_power_generation_capacity - capacity_committed
                total_claims_of_unsatisfied_hhs = 0
                for u in unsatisfied_hhs_en:
                    total_claims_of_unsatisfied_hhs += u.power_h * u.units
                for u in unsatisfied_hhs_en:
                    capacity_share = remaining_free_capacity * u.power_h / total_claims_of_unsatisfied_hhs
                    if capacity_share <= u.electricity_demand:
                        u.electricity_consumption = capacity_share
                        u.electricity_constraint = True
                    else:
                        distribution_completed = False
                        u.electricity_consumption = u.electricity_demand
                        unsatisfied_hhs_en.remove(u)
                        capacity_committed += u.electricity_consumption * u.units
                        u.electricity_constraint = False

        for a in _all_hhs:
            a.electricity_expenditure = a.electricity_price * a.electricity_consumption
        # print("\nTotal electricity supply (m3/month): " + str(sum(a.electricity_consumption * a.units for a in _all_hhs)))
        # print("Total electricity expenditure (INR/month): " + str(sum(a.electricity_expenditure * a.units for a in _all_hhs)))
        # # print(unsatisfied_hhs_en)
        # print("Total electricity use for water (m3/month): " + str(sum(a.electricity_for_water * a.units for a in _all_hhs)))
        # print("Unsatisfied agents left: " + str(len(unsatisfied_hhs_en)))

    def purchase_food(self):
        _all_hhs = [a for a in self.nodes if (a.units > 0) and (a.node_type in ["HouseholdAgent", "SlumHouseholdAgent"])]
        for a in _all_hhs:
            a.food_demand = a.get_food_demand()  # INR/hhld./month
            a.food_consumption = a.food_demand
            a.food_expenditure = a.food_demand
        print("\nTotal household food consumption (INR/month): " + str(sum(a.food_consumption * a.units for a in _all_hhs)))


    def update_agents_to_year_exo(self, year, ssp):
        """
            update the household population, establishment units, income based on SSP scenarios
        """
        _population_projection_by_cell = self.network.exogenous_inputs.population_projection[[year]]
        _establishment_factor = self.network.exogenous_inputs.establishment_projection.loc[year][ssp]
        _income_factor = self.network.exogenous_inputs.income_projection.loc[year][ssp]
        _all_hhs = [a for a in self.nodes if a.units > 0]
        for a in _all_hhs:
            a.setup()
            if a.node_type in ["HouseholdAgent", "SlumHouseholdAgent"]:
                a.population = _population_projection_by_cell.loc[a.id].values[0]
                a.units = (a.population * (1 - a.slum_pct)/
                           a.hh_size) if (a.node_type == "HouseholdAgent") else (a.slum_pct * a.population / a.hh_size)
                a.hh_income = a.base_income * _income_factor
            if a.node_type in ["CommercialAgent", "IndustrialAgent"]:
                a.units = a.base_units * _establishment_factor

    def update_node_properties(self, keyword, value):
        """
            Updating Properties
        """
        for node in self.nodes:
            node.keyword = value

    def query_nc_output(self, ncfile_path, property):
        df = netcdf_to_dataframe(ncfile_path)
        for node in self.nodes:
            node.property = df

        return df

    def double_the_data(self, _property):
        for node in self.nodes:
            _doubled = 2 * (
                self.head_development_monthend)  
            node.__dict__.update({'doubled': _doubled})
        return _doubled


    def setup(self, timestamp):
        pass
