from netsim.model_components.nodes._pune_node import PuneNode
from math import exp, log


class PuneAgent(PuneNode):
    """ the urban_module node class"""

    def __init__(self, **kwargs): # general way does not require defining the attributes up front
        self.__dict__.update(**kwargs)
        # self.X6 = round(self.lon_cwatm, 6)
        # self.Y6 = round(self.lat_cwatm, 6)
        self.X6 = round(self.X, 6)
        self.Y6 = round(self.Y, 6)

        # Variables
        self.piped_demand = 0
        self.piped_consumption = 0
        self.piped_expenditure = 0
        self.piped_constraint = False
        self.well_consumption = 0
        self.well_expenditure = 0
        self.tanker_consumption = 0
        self.tanker_tier2_consumption = 0
        self.tanker_expenditure = 0
        self.tanker_tier2_expenditure = 0
        self.lifeline_consumption = 0
        self.water_satisfied = False
        self.electricity_for_water = 0
        self.electricity_demand = 0
        self.electricity_consumption = 0
        self.electricity_expenditure = 0
        self.electricity_price = 0
        self.food_demand = 0
        self.food_consumption = 0
        self.food_expenditure = 0

        # Parameters 
        self.power_h = 0


    def get_food_demand(self):
        return 0.0

    def setup(self):
        # Variables
        self.piped_demand = 0
        self.piped_consumption = 0
        self.piped_expenditure = 0
        self.piped_constraint = False
        self.well_consumption = 0
        self.well_expenditure = 0
        self.tanker_consumption = 0
        self.tanker_tier2_consumption = 0
        self.tanker_expenditure = 0
        self.tanker_tier2_expenditure = 0
        self.lifeline_consumption = 0
        self.water_satisfied = False
        self.electricity_for_water = 0
        self.electricity_demand = 0
        self.electricity_consumption = 0
        self.electricity_price = 0
        self.food_demand = 0
        self.food_consumption = 0
        self.food_expenditure = 0


class HouseholdAgent(PuneAgent):
    """ the urban_module node class"""

    def __init__(self, **kwargs): # general way does not require defining the attributes up front
        self.__dict__.update(**kwargs)
        PuneAgent.__init__(self, **kwargs)

        self.population = self.population2015 * (1 - self.slum_pct)

        # CK201118: 1 line changed
        self.units = (self.population / self.hh_size) if (self.hh_size > 0) else 0.0

        self.power_h = self.hh_power_h
        self.base_income = self.hh_income

    def get_water_demand(self, price, demand_factor):
        _household_income_factor = 8.395430259
        _income = self.hh_income * _household_income_factor
        _size = self.hh_size
        _adjustment = demand_factor * _size
        # return _adjustment * exp(0.66778 + -0.04677 * log(price) + 0.20831 * log(_income) + -0.69097 * log(_size))
        return _adjustment * exp(1.235814394 + -0.331 * log(price) + 0.20831 * log(_income) + -0.69097 * log(_size))

    def get_electricity_demand(self, price, demand_factor):
        _household_income_factor = 8.395430259
        _income = self.hh_income * _household_income_factor + 100*price
        _size = self.hh_size
        _demand_calibration = 5.395634377
        _adjustment = demand_factor * _size * _demand_calibration
        return _adjustment * exp(2.62688 + -0.20623 * log(price) + 0.15015 * log(_income) + -0.57916 * log(_size) +
                                 0.17427 * 0.33 + 0.22313 * 0.33)

    def get_food_demand(self):
        _food_expenditure = 9394.860097 / 8.395430259
        return _food_expenditure

    def setup(self):
        PuneAgent.setup(self)
        self.population = self.population2015
        self.units = self.population / self.hh_size


class SlumHouseholdAgent(HouseholdAgent):
    """ urban_module node class"""

    def __init__(self, **kwargs): # general way does not require defining the attributes up front
        self.__dict__.update(**kwargs)
        HouseholdAgent.__init__(self, **kwargs)

        self.population = self.population2015 * self.slum_pct

        # CK201118: 1 line changed
        self.units = (self.population / self.hh_size) if (self.hh_size > 0) else 0.0

        self.power_h = self.hh_power_h
        self.base_income = self.hh_income

    def get_water_demand(self, price, demand_factor):
        _slum_income_factor = 0.288614 * 8.395430259
        _slum_size_factor = 1.00821
        _slum_demand_factor = 0.476565414
        _income = self.hh_income * _slum_income_factor
        _size = self.hh_size * _slum_size_factor
        _adjustment = demand_factor * _size * _slum_demand_factor
        # return _adjustment * exp(0.66778 + -0.04677 * log(price) + 0.20831 * log(_income) + -0.69097 * log(_size))
        return _adjustment * exp(1.235814394 + -0.331 * log(price) + 0.20831 * log(_income) + -0.69097 * log(_size))

    def get_electricity_demand(self, price, demand_factor):
        _slum_income_factor = 0.288614 * 8.395430259
        _slum_size_factor = 1.00821
        _slum_demand_factor = 1.181623667
        _income = self.hh_income * _slum_income_factor + 100 * max(0, (price - 4.33))
        _size = self.hh_size * _slum_size_factor
        _demand_calibration = 4.684556626
        _adjustment = demand_factor * _size * _slum_demand_factor * _demand_calibration
        return _adjustment * exp(2.62688 + -0.20623 * log(price) + 0.15015 * log(_income) + -0.57916 * log(_size) +
                                 0.17427 * 0.33 + 0.22313 * 0.33)

    def get_food_demand(self):
        _food_expenditure = 8221.569549 / 8.395430259
        return _food_expenditure

    def setup(self):
        HouseholdAgent.setup(self)
        # self.units = self.Pop_2015 / self.hh_size
        self.population = self.population2015 * self.slum_pct
        self.units = self.population / self.hh_size


class CommercialAgent(PuneAgent):
    """ the urban_module node class"""

    def __init__(self, **kwargs): # general way does not require defining the attributes up front
        self.__dict__.update(**kwargs)
        PuneAgent.__init__(self, **kwargs)

        self.units = self.co_units
        self.base_units = self.units
        self.power_h = self.co_power_h

    def get_water_demand(self, price, demand_factor):
        _adjustment = demand_factor * 3.302836 * (1. / (1000. * 7.)) * (365. / 12.)
        return _adjustment * exp(7.945170823 + -0.2083004 * log(price))

    def get_electricity_demand(self, price, demand_factor):
        _demand_calibration = 0.028026938
        _adjustment = demand_factor * _demand_calibration
        return _adjustment * exp(9.960407139 + -0.26 * log(price))

    def get_food_demand(self):
        return 0.0

    def setup(self):
        PuneAgent.setup(self)
        self.units = self.co_units


class IndustrialAgent(PuneAgent):
    """ the urban_module node class"""

    def __init__(self, **kwargs): # general way does not require defining the attributes up front
        self.__dict__.update(**kwargs)
        PuneAgent.__init__(self, **kwargs)

        self.units = self.it_units
        self.base_units = self.units
        self.power_h = self.co_power_h

    def get_water_demand(self, price, demand_factor):
        _adjustment = demand_factor * 3.302836 * (1. / (1000. * 7.)) * (365. / 12.)
        return _adjustment * exp(7.945170823 + -0.2083004 * log(price))

    def get_electricity_demand(self, price, demand_factor):
        _demand_calibration = 1.4016E-07
        _adjustment = demand_factor * _demand_calibration
        return _adjustment * exp(24.46225564 + -0.32 * log(price))

    def get_food_demand(self):
        return 0.0

    def setup(self):
        PuneAgent.setup(self)
        self.units = self.it_units