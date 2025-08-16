__author__ = 'ankunwang'

import xarray as xr
from basepath_file import basepath
import pandas as pd
from runs.run_modularized.reservoir_rule import sw_interaction
from helper_objects.utils import dailync_monthlyfirst_ak, dailync_monthlysum_ak, dailync_monthlymean_ak
from global_var import adminsegs_df_urban
sw_urban = {}
sw_ag = {}
res_initial_out = {}
release_out = {}


class ImplementationInputs(object):
    def __init__(self,model):
        self.model = model
        self.gw_urban = None
        self.sw_urban = None

    def groundwater_urban(self, path_planning, time_counter, year, month):
        """
        set the available groundwater for the urban_module module
        """
        # ----------------------------------------- Groundwater--------------------------------------------------------
        basepath_GW = path_planning + "_t" + str(time_counter + 1) + "/" + "gwdepth_adjusted" + "_" + "daily.nc"
        gw = dailync_monthlymean_ak(basepath_GW, year, month)
        gw.name = 'groundwater_depth'

        gw_urban = {}
        gw_df = gw.to_dataframe().reset_index().round({'lon': 4, 'lat': 4})
        adminsegs_df_urban_round = adminsegs_df_urban.round({'lon': 4, 'lat': 4})
        gw_urban[time_counter] = gw_df.merge(adminsegs_df_urban_round, on=['lat', 'lon'])

        setattr(self, "gw_urban", gw_urban)
        print('set the attribute gw urban_module')

    def surfacewater_urban(self, t0_year, pp, path_planning, time_counter, year, month):
        """
        set the available surface water for the urban_module module
        """
        # ----------------------------------------- Groundwater--------------------------------------------------------
        basepath_SW = path_planning + "_t" + str(time_counter + 1) + "/" + "lakeResStorage" + "_" + "daily.nc"
        sw = dailync_monthlyfirst_ak(basepath_SW, year, month)

        basepath_Inflow = path_planning + "_t" + str(time_counter + 1) + "/" + "lakeResInflowM" + "_" + "daily.nc"
        Infl = dailync_monthlysum_ak(basepath_Inflow, year, month)

        basepath_Outflow = path_planning + "_t" + str(time_counter + 1) + "/" + "lakeResOutflowM" + "_" + "daily.nc"
        Outfl = dailync_monthlysum_ak(basepath_Outflow, year, month)

        basepath_Leakage = path_planning + "_t" + str(time_counter + 1) + "/" + "leakage" + "_" + "daily.nc"
        Leak = dailync_monthlysum_ak(basepath_Leakage, year, month)

        basepath_Evap = path_planning + "_t" + str(time_counter + 1) + "/" + "EvapWaterBodyM" + "_" + "daily.nc"
        Evaporation = dailync_monthlysum_ak(basepath_Evap, year, month)

        if time_counter == 0:
            (res_initial_out[time_counter], release_out[time_counter], sw_urban[time_counter], sw_ag[time_counter]) = \
                sw_interaction(t0_year, pp, sw, Infl, Outfl, Leak, Evaporation, {}, month)
        else:
            (res_initial_out[time_counter], release_out[time_counter], sw_urban[time_counter], sw_ag[time_counter]) = \
                sw_interaction(t0_year, pp, sw, Infl, Outfl, Leak, Evaporation, res_initial_out[time_counter - 1], month)

        setattr(self, "sw_urban", sw_urban)
        print('set the attribute sw urban_module')
