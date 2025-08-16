__author__ = 'ankunwang'

import xarray as xr
from basepath_file import basepath
import pandas as pd
import numpy as np
from global_var import adminsegs, cellarea, Agent_months, adminsegs_df_urban,ag_input_df, month_days
from runs.run_modularized.historical_condition import first_historical_climate, sum_historical_climate, \
    average_monthly_historical_climate, average_historical_climate, sum_monthly_historical_climate
from runs.run_modularized.reservoir_rule import sw_interaction


class HistoricalAgUrbanInputs(object):
    def __init__(self, model):
        self.model = model
        self.Avai_land = None
        self.sw_2yr = None
        self.river_2yr = None
        self.sw_river_season = None
        self.gw_2yr_season = None
        self.ET_2yr = None
        self.Effrain_2yr = None
        self.water_deficit_ratio = None
        self.sw_urban = None
        self.gw_urban = None

    def avai_land(self):
        """
        set the available land
        """
        irr_file = basepath + r'/modules/hydro/hydro_inputs_external/landuses/frac_irrigated.nc'
        irr_nc = xr.open_dataarray(irr_file)

        paddy_file = basepath + r'/modules/hydro/hydro_inputs_external/landuses/frac_paddy.nc'
        paddy_nc = xr.open_dataarray(paddy_file)

        land_nc = irr_nc + paddy_nc

        Irr = land_nc.isel(time=0)
        Irr.name = 'Pix_AvailableFrac_Irr'
        merger = xr.merge([adminsegs, Irr, cellarea], join="override")
        merger_df = merger.drop('time').to_dataframe().reset_index()
        merger_df[
            'AvailableArea_1000ha_pixel'] = merger_df.Pix_AvailableFrac_Irr * merger_df.cellArea_totalend * 0.0001/1000
        Ava_land = merger_df.groupby('Agent').sum().reset_index()[['Agent', 'AvailableArea_1000ha_pixel']]
        Ava_land.rename(columns={'AvailableArea_1000ha_pixel': 'Land_avai'}, inplace=True)

        setattr(self, "Avai_land", Ava_land)
        print('set the attribute avai land')

    def avai_res(self, pp, initial_year, t0_year, t0_month, t0_day, name, period,interval):
        """
        set the available reservoir water
        """
        variables_daily = ['lakeResStorage', 'readAvlChannelStorageM', 'gwdepth_adjusted', 'lakeResInflowM',
                           'lakeResOutflowM', 'leakage', 'EvapWaterBodyM']

        # get 15 years average 
        t0_year_prior = t0_year - period

        Reservoirwater_month_yr12 = first_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                             variables_daily[0], name, period,interval)
        Inflow_month_yr12 = sum_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                   variables_daily[3], name, period,interval)
        Outflow_month_yr12 = sum_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                    variables_daily[4], name, period,interval)
        Leakage_month_yr12 = sum_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                    variables_daily[5], name, period,interval)
        Evap_month_yr12 = sum_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                 variables_daily[6], name, period,interval)

        sw_urban = {}
        sw_ag = {}
        res_initial_out = {}
        release_out = {}
        (res_initial_out[5], release_out[5], sw_urban[5], sw_ag[5]) = sw_interaction(t0_year, pp,
                                                                                     Reservoirwater_month_yr12[5],
                                                                                     Inflow_month_yr12[5],
                                                                                     Outflow_month_yr12[5],
                                                                                     Leakage_month_yr12[5],
                                                                                     Evap_month_yr12[5],
                                                                                     {}, Inflow_month_yr12[
                                                                                         5].year_month_level_1.item())
        for t in range(6, 24):
            (res_initial_out[t], release_out[t], sw_urban[t], sw_ag[t]) = sw_interaction(t0_year, pp,
                                                                                         Reservoirwater_month_yr12[t],
                                                                                         Inflow_month_yr12[t],
                                                                                         Outflow_month_yr12[t],
                                                                                         Leakage_month_yr12[t],
                                                                                         Evap_month_yr12[t],
                                                                                         res_initial_out[t - 1],
                                                                                         Inflow_month_yr12[
                                                                                         t].year_month_level_1.item())

        (res_initial_out[0], release_out[0], sw_urban[0], sw_ag[0]) = sw_interaction(t0_year, pp,
                                                                                     Reservoirwater_month_yr12[0],
                                                                                     Inflow_month_yr12[0],
                                                                                     Outflow_month_yr12[0],
                                                                                     Leakage_month_yr12[0],
                                                                                     Evap_month_yr12[0],
                                                                                     res_initial_out[23],
                                                                                     Inflow_month_yr12[
                                                                                     0].year_month_level_1.item())
        for t in range(1, 5):
            (res_initial_out[t], release_out[t], sw_urban[t], sw_ag[t]) = sw_interaction(t0_year, pp,
                                                                                         Reservoirwater_month_yr12[t],
                                                                                         Inflow_month_yr12[t],
                                                                                         Outflow_month_yr12[t],
                                                                                         Leakage_month_yr12[t],
                                                                                         Evap_month_yr12[t],
                                                                                         res_initial_out[t - 1],
                                                                                         Inflow_month_yr12[
                                                                                         t].year_month_level_1.item())
        list_sw = []
        for i in range(24):
            list_sw.append(sw_ag[i])

        sw_2yr = pd.concat([df.set_index('AgentID') for df in list_sw], ignore_index=False, axis=1).reset_index()
        sw_2yr.columns = ['Agent', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13',
                          'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24']

        sw_2yr = sw_2yr[Agent_months]

        setattr(self, "sw_2yr", sw_2yr)
        setattr(self, "sw_urban", sw_urban)
        print('set the attribute sw 2yr and sw urban_module')

    def avai_river(self, initial_year, t0_year, t0_month, t0_day, name, period,interval):
        """
        set the available river water
        """
        variables_daily = ['lakeResStorage', 'readAvlChannelStorageM', 'gwdepth_adjusted', 'lakeResInflowM',
                           'lakeResOutflowM', 'leakage', 'EvapWaterBodyM']

        # get 15 years average 

        t0_year_prior = t0_year - period
        Riverwater_month_yr12 = first_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                         variables_daily[1], name, period,interval)
        river_file = {}
        for i in range(24):
            river = Riverwater_month_yr12[i, :, :]
            river.name = 'river'
            merger_river = xr.merge([adminsegs, river, cellarea], join="override")
            merger_river_df = merger_river.to_dataframe().reset_index()
            merger_river_df['river_volume'] = merger_river_df.river * merger_river_df.cellArea_totalend * 1E-9 * 0.2
            river_file[i] = merger_river_df.groupby(['Agent']).sum().reset_index()

        list_river = []
        for i in range(24):
            list_river.append(river_file[i][['Agent', 'river_volume']])

        river_2yr = pd.concat([df.set_index('Agent') for df in list_river], ignore_index=False, axis=1).reset_index()

        river_2yr.columns = ['Agent', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13',
                             'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24']
        river_2yr = river_2yr[Agent_months]

        setattr(self, "river_2yr", river_2yr)
        print('set the attribute river 2yr')

    def avai_sw(self): 

        sw_river_combine = self.sw_2yr.set_index('Agent').add(self.river_2yr.set_index('Agent'), fill_value=0).reset_index()
        # for now ignore river component
        # sw_river_combine = sw_2yr

        sw_river_combine['khy1'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in np.arange(6, 11))
        sw_river_combine['rby1'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in np.arange(11, 18))
        sw_river_combine['khy2'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in np.arange(18, 23))
        sw_river_combine['rby2'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in [23, 24, 1, 2, 3, 4, 5])
        sw_river_season = sw_river_combine[['Agent', 'khy1', 'rby1', 'khy2', 'rby2']]

        setattr(self, "sw_river_season", sw_river_season)
        print('set the attribute sw river season')

    def avai_gw(self, initial_year, t0_year, t0_month, t0_day,name, period, interval):
        """
        set the available river water
        """
        variables_daily = ['lakeResStorage', 'readAvlChannelStorageM', 'gwdepth_adjusted', 'lakeResInflowM',
                           'lakeResOutflowM', 'leakage', 'EvapWaterBodyM']

        # get 15 years average 

        t0_year_prior = t0_year - period
        gw_month_yr12 = average_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year,
                                                   variables_daily[2], name, period, interval)
        gw_file = {}
        merger_df_gw = {}
        gw_urban = {}
        for i in range(24):
            gw = gw_month_yr12[i, :, :]
            gw.name = 'groundwater_depth'
            merger = xr.merge([adminsegs, gw], join="override")
            merger_df_gw[i] = merger.to_dataframe().reset_index()
            gw_file[i] = merger_df_gw[i].groupby(['Agent']).mean().reset_index()

            gw_df = gw.to_dataframe().reset_index().round({'lon': 4, 'lat': 4})
            adminsegs_df_urban_round = adminsegs_df_urban.round({'lon': 4, 'lat': 4})
            gw_urban[i] = gw_df.merge(adminsegs_df_urban_round, on=['lat', 'lon'])

        list_gw = []
        for i in range(24):
            list_gw.append(gw_file[i][['Agent', 'groundwater_depth']])

        gw_2yr = pd.concat([df.set_index('Agent') for df in list_gw], ignore_index=False, axis=1).reset_index()
        gw_2yr.columns = ['Agent', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13',
                          'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24']
        gw_2yr = gw_2yr[Agent_months]

        gw_2yr['khy1'] = sum(gw_2yr.loc[:, 'm' + str(i)] for i in np.arange(6, 11)) / 5
        gw_2yr['rby1'] = sum(gw_2yr.loc[:, 'm' + str(i)] for i in np.arange(11, 18)) / 7
        gw_2yr['khy2'] = sum(gw_2yr.loc[:, 'm' + str(i)] for i in np.arange(18, 23)) / 5
        gw_2yr['rby2'] = sum(gw_2yr.loc[:, 'm' + str(i)] for i in [23, 24, 1, 2, 3, 4, 5]) / 7
        gw_2yr_season = gw_2yr[['Agent', 'khy1', 'rby1', 'khy2', 'rby2']]

        setattr(self, "gw_2yr_season", gw_2yr_season)
        setattr(self, "gw_urban", gw_urban)
        print('set the attribute gw 2yr season and gw urban_module')

    def hist_ET(self, t0_year, t0_year_prior, t0_month, t0_day, initial_year, name, period, interval):

        ET_month_yr12 = average_monthly_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, name, period, interval)
        ET_file = {}
        for i in range(24):
            ET_planning = ET_month_yr12[i, :, :]
            ET_planning.name = 'ET'
            merger = xr.merge([adminsegs, ET_planning], join="override")
            merger_df = merger.to_dataframe().reset_index()
            ET_file[i] = merger_df.groupby(['Agent']).mean().reset_index()

        list_ET = []
        for i in range(24):
            list_ET.append(ET_file[i][['Agent', 'ET']])

        ET_2yr = pd.concat([df.set_index('Agent') for df in list_ET], ignore_index=False, axis=1).reset_index()
        ET_2yr.columns = ['Agent', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13',
                          'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24']
        ET_2yr = ET_2yr[Agent_months]

        setattr(self, "ET_2yr", ET_2yr)

    def hist_Effrain(self, t0_year, t0_year_prior, t0_month, t0_day, initial_year, name, period,interval):

        Effrain_month_yr12 = sum_monthly_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, name, period,interval)
        Effrain_file = {}
        for i in range(24):
            Effrain_planning = Effrain_month_yr12[i, :, :]
            Effrain_planning.name = 'Effrain'
            merger = xr.merge([adminsegs, Effrain_planning], join="override")
            merger_df = merger.to_dataframe().reset_index()
            Effrain_file[i] = merger_df.groupby(['Agent']).mean().reset_index()

        list_Effrain = []
        for i in range(24):
            list_Effrain.append(Effrain_file[i][['Agent', 'Effrain']])

        Effrain_2yr = pd.concat([df.set_index('Agent') for df in list_Effrain], ignore_index=False,
                                axis=1).reset_index()
        Effrain_2yr.columns = ['Agent', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12',
                               'm13', 'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20', 'm21', 'm22', 'm23', 'm24']
        Effrain_2yr = Effrain_2yr[Agent_months]

        setattr(self, "Effrain_2yr", Effrain_2yr)
        print('set the attribute effrain 2 yr')

    def updated_crop_water_use(self, paddy_eff=0.6, nonpaddy_eff=0.6):

        rainfall = self.Effrain_2yr
        ref_evap = self.ET_2yr
        crop_coeff = ag_input_df['crop_coeff']

        full_irr = {}
        for crop in range(len(crop_coeff['Crop'])):
            full_irr[crop_coeff['Crop'][crop]] = np.maximum(0, (((ref_evap.mul(crop_coeff.iloc[crop][1:], axis=1)
                                                                  .mul(month_days.iloc[0],
                                                                       axis=1)) - rainfall)) / nonpaddy_eff / 100)
        full_irr['Rice1'] = np.maximum(0, (((ref_evap.mul(crop_coeff.iloc[0][1:], axis=1)
                                             .mul(month_days.iloc[0], axis=1)) - rainfall)) / paddy_eff / 100)
        full_irr['Rice2'] = np.maximum(0, (((ref_evap.mul(crop_coeff.iloc[1][1:], axis=1)
                                             .mul(month_days.iloc[0], axis=1)) - rainfall)) / paddy_eff / 100)
        # unit conversion 1km3/1000hac = 100meter

        final_combine = pd.concat([df for df in full_irr.values()], ignore_index=True, axis=0).reset_index()
        final_combine = final_combine.fillna(0)

        gir = final_combine.iloc[:, 2:]
        gir.loc["two"] = [int(s[1:]) for s in gir.columns]
        rslt_df = gir.sort_values(by='two', axis=1)

        rslt_df['khy1_ori'] = 0
        rslt_df['rby1_ori'] = 0
        rslt_df['khy2_ori'] = 0
        rslt_df['rby2_ori'] = 0
        for i in range(6, 11):
            rslt_df['khy1_ori'] += rslt_df.loc[:, "m" + str(i)] * 1000000  # 1km3/1000ha = 1000000m3/ha
        for i in range(11, 18):
            rslt_df['rby1_ori'] += rslt_df.loc[:, "m" + str(i)] * 1000000
        for i in range(18, 23):
            rslt_df['khy2_ori'] += rslt_df.loc[:, "m" + str(i)] * 1000000
        for i in [23, 24, 1, 2, 3, 4, 5]:
            rslt_df['rby2_ori'] += rslt_df.loc[:, "m" + str(i)] * 1000000

        grs_ir_water_req = rslt_df.iloc[:-1, -4:].fillna(0)

        water_percent = ag_input_df['crop_wat_app'][['Crop', 'Agent']].join(rslt_df.iloc[:-1, :].fillna(0))

        # adjust the full irrigation water requirement by deficit coefficient
        deficit_efficiency = pd.read_csv(basepath + r'/modules/ag_seasonal/ag_inputs/deficit_efficiency.csv')
        water_deficit_ratio = water_percent.merge(deficit_efficiency, how='left', on=['Crop', 'Agent'])
        water_deficit_ratio = water_deficit_ratio.fillna(0)

        water_deficit_ratio['khy1'] = water_deficit_ratio['khy1_ori'] * (1 - water_deficit_ratio['khy1_ratio'])
        water_deficit_ratio['khy2'] = water_deficit_ratio['khy2_ori'] * (1 - water_deficit_ratio['khy2_ratio'])
        water_deficit_ratio['rby1'] = water_deficit_ratio['rby1_ori'] * (1 - water_deficit_ratio['rby1_ratio'])
        water_deficit_ratio['rby2'] = water_deficit_ratio['rby2_ori'] * (1 - water_deficit_ratio['rby2_ratio'])


        setattr(self, "water_deficit_ratio", water_deficit_ratio)
        print('set the attribute water deficit ratio')



