__author__ = 'ankunwang'

import numpy as np
import pandas as pd


class AgOutput_season_to_mo():

    def totalwater_season_to_mo(self, water_deficit_ratio, season_crop_agent_totalwater_df):

        for i in np.arange(1, 25):
            if i in np.arange(6, 11):
                water_deficit_ratio['m' + str(i) + 'percent'] = water_deficit_ratio['m' + str(i)] / \
                                                                     water_deficit_ratio['khy1_ori'] * 1000000
            if i in np.arange(11, 18):
                water_deficit_ratio['m' + str(i) + 'percent'] = water_deficit_ratio['m' + str(i)] / \
                                                                     water_deficit_ratio['rby1_ori'] * 1000000
            if i in np.arange(18, 23):
                water_deficit_ratio['m' + str(i) + 'percent'] = water_deficit_ratio['m' + str(i)] / \
                                                                     water_deficit_ratio['khy2_ori'] * 1000000
            if i in [23, 24, 1, 2, 3, 4, 5]:
                water_deficit_ratio['m' + str(i) + 'percent'] = water_deficit_ratio['m' + str(i)] / \
                                                                     water_deficit_ratio['rby2_ori'] * 1000000

        water_percent_output = season_crop_agent_totalwater_df.merge(water_deficit_ratio, on=['Crop', 'Agent'])

        for i in np.arange(1, 25):
            water_percent_output['m' + str(i) + '_water_use'] = 0
            if i in np.arange(6, 11):
                water_percent_output.loc[water_percent_output['Month'] == 'khy1', 'm' + str(i) + '_water_use'] += \
                    water_percent_output.loc[water_percent_output['Month'] == 'khy1', 'wateruse_km3'] * \
                    water_percent_output.loc[water_percent_output['Month'] == 'khy1', 'm' + str(i) + 'percent']
            if i in np.arange(11, 18):
                water_percent_output.loc[water_percent_output['Month'] == 'rby1', 'm' + str(i) + '_water_use'] += \
                    water_percent_output.loc[water_percent_output['Month'] == 'rby1', 'wateruse_km3'] * \
                    water_percent_output.loc[water_percent_output['Month'] == 'rby1', 'm' + str(i) + 'percent']
            if i in np.arange(18, 23):
                water_percent_output.loc[water_percent_output['Month'] == 'khy2', 'm' + str(i) + '_water_use'] += \
                    water_percent_output.loc[water_percent_output['Month'] == 'khy2', 'wateruse_km3'] * \
                    water_percent_output.loc[water_percent_output['Month'] == 'khy2', 'm' + str(i) + 'percent']
            if i in [23, 24, 1, 2, 3, 4, 5]:
                water_percent_output.loc[water_percent_output['Month'] == 'rby2', 'm' + str(i) + '_water_use'] += \
                    water_percent_output.loc[water_percent_output['Month'] == 'rby2', 'wateruse_km3'] * \
                    water_percent_output.loc[water_percent_output['Month'] == 'rby2', 'm' + str(i) + 'percent']

        water_land_month_output = water_percent_output.groupby(['Crop', 'Agent']).sum().reset_index()

        mo_crop_agent_totalwater_df = water_land_month_output[['Crop', 'Agent'] +
                                                              ['m' + str(i) + '_water_use' for i in
                                                               range(1, 25)]].fillna(0)
        mo_agent_totalwater_df = mo_crop_agent_totalwater_df.groupby(['Agent']).sum().reset_index().fillna(0)

        return mo_agent_totalwater_df

    def surfacewater_season_to_mo(self, sw_avai_mo_input, mo_agent_totalwater_df, season_crop_agent_surfacewater_df):

        sw_Month_agent = season_crop_agent_surfacewater_df.groupby(['Month', 'Agent']).sum().reset_index()

        for i in np.arange(sw_avai_mo_input.shape[0]):
            sw_avai_mo_input.loc[i, 'sw_use_km3_khy1'] = \
            sw_Month_agent[sw_Month_agent['Month'] == 'khy1'].reset_index().loc[
                i, 'sw_use_km3']
            sw_avai_mo_input.loc[i, 'sw_use_km3_rby1'] = \
            sw_Month_agent[sw_Month_agent['Month'] == 'rby1'].reset_index().loc[
                i, 'sw_use_km3']
            sw_avai_mo_input.loc[i, 'sw_use_km3_khy2'] = \
            sw_Month_agent[sw_Month_agent['Month'] == 'khy2'].reset_index().loc[
                i, 'sw_use_km3']
            sw_avai_mo_input.loc[i, 'sw_use_km3_rby2'] = \
            sw_Month_agent[sw_Month_agent['Month'] == 'rby2'].reset_index().loc[
                i, 'sw_use_km3']

        # if total water use > available surface water --> first use all the available surface water, the rest are groundwater
        # if total water use < available surface water --> all the water use is from surface water
        for j in np.arange(sw_avai_mo_input.shape[0]):
            for i in np.arange(1, 25):
                water_use = mo_agent_totalwater_df[mo_agent_totalwater_df['Agent'] == sw_avai_mo_input.loc[
                    j]['Agent']]['m' + str(i) + '_water_use'].iloc[-1]
                if sw_avai_mo_input.loc[j, 'm' + str(i)] > water_use:
                    sw_avai_mo_input.loc[j, 'm' + str(i) + '_sw_use'] = water_use
                else:
                    sw_avai_mo_input.loc[j, 'm' + str(i) + '_sw_use'] = sw_avai_mo_input.loc[j, 'm' + str(i)]

        mo_agent_surfacewater_df = sw_avai_mo_input[
            ['Agent'] + ['m' + str(i) + '_sw_use' for i in range(1, 25)]].fillna(0)

        return mo_agent_surfacewater_df

    def groundwater_season_to_mo(self, mo_agent_totalwater_df, mo_agent_surfacewater_df):

        mo_agent_groundwater_df = pd.DataFrame(mo_agent_surfacewater_df['Agent'])
        for j in np.arange(mo_agent_groundwater_df.shape[0]):
            for i in np.arange(1, 25):
                mo_agent_groundwater_df.loc[j, 'm' + str(i) + '_gw_use'] = mo_agent_totalwater_df.loc[
                                                                               j, 'm' + str(i) + '_water_use'] - \
                                                                           mo_agent_surfacewater_df.loc[
                                                                               j, 'm' + str(i) + '_sw_use']
                if mo_agent_groundwater_df.loc[j, 'm' + str(i) + '_gw_use'] < 0:
                    mo_agent_groundwater_df.loc[j, 'm' + str(i) + '_gw_use'] = 0

        return mo_agent_groundwater_df
