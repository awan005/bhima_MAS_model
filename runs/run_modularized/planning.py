__author__ = 'ankunwang'

import os
import shutil

from helper_objects.helper_functions import toc, tic, create_array, create_netcdf_from_array, add_months, get_excel_data
# Hydro module
from netsim.model_components.institutions.cwatm_decisions import HydroCWATM
# Ag module
from netsim.model_components.institutions.farm_seasonal_decisions import FarmSeasonalDecisionMaker

import numpy as np
import xarray as xr
from global_var import ag_input_df, adminsegs, cellarea, lat_lon, nc_template_file, adminsegs_df, \
    croplist_hydro, Kharif_crop_list, Rabi_crop_list, Jan_crop_list, croplist_agro, Months
from basepath_file import basepath
from datetime import date, timedelta
import pandas as pd
from runs.run_modularized.prepare_ag_inputs import HistoricalAgUrbanInputs
from runs.run_modularized.ag_outputs_season_to_mo import AgOutput_season_to_mo
from runs.run_modularized.urban_to_cwatm import urban_df_to_cwatm

from scenario_intervention_input import dic

class planningstage(object):
    def __init__(self, model):
        self.model = model 
        self.urban = self.model.urban

    def run(self, share, year_plan, count_plan, para, name):
        """
        Planning stage of the integrated model 
        """
        simulationID = list(dic.keys())[para]
        print(simulationID)

        # File inputs meteo under different SSP scenarios
        PathMeteo = basepath + r'/modules/hydro/hydro_inputs/meteo' + '/' + \
            dic[simulationID]['scenario'].loc['PathMeteo'][0]
        WindMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['WindMaps'][0]
        PreciMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['PrecipitationMaps'][0]
        TavgMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['TavgMaps'][0]
        TminMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['TminMaps'][0]
        TmaxMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['TmaxMaps'][0]
        PSurfMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['PSurfMaps'][0]
        RSDSMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['RSDSMaps'][0]
        QAirMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['QAirMaps'][0]
        RhsMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['RhsMaps'][0] 
        RSDLMaps = PathMeteo + '/' + dic[simulationID]['scenario'].loc['RSDLMaps'][0]
 
         # File inputs under different SSP scenarios        
        projection_filepath = basepath + r'/modules/urban_module/Inputs' + '/' + \
                          dic[simulationID]['scenario'].loc['projection_file'][0]
        update_years = dic[simulationID]['scenario'].loc['update_years'][0]
        urbanfrac_filepath = basepath + r'/modules/urban_module/urban_growth_files/bhima_urban_extent' + '/' + \
                         dic[simulationID]['scenario'].loc['urbanfrac_file'][0]
        excel_file_path = basepath + r'/modules/urban_module/Inputs' + '/' + 'HumanInputs220619-Final5_' + \
                          dic[simulationID]['scenario'].loc['excel_file_path'][0]
        reservoir_cca_file = basepath + r'/modules/hydro/hydro_inputs/landsurface/waterDemand' + '/'  + \
                          dic[simulationID]['scenario'].loc['reservoir_cca_file'][0]                          
                          
        
        if type(dic[simulationID]['scenario'].loc['well_type'][0]) == str:
            water_table_limit_for_pumping = basepath + r'/modules/hydro/hydro_inputs/landsurface/waterDemand' + '/' + \
                                            dic[simulationID]['scenario'].loc['well_type'][0]
        else:
            water_table_limit_for_pumping = dic[simulationID]['scenario'].loc['well_type'][0]
        
        interval = dic[simulationID]['scenario'].loc['interval'][0]
        initial_year = dic[simulationID]['scenario'].loc['initial year'][0]
        planning_period = dic[simulationID]['scenario'].loc['planning period'][0]
        tankercap_base = dic[simulationID]['scenario'].loc['tankercap'][0]
        
        t0_year = year_plan
        intervention_startyear = 1 
        intervention_endyear = 6
        
        # File inputs cwatm_settings under different interventions
        # if t0_year <= 2017:
        #     cwatm_settings = basepath + r'/modules/hydro/hydro_inputs/settings/cwatm_settings_noadd.xlsx'
        if (t0_year - initial_year - interval < intervention_startyear) or (t0_year - initial_year - interval > intervention_endyear):
            cwatm_settings = basepath + r'/modules/hydro/hydro_inputs/settings/cwatm_settings_noadd.xlsx'
            # cwatm_settings = basepath + r'/modules/hydro/hydro_inputs/settings' + '/' + dic['historical_run']['intervention'].loc['cwatm_settings'][0]
        else:
            cwatm_settings = basepath + r'/modules/hydro/hydro_inputs/settings' + '/' + dic[simulationID]['intervention'].loc['cwatm_settings'][0]
        
        # File inputs tanker cap under different interventions
        if (t0_year - initial_year - interval < intervention_startyear) or (t0_year - initial_year - interval > intervention_endyear):
            tankercap_factor = dic['historical_run']['intervention'].loc['tankercap_factor'][0]
        else:
            tankercap_factor = dic[simulationID]['intervention'].loc['tankercap_factor'][0]
        
        # File inputs gw cap under different interventions
        if (t0_year - initial_year - interval < intervention_startyear) or (t0_year - initial_year - interval > intervention_endyear):
            gw_cap = dic['historical_run']['intervention'].loc['gw_cap'][0]
        else:
            gw_cap = dic[simulationID]['intervention'].loc['gw_cap'][0]
            
        # File inputs sugarcane price factor under different interventions
        if (t0_year - initial_year - interval > intervention_endyear):
            sugar_factor = dic['historical_run']['intervention'].loc['sugarcane_price_factor'][0]
            food_factor = dic['historical_run']['intervention'].loc['food_price_factor'][0]
        else:
            sugar_factor = dic[simulationID]['intervention'].loc['sugarcane_price_factor'][0]
            food_factor = dic[simulationID]['intervention'].loc['food_price_factor'][0]
            
        # File inputs user_params_file under different interventions
        if (t0_year - initial_year - interval < intervention_startyear) or (t0_year - initial_year - interval > intervention_endyear):
            user_params_file = basepath + r'/modules/urban_module/Inputs'+ '/' + dic['historical_run']['intervention'].loc['user_params_file'][0]
        else:
            user_params_file = basepath + r'/modules/urban_module/Inputs'+ '/' + dic[simulationID]['intervention'].loc['user_params_file'][0]
        
        # File inputs irrigation efficiency under different interventions 
        if (t0_year - initial_year - interval > intervention_endyear):
            irrigation_efficiency_paddy = dic['historical_run']['intervention'].loc['irrigation_efficiency_paddy'][0]
            irrigation_efficiency_nonpaddy = dic['historical_run']['intervention'].loc['irrigation_efficiency_nonpaddy'][0]
        else:
            irrigation_efficiency_paddy = dic[simulationID]['intervention'].loc['irrigation_efficiency_paddy'][0]
            irrigation_efficiency_nonpaddy = dic[simulationID]['intervention'].loc['irrigation_efficiency_nonpaddy'][0]
            
        # File inputs pipe loss under different interventions
        if (t0_year - initial_year - interval > intervention_endyear):
            pipe_loss = dic['historical_run']['intervention'].loc['pipe_loss'][0]
        else:
            pipe_loss = dic[simulationID]['intervention'].loc['pipe_loss'][0]
            
        # File inputs interventionID under different interventions
        if (t0_year - initial_year - interval < intervention_startyear) or (t0_year - initial_year - interval > intervention_endyear):
            interventionID_tanker = dic['historical_run']['intervention'].columns[0]
        else:
            interventionID_tanker = dic[simulationID]['intervention'].columns[0]
            
        # File inputs scenarioID
        scenarioID = dic[simulationID]['scenario'].columns[0]
        
        # File inputs Solar farming land constraints
        if (t0_year - initial_year - interval > intervention_endyear):
            solar_input = dic['historical_run']['intervention'].loc['solar_input'][0]
        else:
            solar_input = dic[simulationID]['intervention'].loc['solar_input'][0]
            
        # create folder for each simulation 
        name_path = basepath + r'/modules/hydro' + '/' + name
        filepath_planning = name_path + '/' + 'hydro_outputs_' + simulationID 
        if not os.path.exists(filepath_planning):
            os.makedirs(filepath_planning)
            print("created folder : ", filepath_planning)
        else:
            print(filepath_planning, "folder already exists.")

        t0_month = 6  # start in June
        t0_day = 1
        currentDate = date(t0_year, t0_month, t0_day)

        # get 15 years average 
        t0_year_prior = t0_year - planning_period

        TestNet = self.urban.network
        print(TestNet)
        
        if update_years == 'historical':
            pp = TestNet.exogenous_inputs.population_projection[['X', 'Y', t0_year]].rename(
                columns={'X': 'lon', 'Y': 'lat'})  
        else:
            pp = TestNet.exogenous_inputs.population_projection[['X', 'Y', 2050]].rename(
                columns={'X': 'lon', 'Y': 'lat', 2050: t0_year})  
                
        # save urban_module module output
        urban_water_output_df = {}
        urban_water_output_hh_df = {}
        urban_water_output_co_df = {}
        urban_water_output_in_df = {}

        # --------------------------------read historical (15-year average) ag urban module inputs --------------------------------
        tic()
        hist_inputs = HistoricalAgUrbanInputs('planning')
        hist_inputs.avai_land()
        hist_inputs.avai_res(pp, initial_year, t0_year, t0_month, t0_day, filepath_planning, planning_period,interval)
        hist_inputs.avai_river(initial_year, t0_year, t0_month, t0_day, filepath_planning, planning_period,interval)
        hist_inputs.avai_sw()
        hist_inputs.avai_gw(initial_year, t0_year, t0_month, t0_day,filepath_planning, planning_period,interval)
        hist_inputs.hist_ET(t0_year, t0_year_prior, t0_month, t0_day, initial_year, filepath_planning, planning_period,interval)
        hist_inputs.hist_Effrain(t0_year, t0_year_prior, t0_month, t0_day, initial_year, filepath_planning, planning_period,interval)
        hist_inputs.updated_crop_water_use(0.6, 0.6) #scale to irrigation_efficiency_paddy,irrigation_efficiency_nonpaddy later
        toc()

        # --------------------------------import historical ag module inputs for optimization--------------------------------
        tic()
        ag_opt = FarmSeasonalDecisionMaker('ag land test')

        # Create each agents inputs
        # replace the ag module inputs
        ag_input_df['land_avai'] = hist_inputs.Avai_land
        ag_input_df['gw_depth'] = hist_inputs.gw_2yr_season
        ag_input_df['sw_avai'] = hist_inputs.sw_river_season
        ag_input_df['crop_wat_app'].iloc[:, 2:] = hist_inputs.water_deficit_ratio[['khy1', 'rby1', 'khy2', 'rby2']]
        ag_input_df['eff_rainfall'] = hist_inputs.Effrain_2yr
        ag_input_df['ref_evapotransp'] = hist_inputs.ET_2yr

        # change the data type of agent from float64 to int64
        ag_input_df['land_avai'] = ag_input_df['land_avai'].astype({'Agent': 'int64'})
        ag_input_df['gw_depth'] = ag_input_df['gw_depth'].astype({'Agent': 'int64'})
        ag_input_df['sw_avai'] = ag_input_df['sw_avai'].astype({'Agent': 'int64'})
        ag_input_df['eff_rainfall'] = ag_input_df['eff_rainfall'].astype({'Agent': 'int64'})
        ag_input_df['ref_evapotransp'] = ag_input_df['ref_evapotransp'].astype({'Agent': 'int64'})

        sw_river_combine = hist_inputs.sw_2yr.set_index('Agent').add(hist_inputs.river_2yr.set_index('Agent'),
                                                                     fill_value=0).reset_index()

        sw_river_combine['khy1'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in np.arange(6, 11))
        sw_river_combine['rby1'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in np.arange(11, 18))
        sw_river_combine['khy2'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in np.arange(18, 23))
        sw_river_combine['rby2'] = sum(sw_river_combine.loc[:, 'm' + str(i)] for i in [23, 24, 1, 2, 3, 4, 5])

        with pd.ExcelWriter(filepath_planning +
                            '/ag_planning_input_' + str(count_plan) + '.xlsx') as writer:
            for sheetname in ag_input_df.keys():
                ag_input_df[sheetname].to_excel(writer, index=False, sheet_name=sheetname)

        full_agent_list = ag_input_df['sw_avai'].Agent.unique()  # change agents list to your agent codes
        agent_list = full_agent_list

        # Create each agents inputs
        input_df = ag_opt.create_seasonal_agent_input_subset(agent_list, ag_input_df)

        # Modify the Inputs as Necessary
        modified_agent_inputs = {}
        
        for agent in agent_list: 
        
            crop_price = input_df[agent]['net_revenue'].copy().set_index('Crop')
        
            # Price Change ---- this section would change the price of sugarcane by the multiplier
            crops_to_change = ['SugarAdsali', 'SugarPreseasonal', 'SugarSuru1', 'SugarSuru2']
            for crop in crops_to_change:
                old_price = crop_price.loc[crop].net_revenue
                new_price = old_price * sugar_factor
                crop_price.at[crop, 'net_revenue'] = new_price
                
            foodcrops_to_change = ['Rice1', 'MaizeK1', 'MaizeR1', 'Rice2', 'MaizeK2', 'MaizeR2']
            for crop in foodcrops_to_change:
                old_price = crop_price.loc[crop].net_revenue
                new_price = old_price * food_factor
                crop_price.at[crop, 'net_revenue'] = new_price
                
        
            modified_agent_inputs[agent] = input_df[agent].copy()
            modified_agent_inputs[agent]['net_revenue'] = crop_price.reset_index()
        

        # Run the Optimization # see the ag_looping_functions for description
        agent_mod_obj_dict = ag_opt.ag_seasonal_optimization_loop(agent_list, modified_agent_inputs, interventionID_tanker,scenarioID,solar_input)

        # Extract the results # see the ag_looping_functions for description
        ag_opt_output = ag_opt.extract_seasonal_ag_outputs(agent_list, agent_mod_obj_dict,
                                                           filepath_planning +
                                                           '/ag_planning_' + str(count_plan) + '.xlsx', toExcel=True)

        # Ag output for LAND
        crop_agent_land_df = ag_opt_output['Land']

        # Ag output for surface water and groundwater
        season_crop_agent_surfacewater_df = ag_opt_output['SW']
        season_crop_agent_groundwater_df = ag_opt_output['GW']
        season_crop_agent_totalwater_df = ag_opt_output['Water']

        # Correct the data types
        season_crop_agent_surfacewater_df = season_crop_agent_surfacewater_df.astype(
            {'Month': 'string', 'Crop': 'string', 'Agent': 'float64'})
        season_crop_agent_groundwater_df = season_crop_agent_groundwater_df.astype(
            {'Month': 'string', 'Crop': 'string', 'Agent': 'float64'})
        season_crop_agent_totalwater_df = season_crop_agent_totalwater_df.astype(
            {'Month': 'string', 'Crop': 'string', 'Agent': 'float64'})

        #######################################################################################################
        # AG Seasonal output dis aggregate to monthly output
        # start with total water use (in proportion with the crop water requirement)
        # agent within CCA uses SW (and GW), if sw_use <= sw_avai, only sw (sw=total water use)
        # if sw_use > sw_avai, use sw and gw
        # gw use = total water use - surface water use
        #######################################################################################################
        # total
        Z = AgOutput_season_to_mo()
        mo_agent_totalwater_df = Z.totalwater_season_to_mo(hist_inputs.water_deficit_ratio,
                                                           season_crop_agent_totalwater_df)
        mo_agent_surfacewater_df = Z.surfacewater_season_to_mo(sw_river_combine, mo_agent_totalwater_df,
                                                               season_crop_agent_surfacewater_df)
        mo_agent_groundwater_df = Z.groundwater_season_to_mo(mo_agent_totalwater_df, mo_agent_surfacewater_df)

        # communicate variables between planning stage and implementation stag
        share.ag_input_planning = ag_input_df
        share.ag_output_planning = ag_opt_output
        share.ag_water_percent_planning = hist_inputs.water_deficit_ratio
        share.ag_sw_planning = sw_river_combine

        # Note: this uses the cwatm_time_loop
        for time_counter in range(12):
            '''
            run cwatm using time counter, starting June. Month1 data is from the initial file
            '''
            # ----------------------------------------- TIME SETTINGS - -----------------------------------------------
            if time_counter == 0:
                year = t0_year
                month = t0_month
                day = t0_day
                currentDate = date(year, month, day)
            else:
                year = currentDate.year
                month = currentDate.month
                day = currentDate.day

            # set number of Months
            numMonths = 1

            if time_counter == 0:
                priorDate = currentDate
            else:
                priorDate = currentDate - timedelta(days=1)
            nextDate = add_months(currentDate, numMonths)

            # Use the last month's information
            if time_counter == 0:
                priorDate_month = currentDate
            else:
                priorDate_month = add_months(currentDate, -numMonths)

            print('time_counter: ' + str(time_counter))
            print('year: ' + str(year))
            print('month: ' + str(month))
            print('day: ' + str(day))

            tic()
            cap = tankercap_base * tankercap_factor
            human_data = pd.read_excel(excel_file_path)
            if update_years =='historical':
                (urban_water_output_df[time_counter], urban_water_output_hh_df[time_counter],
                 urban_water_output_co_df[time_counter], urban_water_output_in_df[time_counter]) = \
                    self.urban.run_urban_module(interventionID_tanker,year, update_years, hist_inputs.gw_urban[time_counter],
                                                hist_inputs.sw_urban[time_counter],
                                                human_data, gw_cap,cap, get_excel_data(user_params_file).parse("human_parameters"))
            else:
                (urban_water_output_df[time_counter], urban_water_output_hh_df[time_counter],
                 urban_water_output_co_df[time_counter], urban_water_output_in_df[time_counter]) = \
                    self.urban.run_urban_module(interventionID_tanker,2050, update_years, hist_inputs.gw_urban[time_counter],
                                                hist_inputs.sw_urban[time_counter],
                                                human_data, gw_cap,cap, get_excel_data(user_params_file).parse("human_parameters"))                
            toc()

            tic()
            print('Now running month' + str(time_counter + 6))

            with pd.ExcelWriter(filepath_planning + '/planning_human_outputs_y'+str(count_plan+1) + "_t" + str(time_counter + 1) +'.xlsx') as writer:
                count = 0
                for sheetname in ['tot', 'hh', 'co', 'in']:
                    df = [urban_water_output_df, urban_water_output_hh_df, urban_water_output_co_df,
                          urban_water_output_in_df]
                    df[count][time_counter].to_excel(writer, index=False, sheet_name=sheetname)
                    count += 1
            
            print("Total quantities EXCEL file generated")

            # ----------------------------------------- CWATM SETTINGS  ------------------------------------------------

            cwatm_test = HydroCWATM('cwatm land test')

            PathOut_folder = filepath_planning + '/planning_y' + str(count_plan+1)
            PathRoot = basepath + r'/modules/hydro'

            PathOut = PathOut_folder + "_t" + str(time_counter + 1)
            PathOut_prior = PathOut_folder + "_t" + str(time_counter)

            if not os.path.isdir(PathOut):
                os.makedirs(PathOut)
                print("created hydro-out folder : ", PathOut)
            else:
                print(PathOut, "hydro-out folder already exists.")

            excel_settings_file = cwatm_settings
                
            settings_template = basepath + r'/modules/hydro/hydro_inputs/settings/settings_CWatM_template_Agent.ini'
            new_settings_file = filepath_planning + '/planning_CWatM_new.ini'
            

            StepStart = currentDate.strftime('%d/%m/%Y')
            StepEnd = nextDate.strftime('%d/%m/%Y')  # the start of the next month
            StepInit = (nextDate - timedelta(days=1)).strftime('%d/%m/%Y')

            if update_years == 'historical':
                urbanfrac_file = urbanfrac_filepath + r'/UBB_built-up_res_' + str(year) + '.nc' # str(year) + '.nc'
            else:
                urbanfrac_file = urbanfrac_filepath + r'/UBB_built-up_res_' + str(2050) + '.nc' # str(year) + '.nc'

            if time_counter == 0:
                initLoadDate = priorDate.strftime('%Y%m%d')  # the end of the prior month
                if year == initial_year+interval:
                    src_file = basepath + r'/modules/hydro/hydro_inputs_external/initial_condition/Bhima_preMonsoon.nc'
                else:
                    src_file = filepath_planning + '/init_implementation_in_' + \
                               (currentDate - timedelta(days=1)).strftime('%Y%m%d') + ".nc"
                dest_file = filepath_planning + '/init_planning_' + initLoadDate + ".nc"
                shutil.copy2(src_file, dest_file)

            # init date
            else:
                initLoadDate = priorDate.strftime('%Y%m%d')  # StepEnd

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # + Getting Agents + Land Available from Hydro
            # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # available land from cwatm lat,long, avail land for segments and for pixels [changes with each run]
            if time_counter == 0:  # initial conditions
                if year == initial_year+interval:
                    pix_available_frac_Irr_file = xr.open_dataarray(
                        basepath + r'/modules/hydro/hydro_inputs_external/initial_condition/fallowIrr_Time0_daily.nc')
                else:
                    Path = filepath_planning + '/planning_y' + \
                           str(count_plan) + '_t12'
                    pix_available_frac_Irr_file = xr.open_dataarray(Path + r'/fallowIrr_daily.nc')
            else:
                pix_available_frac_Irr_file = xr.open_dataarray(PathOut_prior + r'/fallowIrr_daily.nc')

            pix_available_frac_Irr_file.name = 'Pix_AvailableFrac_Irr'


            sel_date = currentDate.strftime('%Y-%m-%d')
            if time_counter == 0:
                if year == initial_year+interval:
                    pix_available_frac_Irr = pix_available_frac_Irr_file[0]
                else:
                    pix_available_frac_Irr = pix_available_frac_Irr_file.loc[sel_date]

            else:
                pix_available_frac_Irr = pix_available_frac_Irr_file.loc[sel_date]

            # print(sel_date)

            # merge and create a data frame
            merger = xr.merge([adminsegs, pix_available_frac_Irr, cellarea], join='override')
            merger = merger.drop('time')
            merger_df = merger.to_dataframe()

            # calculate the actual available area equipped with irrigation by pixel
            merger_df['Pix_AvailableArea_m2_Irr'] = merger_df.Pix_AvailableFrac_Irr * merger_df.cellArea_totalend
            merger_df['AvailableArea_1000ha_Irr'] = merger_df['Pix_AvailableArea_m2_Irr'] * 0.0001 / 1000
            seg_available_area_Irr = merger_df.groupby('Agent').sum().reset_index()[
                ['Agent', 'AvailableArea_1000ha_Irr']]

            main_df = merger_df.reset_index()
            main_df.rename(columns={'AvailableArea_1000ha_Irr': 'AvailableArea_1000ha_pixel'}, inplace=True)

            # print('Pixel and Segment Areas:')
            # print(main_df.head())

            # -------------------------------------------------------------------------------
            #                                                                  _
            #      _   ___ ___  ___        _____  ___          _   _ _   _  __| |_ -- ---
            #     /_\ / __| _ \/ _ \      |_   _|/ _ \        | |_| | | | |/ _` | '__/ _ \
            #    / _ | (_ |   | (_) |       | | | (_) |       |  _  | |_| | (_| | | | (_) |
            #   /_/ \_\___|_|_\\___/        |_|  \___/        |_| |_|\__, |\__,_|_|  \___/
            #                                                        |___/
            # --------------------------------------------------------------------------------

            # ---------------------------------------------------LAND-----------------------------------------------------------
            # Correct the datatypes
            crop_agent_land_df = crop_agent_land_df.astype(
                {'Crop': 'string', 'Agent': 'float64'})

            # Cropland
            cropland_df = crop_agent_land_df[['Crop', 'Agent', 'Land_1000ha']].groupby(
                ['Agent', 'Crop']).sum().reset_index()  # check length = number of crops x number agents

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # + Creating a netcdf of land area with calculation
            # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            outpath_ag = filepath_planning + '/ag_to_hydro'
            if not os.path.exists(outpath_ag):
                os.makedirs(outpath_ag)

            # To use this, the dataframe must have a lat and lon component
            selected_croplist_agro = croplist_agro  # selected all the crops

            for cropname in selected_croplist_agro:
                # print('Processing Crop: ' + cropname)

                # --------------------------------- CALCULATION ----------------------------------------------
                crop_df = cropland_df[cropland_df['Crop'] == cropname]
                full_df = main_df.merge(crop_df, on='Agent')
                full_df = full_df.merge(seg_available_area_Irr, on='Agent')

                full_df['frac'] = (full_df['Land_1000ha'] / full_df['AvailableArea_1000ha_Irr']) * (
                    full_df['Pix_AvailableFrac_Irr'])  
                cropfrac = full_df[['lat', 'lon', 'Crop', 'frac']]
                # --------------------------------------------------------------------------------------------

                df_in_land = cropfrac

                if ('lat' in df_in_land.columns) and ('lon' in df_in_land.columns):
                    array_in_land = create_array(df_in_land, lat_lon, 'frac')
                    outfile_land = outpath_ag + '/' + 'planning_' + str(time_counter) + '_' + cropname + "_" + str(count_plan) + '.nc'

                    create_netcdf_from_array(nc_template_file, outfile_land, array_in_land, cropname + "_area")

                else:
                    print('lat or lon is not a column name -- check your input dataframe')
                # print(cropname + " netcdf file created")

            # --------------------------------------------GROUNDWATER-----------------------------------------------------------
            print('generating gw use file for' + Months[time_counter])
            var_gw = mo_agent_groundwater_df[['Agent', Months[time_counter] + '_gw_use']]
            merger_gw = adminsegs_df.merge(var_gw, on='Agent')
            merger_gw['gw_m3'] = merger_gw[Months[time_counter] + '_gw_use'] * 1e+9  * 0.6 / irrigation_efficiency_nonpaddy # change the unit from km3 to m3

            df_in_gw = merger_gw

            if ('lat' in df_in_gw.columns) and ('lon' in df_in_gw.columns):
                array_in_gw = create_array(df_in_gw, lat_lon, 'gw_m3')
                outfile_gw = outpath_ag + r'/planning_gw_' + Months[time_counter] + "_" + str(count_plan) + '.nc'
                create_netcdf_from_array(nc_template_file, outfile_gw, array_in_gw, 'gw_m3')

            else:
                print('lat or lon is not a column name -- check your input dataframe')
            # print("netcdf file created")

            # --------------------------------------------SURFACE WATER-------------------------------------------------
            print('generating sw use file for' + Months[time_counter])
            var_sw = mo_agent_surfacewater_df[['Agent', Months[time_counter] + '_sw_use']]
            merger_sw = adminsegs_df.merge(var_sw, on='Agent')
            merger_sw['sw_m3'] = merger_sw[Months[time_counter] + '_sw_use'] * 1e+9  * 0.6 / irrigation_efficiency_nonpaddy # change the unit from km3 to m3

            df_in_sw = merger_sw

            if ('lat' in df_in_sw.columns) and ('lon' in df_in_sw.columns):
                array_in_sw = create_array(df_in_sw, lat_lon, 'sw_m3')
                outfile_sw = outpath_ag + r'/planning_sw_' + Months[time_counter] + "_" + str(count_plan) + '.nc'
                create_netcdf_from_array(nc_template_file, outfile_sw, array_in_sw, 'sw_m3')

            else:
                print('lat or lon is not a column name -- check your input dataframe')
            # print("netcdf file created")

            # -----------------------------------------------------------------------------------------
            #                _                                                              _
            #    | | | |_ __| |__   __ _ _ __           _____  ___        | | | | _   _  __| |_-- ---
            #    | | | | '__| '_ \ / _` | '_ \         |_   _|/ _ \       | |_| | | | |/ _` | '__/ _ \
            #    | |_| | |  | |_) | (_| | | | |          | | | (_) |      |  _  | |_| | (_| | | | (_) |
            #     \___/|_|  |_.__/ \__,_|_| |_|          |_|  \___/       |_| |_|\__, |\__,_|_|  \___/
            #                                                                    |___/
            # ----------------------------------------------------------------------------------------

            urban_CWatM_df_in = urban_df_to_cwatm(urban_water_output_df[time_counter], urbanfrac_file, year, pipe_loss)

            outpath_urban = filepath_planning + '/urban_to_hydro'
            if not os.path.exists(outpath_urban):
                os.makedirs(outpath_urban)

            if ('lat' in urban_CWatM_df_in.columns) and ('lon' in urban_CWatM_df_in.columns):
                array_in_urban_sw = create_array(urban_CWatM_df_in, lat_lon, 'piped_m3/month_with_loss')
                outfile_urban_sw = outpath_urban + '/' + 'urban_CWatM_sw_t' + str(time_counter) + "_" + str(count_plan) + '.nc'
                create_netcdf_from_array(nc_template_file, outfile_urban_sw, array_in_urban_sw, 'sw_m3')

            else:
                print('lat or lon is not a column name -- check your input dataframe')
            # print("netcdf file created")

            if ('lat' in urban_CWatM_df_in.columns) and ('lon' in urban_CWatM_df_in.columns):
                array_in_urban_gw = create_array(urban_CWatM_df_in, lat_lon, 'non_piped_m3/mo_new')
                outfile_urban_gw = outpath_urban + '/' + 'urban_CWatM_gw_t' + str(time_counter) + "_" + str(count_plan) + '.nc'
                create_netcdf_from_array(nc_template_file, outfile_urban_gw, array_in_urban_gw, 'gw_m3')

            else:
                print('lat or lon is not a column name -- check your input dataframe')
            # print("netcdf file created")
            toc()
            # --------------------------------
            #    _   _           _
            #   | | | |_   _  __| |_ __ ___
            #   | |_| | | | |/ _` | '__/ _ \
            #   |  _  | |_| | (_| | | | (_) |
            #   |_| |_|\__, |\__,_|_|  \___/
            #          |___/
            # --------------------------------

            # ----------------------------------------- CWATM SETTINGS DICTIONARY --------------------------------------
            tic()
            print('Creating Settings for CWATM')
            # Create settings dictionary
            settings_dict = cwatm_test.create_cwatm_settings_dict_obj(excel_settings_file)

            # Time Settings
            settings_dict['StepStart'] = StepStart  # 6
            settings_dict['StepEnd'] = StepEnd  # 7
            settings_dict[
                'initLoad'] = filepath_planning + '/init_planning_' + initLoadDate + '.nc'  # 5
            settings_dict['initSave'] = filepath_planning + '/init' + '_' + 'planning'
            settings_dict['StepInit'] = StepInit

            # paths
            settings_dict['PathRoot'] = PathRoot
            settings_dict['PathOut'] = PathOut
            settings_dict['Excel_settings_file'] = excel_settings_file

            # LAND USES/ LAND COVERS
            settings_dict['leftoverIrrigatedCropIsRainfed'] = 0
            settings_dict['irrPaddy_efficiency'] = irrigation_efficiency_paddy
            settings_dict['irrNonPaddy_efficiency'] = irrigation_efficiency_nonpaddy

            # urban_module
            settings_dict['sealed_fracVegCover'] = urbanfrac_file

            settings_dict['domestic_agent_GW_request_month_m3'] = outpath_urban + "/" + \
                                                                  'urban_CWatM_gw_t' + str(time_counter) + "_" + str(count_plan) + '.nc'

            settings_dict['domestic_agent_SW_request_month_m3'] = outpath_urban + "/" + \
                                                                  'urban_CWatM_sw_t' + str(time_counter) + "_" + str(count_plan) + '.nc'

            # Climate
            settings_dict['PathMeteo'] = PathMeteo
        
            settings_dict['WindMaps'] = WindMaps 
            settings_dict['PrecipitationMaps'] = PreciMaps
            settings_dict['TavgMaps'] = TavgMaps 
            settings_dict['TminMaps'] = TminMaps
            settings_dict['TmaxMaps'] = TmaxMaps
            settings_dict['PSurfMaps'] = PSurfMaps
            settings_dict['RSDSMaps'] = RSDSMaps
            settings_dict['QAirMaps'] = QAirMaps
            settings_dict['RhsMaps'] = RhsMaps
            settings_dict['RSDLMaps'] = RSDLMaps

            # Others
            settings_dict['reservoir_command_areas'] = reservoir_cca_file

            settings_dict['water_table_limit_for_pumping'] = water_table_limit_for_pumping


            if time_counter == 0:
                if year == initial_year+interval:
                    settings_dict['init_water_table'] = basepath + r'/modules/hydro/hydro_inputs/Modflow/modflow_inputs' \
                                                                   r'/modflow_watertable_totalend.nc'
                else:
                    settings_dict[
                        'init_water_table'] = filepath_planning + \
                                              '/final_y' + str(count_plan) + '_t12' + '/modflow_watertable_totalend.nc'
            else:
                settings_dict[
                    'init_water_table'] = filepath_planning + '/planning_y' + str(count_plan+1) + '_t' + str(time_counter) + r'/modflow_watertable_totalend.nc'

            sum_irr_data = np.empty([320, 370])

            croplist_hydro.sort()
            croplist_agro.sort()
            special_crop = ['Rice1', 'Rice2']
            agro_list = [ele for ele in croplist_agro if ele not in special_crop]
            cropname_dict = dict(zip(croplist_hydro, agro_list))

            for cropname in croplist_hydro:  # hydro
                zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                settings_dict[cropname + "_Irr"] = zeros_array
                # settings_dict[cropname + "_nonIrr"] = zeros_array
                settings_dict['irrPaddy_fracVegCover'] = zeros_array

            if time_counter == 0:  # June
                for cropname_kharif in Kharif_crop_list:  # hydro
                    if 'planning_' + str(time_counter) + "_" + cropname_dict[cropname_kharif] + "_" + str(count_plan) + \
                            ".nc" in os.listdir(outpath_ag):
                        print(cropname_kharif)

                        nc_path = outpath_ag + '/' + 'planning_' + str(time_counter) + "_" + cropname_dict[
                            cropname_kharif] + "_" + str(count_plan) + '.nc'

                        settings_dict[cropname_kharif + "_Irr"] = nc_path
                        # settings_dict[cropname_kharif + "_nonIrr"] = nc_path

                        nc = xr.open_dataset(nc_path)
                        varname = list(nc.data_vars)
                        data = nc.variables[varname[0]].data
                        # plt.figure()
                        # plt.imshow(data); plt.colorbar(); plt.title(crop)
                        sum_irr_data = sum_irr_data + data
                    else:
                        # need a zero's netcdf
                        zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                        settings_dict[cropname_kharif + "_Irr"] = zeros_array
                        # settings_dict[cropname_kharif + "_nonIrr"] = zeros_array

            if time_counter in range(6):
                settings_dict['irrPaddy_fracVegCover'] = outpath_ag + "/" + 'planning_' + str(
                    time_counter) + "_" + "Rice1" + "_" + str(count_plan) + ".nc"

            if time_counter == 5:  # November
                for cropname_rabi in Rabi_crop_list:  # hydro
                    if 'planning_' + str(time_counter) + "_" + cropname_dict[cropname_rabi] + "_" + str(count_plan) + \
                            ".nc" in os.listdir(outpath_ag):
                        print(cropname_rabi)

                        nc_path = outpath_ag + '/' + 'planning_' + str(time_counter) + "_" + cropname_dict[
                            cropname_rabi] + "_" + str(count_plan) + '.nc'

                        settings_dict[cropname_rabi + "_Irr"] = nc_path
                        # settings_dict[cropname_rabi + "_nonIrr"] = nc_path

                        nc = xr.open_dataset(nc_path)
                        varname = list(nc.data_vars)
                        data = nc.variables[varname[0]].data
                        # plt.figure()
                        # plt.imshow(data); plt.colorbar(); plt.title(crop)
                        sum_irr_data = sum_irr_data + data
                    else:
                        # need a zero's netcdf
                        zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                        settings_dict[cropname_rabi + "_Irr"] = zeros_array
                        # settings_dict[cropname_rabi + "_nonIrr"] = zeros_array

            if time_counter == 7:  # Jan
                for cropname_jan in Jan_crop_list:  # hydro
                    if 'planning_' + str(time_counter) + "_" + cropname_dict[cropname_jan] + "_" + str(count_plan) + \
                            ".nc" in os.listdir(outpath_ag):
                        print(cropname_jan)

                        nc_path = outpath_ag + '/' + 'planning_' + str(time_counter) + "_" + cropname_dict[
                            cropname_jan] + "_" + str(count_plan) + '.nc'

                        settings_dict[cropname_jan + "_Irr"] = nc_path
                        # settings_dict[cropname_jan + "_nonIrr"] = nc_path

                        nc = xr.open_dataset(nc_path)
                        varname = list(nc.data_vars)
                        data = nc.variables[varname[0]].data
                        # plt.figure()
                        # plt.imshow(data); plt.colorbar(); plt.title(crop)
                        sum_irr_data = sum_irr_data + data
                    else:
                        # need a zero's netcdf
                        zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs/zeros_array.nc'

                        settings_dict[cropname_jan + "_Irr"] = zeros_array
                        # settings_dict[cropname_jan + "_nonIrr"] = zeros_array

            settings_dict['irrigation_agent_SW_request_month_m3'] = outpath_ag + r'/planning_sw_' + Months[
                time_counter] + "_" + str(count_plan) + '.nc'
            settings_dict['irrigation_agent_GW_request_month_m3'] = outpath_ag + r'/planning_gw_' + Months[
                time_counter] + "_" + str(count_plan) + '.nc'

            # OUTPUTS Settings

            # select outputs ------------------------------------------------------------------------------------------
            settings_dict['save_initial'] = 1

            settings_dict['OUT_Map_MonthEnd'] = open(basepath + r"/OUT_MAP_MonthEnd.txt", "r").readline()
            settings_dict['OUT_MAP_Daily'] = open(basepath + r"/OUT_MAP_Daily.txt", "r").readline()
            settings_dict['OUT_Map_MonthTot'] = open(basepath + r"/OUT_MAP_MonthTot.txt", "r").readline()
            settings_dict['OUT_Map_MonthAvg'] = open(basepath + r"/OUT_MAP_MonthAvg.txt", "r").readline()
            settings_dict['OUT_MAP_AnnualTot'] = ""
            settings_dict['OUT_MAP_TotalEnd'] = open(basepath + r"/OUT_MAP_TotalEnd.txt", "r").readline()
            # ----------------------------------------- RUNNING CWATM  -------------------------------------------------
            # create the new settings file
            cwatm_test.define_cwatm_settings(settings_template, new_settings_file, settings_dict)

            cwatm_test.save_cwatm_settings_file(new_settings_file, settings_dict['PathOut'], '_settings')
            toc()
            # run cwatm
            tic()
            cwatm_test.run_cwatm(new_settings_file)
            toc()

            currentDate = nextDate
            print(currentDate)

            # close the netcdf file every time when open it
            for cropname in croplist_hydro:
                file = outpath_ag + '/' + 'planning_' + str(time_counter) + "_" + cropname_dict[cropname] + "_" + str(count_plan) + '.nc'
                open_Chickpea = xr.open_dataset(file)
                open_Chickpea.close()


