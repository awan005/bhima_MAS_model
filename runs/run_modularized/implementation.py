__author__ = 'ankunwang'

import os
import shutil
import sys

from helper_objects.helper_functions import toc, tic, create_array, create_netcdf_from_array, add_months, get_excel_data
# Hydro module
from netsim.model_components.institutions.cwatm_decisions import HydroCWATM
# Ag module
from netsim.model_components.institutions.farm_seasonal_decisions import FarmSeasonalDecisionMaker

import numpy as np
import xarray as xr
import pandas as pd
from global_var import adminsegs, cellarea, lat_lon, nc_template_file, adminsegs_df, Months,\
    croplist_hydro, Kharif_crop_list, Rabi_crop_list, Jan_crop_list, croplist_agro, month_days
from datetime import date, timedelta
from basepath_file import basepath
from runs.run_modularized.reservoir_rule import sw_interaction

from runs.run_modularized.prepare_implementation_inputs import ImplementationInputs
from runs.run_modularized.urban_to_cwatm import urban_df_to_cwatm
from runs.run_modularized.ag_outputs_season_to_mo import AgOutput_season_to_mo

from scenario_intervention_input import dic


class implementationstage(object):
    def __init__(self, model):
        self.model = model 
        self.urban = self.model.urban

    def run(self, share, year_imple, count_imple, para, name):
        """
        Implementation stage of the integrated model
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
        
        t0_year = year_imple
        intervention_startyear = 1
        intervention_endyear = 4
        
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
            
        #  File inputs scenarioID
        scenarioID = dic[simulationID]['scenario'].columns[0]
        
        # File inputs Solar farming land constraints
        if (t0_year - initial_year - interval > intervention_endyear):
            solar_input = dic['historical_run']['intervention'].loc['solar_input'][0]
        else:
            solar_input = dic[simulationID]['intervention'].loc['solar_input'][0]
        
        name_path = basepath + r'/modules/hydro' + '/' + name
        filepath_sim = name_path + '/' + 'hydro_outputs_' + simulationID
        filepath_planning = filepath_sim + '/planning_y' + str(count_imple + 1)
        filepath_implementation = filepath_sim + '/implementation_y' + str(count_imple + 1)
        filepath_final = filepath_sim + '/final_y' + str(count_imple + 1)

        t0_month = 6  # start in June
        t0_day = 1
        currentDate = date(t0_year, t0_month, t0_day)

        # save urban_module module output
        urban_water_output_df = {}
        urban_water_output_hh_df = {}
        urban_water_output_co_df = {}
        urban_water_output_in_df = {}

        TestNet = self.urban.network
        if update_years == 'historical':
            pp = TestNet.exogenous_inputs.population_projection[['X', 'Y', t0_year]].rename(
                columns={'X': 'lon', 'Y': 'lat'})  
        else:
            pp = TestNet.exogenous_inputs.population_projection[['X', 'Y', 2050]].rename(
                columns={'X': 'lon', 'Y': 'lat', 2050: t0_year})  
        # ++++++

        res_initial_out_season = {}
        release_out_season = {}

        # Note: this uses the cwatm_time_loop
        for time_counter in range(12):
            tic()
            '''
            run cwatm using time counter, starting June. 
            Month1 data is from the initial file
            update the Planning stage seasonally, 
            including kharif season (time counter 0-4: June-October) and Rabi season (time counter 5-11: Nov-May)
            '''
            # ----------------------------------------- TIME SETTINGS - ------------------------------------------------
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

            # ----------------------------------------- CWATM SETTINGS  ------------------------------------------------

            cwatm_test = HydroCWATM('cwatm land test')

            PathOut_folder = filepath_implementation

            PathRoot = basepath + r'/modules/hydro'

            PathOut = PathOut_folder + "_t" + str(time_counter + 1)
            PathOut_prior = PathOut_folder + "_t" + str(time_counter)

            if not os.path.isdir(PathOut):
                os.makedirs(PathOut)
                print("created hydro-out folder : ", PathOut)
            else:
                print(PathOut, "hydro-out folder already exists.")

            # File inputs
            excel_settings_file = cwatm_settings
            settings_template = basepath + r'/modules/hydro/hydro_inputs/settings/settings_CWatM_template_Agent.ini'
            new_settings_file = filepath_sim + '/implementation_CWatM_new.ini'

            StepStart = currentDate.strftime('%d/%m/%Y')
            StepEnd = nextDate.strftime('%d/%m/%Y')  # the start of the next month
            StepInit = (nextDate - timedelta(days=1)).strftime('%d/%m/%Y')

            if update_years == 'historical':
                urbanfrac_file = urbanfrac_filepath + r'/UBB_built-up_res_' + str(year) + '.nc' # str(year) + '.nc'
            else:
                urbanfrac_file = urbanfrac_filepath + r'/UBB_built-up_res_' + str(2050) + '.nc' # str(year) + '.nc'

            if time_counter == 0:
                # the source file should be the init file from the last month in the planning module stage
                initLoadDate = priorDate.strftime('%Y%m%d')  # the end of the prior month
                src_file = filepath_sim + '/init_planning_' + initLoadDate + '.nc'
                dest_file = filepath_sim + '/init_implementation_' + initLoadDate + ".nc"
                shutil.copy2(src_file, dest_file)

            elif time_counter == 5:
                # the source file should be the init file from the last month in the planning module stage
                initLoadDate = priorDate.strftime('%Y%m%d')  # the end of the prior month
                src_file = filepath_sim + '/init_implementation_in_' + initLoadDate + '.nc'
                dest_file = filepath_sim + '/init_implementation_' + initLoadDate + ".nc"
                shutil.copy2(src_file, dest_file)

            # init date
            else:
                initLoadDate = priorDate.strftime('%Y%m%d')  # StepEnd

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # + Getting Agents + Land Available from Hydro
            # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # available land from cwatm lat,long, avail land for segments and for pixels [changes with each run]
            if time_counter == 0:  # initial conditions

                # change the available arable land to available land that is equipped with irrigation
                pix_available_frac_Irr_file = xr.open_dataarray(
                    basepath + r'/modules/hydro/hydro_inputs_external/initial_condition/fallowIrr_Time0_daily.nc')

            elif time_counter == 5:
                Path = filepath_final + '_t' + str(time_counter)
                pix_available_frac_Irr_file = xr.open_dataarray(Path + r'/fallowIrr_daily.nc', decode_times=False)
                reference_date = priorDate_month.strftime("%Y-%m-%d")
                pix_available_frac_Irr_file['time'] = \
                    pd.date_range(start=reference_date, periods=pix_available_frac_Irr_file.sizes['time'], freq='D')

            else:
                # change the available arable land to available land that is equipped with irrigation
                pix_available_frac_Irr_file = xr.open_dataarray(PathOut_prior + r'/fallowIrr_daily.nc',
                                                                decode_times=False)
                reference_date = priorDate_month.strftime("%Y-%m-%d")
                pix_available_frac_Irr_file['time'] = \
                    pd.date_range(start=reference_date, periods=pix_available_frac_Irr_file.sizes['time'], freq='D')

            pix_available_frac_Irr_file.name = 'Pix_AvailableFrac_Irr'

            sel_date = currentDate.strftime('%Y-%m-%d')
            if time_counter == 0:
                pix_available_frac_Irr = pix_available_frac_Irr_file[0]
            else:
                pix_available_frac_Irr = pix_available_frac_Irr_file.loc[sel_date]

            # print(sel_date)

            # merge and create a data frame
            merger = xr.merge([adminsegs, pix_available_frac_Irr, cellarea])
            merger = merger.drop('time')
            merger_df = merger.to_dataframe()

            # calculate the actual available area equipped with irrigation by pixel
            merger_df['Pix_AvailableArea_m2_Irr'] = merger_df.Pix_AvailableFrac_Irr * merger_df.cellArea_totalend
            merger_df['AvailableArea_1000ha_Irr'] = merger_df['Pix_AvailableArea_m2_Irr'] * 0.0001 / 1000
            seg_available_area_Irr = merger_df.groupby('Agent').sum().reset_index()[
                ['Agent', 'AvailableArea_1000ha_Irr']]

            main_df = merger_df.reset_index()
            main_df.rename(columns={'AvailableArea_1000ha_Irr': 'AvailableArea_1000ha_pixel'}, inplace=True)

            # -------------------------------------------------------------------------------
            #                                                                  _
            #      _   ___ ___  ___        _____  ___          _   _ _   _  __| |_ -- ---
            #     /_\ / __| _ \/ _ \      |_   _|/ _ \        | |_| | | | |/ _` | '__/ _ \
            #    / _ | (_ |   | (_) |       | | | (_) |       |  _  | |_| | (_| | | | (_) |
            #   /_/ \_\___|_|_\\___/        |_|  \___/        |_| |_|\__, |\__,_|_|  \___/
            #                                                        |___/
            # --------------------------------------------------------------------------------

            # ---------------------------------------------------LAND-----------------------------------------------------------
            if time_counter < 5:
                ag_opt_output = share.ag_output_planning
            else:
                ag_opt_output = share.ag_output_implementation

            # Ag output for LAND
            crop_agent_land_df = ag_opt_output['Land']

            # Correct the datatypes
            crop_agent_land_df = crop_agent_land_df.astype(
                {'Crop': 'string', 'Agent': 'float64'})

            # Cropland
            cropland_df = crop_agent_land_df[['Crop', 'Agent', 'Land_1000ha']].groupby(
                ['Agent', 'Crop']).sum().reset_index()  

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # + Creating a netcdf of land area with calculation
            # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            outpath_ag = filepath_sim + '/ag_to_hydro'
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
                # print('Calculating the crop fraction')
                # Calculating fraction of pixel that is cropped
                full_df['frac'] = (full_df['Land_1000ha'] / full_df['AvailableArea_1000ha_Irr']) * (
                    full_df['Pix_AvailableFrac_Irr'])  
                cropfrac = full_df[['lat', 'lon', 'Crop', 'frac']]
                # --------------------------------------------------------------------------------------------

                df_in_land = cropfrac

                if ('lat' in df_in_land.columns) and ('lon' in df_in_land.columns):
                    array_in_land = create_array(df_in_land, lat_lon, 'frac')
                    outfile_land = outpath_ag + '/' + 'implementation_' + str(time_counter) + '_' + cropname + "_" + \
                                   str(count_imple) + '.nc'
                    create_netcdf_from_array(nc_template_file, outfile_land, array_in_land, cropname + "_area")

                else:
                    print('lat or lon is not a column name -- check your input dataframe')
                # print(cropname + " netcdf file created")

            # ----------------------------------
            #    _   _      _
            #   | | | |_ __| |__   __ _ _ __
            #   | | | | '__| '_ \ / _` | '_ \
            #   | |_| | |  | |_) | (_| | | | |
            #    \___/|_|  |_.__/ \__,_|_| |_|
            #
            # ----------------------------------
            T = ImplementationInputs('implementation')
            T.groundwater_urban(filepath_planning, time_counter, year, month)
            T.surfacewater_urban(t0_year, pp, filepath_planning, time_counter, year, month)
            # -----------------------------------------Surface water----------------------------------------------------
            cap = tankercap_base * tankercap_factor
            human_data = pd.read_excel(excel_file_path)
            if update_years == 'historical':
                (urban_water_output_df[time_counter], urban_water_output_hh_df[time_counter],
                 urban_water_output_co_df[time_counter], urban_water_output_in_df[time_counter]) = \
                    self.urban.run_urban_module(interventionID_tanker, year,update_years, T.gw_urban[time_counter],T.sw_urban[time_counter], human_data, gw_cap,cap, get_excel_data(user_params_file).parse("human_parameters"))
            else:
                (urban_water_output_df[time_counter], urban_water_output_hh_df[time_counter],
                 urban_water_output_co_df[time_counter], urban_water_output_in_df[time_counter]) = \
                    self.urban.run_urban_module(interventionID_tanker, 2050,update_years, T.gw_urban[time_counter],T.sw_urban[time_counter], human_data, gw_cap,cap, get_excel_data(user_params_file).parse("human_parameters"))
                    
            print('Now running month' + str(time_counter + 6))

            with pd.ExcelWriter(filepath_implementation + "_t" + str(time_counter + 1) +  '/implementation_human_outputs.xlsx') as writer:
                count = 0
                for sheetname in ['tot', 'hh', 'co', 'in']:
                    df = [urban_water_output_df, urban_water_output_hh_df, urban_water_output_co_df, urban_water_output_in_df]
                    df[count][time_counter].to_excel(writer, index=False, sheet_name=sheetname)
                    count += 1
            
            print("Total quantities EXCEL file generated")

            urban_CWatM_df_in = urban_df_to_cwatm(urban_water_output_df[time_counter], urbanfrac_file, year, pipe_loss)

            outpath_urban = filepath_sim + '/urban_to_hydro'
            if not os.path.exists(outpath_urban):
                os.makedirs(outpath_urban)

            if ('lat' in urban_CWatM_df_in.columns) and ('lon' in urban_CWatM_df_in.columns):
                array_in_urban_sw = create_array(urban_CWatM_df_in, lat_lon, 'piped_m3/month_with_loss')
                outfile_urban_sw = outpath_urban + '/' + 'urban_CWatM_implementation_sw_t' + str(time_counter) \
                                   + "_" + str(count_imple) + '.nc'
                create_netcdf_from_array(nc_template_file, outfile_urban_sw, array_in_urban_sw, 'sw_m3')

            else:
                print('lat or lon is not a column name -- check your input dataframe')
            # print("netcdf file created")

            if ('lat' in urban_CWatM_df_in.columns) and ('lon' in urban_CWatM_df_in.columns):
                array_in_urban_gw = create_array(urban_CWatM_df_in, lat_lon, 'non_piped_m3/mo_new')
                outfile_urban_gw = outpath_urban + '/' + 'urban_CWatM_implementation_gw_t' + str(time_counter) \
                                   + "_" + str(count_imple) + '.nc'
                create_netcdf_from_array(nc_template_file, outfile_urban_gw, array_in_urban_gw, 'gw_m3')

            else:
                print('lat or lon is not a column name -- check your input dataframe')
            # print("netcdf file created")

            # ---------------------------------------------
            #     _______          __  _______ __  __
            #    / ____\ \        / /\|__   __|  \/  |
            #   | |     \ \  /\  / /  \  | |  | \  / |
            #   | |      \ \/  \/ / /\ \ | |  | |\/| |
            #   | |____   \  /\  / ____ \| |  | |  | |
            #    \_____|   \/  \/_/    \_\_|  |_|  |_|
            # ----------------------------------------------

            # ----------------------------------------- CWATM SETTINGS DICTIONARY -------------------------------------
            print('Creating Settings for CWATM')
            # Create settings dictionary
            settings_dict = cwatm_test.create_cwatm_settings_dict_obj(excel_settings_file)

            # Time Settings
            settings_dict['StepStart'] = StepStart  # 6
            settings_dict['StepEnd'] = StepEnd  # 7
            settings_dict[
                'initLoad'] = filepath_sim + '/init_implementation_' + initLoadDate + '.nc'
            settings_dict['initSave'] = filepath_sim + '/init_implementation'
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

            settings_dict[
                'domestic_agent_GW_request_month_m3'] = outpath_urban + "/" + 'urban_CWatM_implementation_gw_t' \
                                                        + str(time_counter) + "_" + str(count_imple) + '.nc'

            settings_dict[
                'domestic_agent_SW_request_month_m3'] = outpath_urban + "/" + 'urban_CWatM_implementation_sw_t' \
                                                        + str(time_counter) + "_" + str(count_imple) + '.nc'

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
            
            # Other
            settings_dict['reservoir_command_areas'] = reservoir_cca_file

            settings_dict['water_table_limit_for_pumping'] = water_table_limit_for_pumping


            if time_counter == 0:
                if year == initial_year+interval:
                    settings_dict['init_water_table'] = basepath + r'/modules/hydro/hydro_inputs/Modflow/modflow_inputs'\
                                                                   r'/modflow_watertable_totalend.nc'
                else:
                    settings_dict[
                        'init_water_table'] = filepath_sim + '/final_y' + str(count_imple) + '_t12' + '/modflow_watertable_totalend.nc'
            elif time_counter == 5:
                settings_dict[
                    'init_water_table'] = filepath_final + '_t5' + '/modflow_watertable_totalend.nc'
            else:
                settings_dict[
                    'init_water_table'] = filepath_implementation + '_t' + str(time_counter) + r'/modflow_watertable_totalend.nc'


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
                    if 'implementation_' + str(time_counter) + "_" + cropname_dict[cropname_kharif] + "_" + str(count_imple) \
                            + ".nc" in os.listdir(outpath_ag):
                        print(cropname_kharif)

                        nc_path = outpath_ag + '/' + 'implementation_' + str(time_counter) + "_" + \
                                  cropname_dict[cropname_kharif] + "_" + str(count_imple) + '.nc'

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
                settings_dict['irrPaddy_fracVegCover'] = outpath_ag + "/" + 'implementation_' + str(time_counter) + "_" \
                                                         + "Rice1" + "_" + str(count_imple) + ".nc"

            if time_counter == 5:  # November
                for cropname_rabi in Rabi_crop_list:  # hydro
                    if 'implementation_' + str(time_counter) + "_" + cropname_dict[cropname_rabi] + "_" + str(count_imple) + \
                            ".nc" in os.listdir(outpath_ag):
                        print(cropname_rabi)

                        nc_path = outpath_ag + '/' + 'implementation_' + str(time_counter) + "_" + cropname_dict[
                            cropname_rabi] + "_" + str(count_imple) + '.nc'

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
                    if 'implementation_' + str(time_counter) + "_" + cropname_dict[cropname_jan] + "_" + str(count_imple)\
                            + ".nc" in os.listdir(outpath_ag):
                        print(cropname_jan)

                        nc_path = outpath_ag + '/' + 'implementation_' + str(time_counter) + "_" + cropname_dict[
                            cropname_jan] + "_" + str(count_imple) + '.nc'

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
                        zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                        settings_dict[cropname_jan + "_Irr"] = zeros_array
                        # settings_dict[cropname_jan + "_nonIrr"] = zeros_array

            settings_dict['irrigation_agent_SW_request_month_m3'] = outpath_ag + r'/planning_sw_' + Months[
                time_counter] + "_" + str(count_imple) + '.nc'
            settings_dict['irrigation_agent_GW_request_month_m3'] = outpath_ag + r'/planning_gw_' + Months[
                time_counter] + "_" + str(count_imple) + '.nc'

            # OUTPUTS Settings

            # select outputs -------------------------------------------------------------------------------------------
            settings_dict['save_initial'] = 1
            settings_dict['OUT_Map_MonthEnd'] = open(basepath + r"/OUT_MAP_MonthEnd.txt", "r").readline()
            settings_dict['OUT_MAP_Daily'] = open(basepath + r"/OUT_MAP_Daily.txt", "r").readline()
            settings_dict['OUT_Map_MonthTot'] = open(basepath + r"/OUT_MAP_MonthTot.txt", "r").readline()
            settings_dict['OUT_Map_MonthAvg'] = open(basepath + r"/OUT_MAP_MonthAvg.txt", "r").readline()
            settings_dict['OUT_MAP_AnnualTot'] = ""
            settings_dict['OUT_MAP_TotalEnd'] = open(basepath + r"/OUT_MAP_TotalEnd.txt", "r").readline()
            # ----------------------------------------- RUNNING CWATM  ------------------------------------------------
            # create the new settings file
            cwatm_test.define_cwatm_settings(settings_template, new_settings_file, settings_dict)

            cwatm_test.save_cwatm_settings_file(new_settings_file, settings_dict['PathOut'], '_settings')

            # run cwatm
            cwatm_test.run_cwatm(new_settings_file)

            # close the netcdf file every time when open it
            for cropname in croplist_hydro:
                file = outpath_ag + '/' + 'implementation_' + str(time_counter) + "_" + cropname_dict[
                    cropname] + "_" + str(count_imple) + '.nc'
                open_Chickpea = xr.open_dataset(file)
                open_Chickpea.close()

            # -------------------------------
            #      _   ___ ___  ___
            #     /_\ / __| _ \/ _ \
            #    / _ | (_ |   | (_) |
            #   /_/ \_\___|_|_\\___/
            # --------------------------------

            # season_end = [4, 11, 16, 23]
            season_end = [4, 11]

            if time_counter in season_end:
                index = season_end.index(time_counter)
                if index == 0:
                    season_range = np.arange(season_end[index] + 1)
                else:
                    season_range = np.arange(season_end[index - 1] + 1, season_end[index] + 1)
                season_length = len(season_range) - 1

                currentDate_in = add_months(currentDate, -season_length)

                ava_land = pd.DataFrame()
                gw_file = {}
                sw_file = {}
                river_file = {}
                sw_river_combine = {}
                ET_file = {}
                Effrain_file = {}
                grs_file = {}

                if index == 0:
                    ag_input_df = share.ag_input_planning
                else:
                    ag_input_df = share.ag_input_kharif

                count = 0
                for t_in in season_range:
                    output_folder_in = filepath_implementation
                    output_path_in = output_folder_in + "_t" + str(t_in + 1)
                    variables_daily = ['lakeResStorage', 'gwdepth_adjusted', 'readAvlChannelStorageM', 'lakeResInflowM',
                                       'lakeResOutflowM', 'leakage', 'EvapWaterBodyM']
                    ET_path = output_path_in + r"/ETRefAverage_segments_monthtot.nc"
                    Rainfall_path = output_path_in + r"/precipEffectiveAverage_segments_monthtot.nc"

                    # files -- daily sw, gw, river; monthly effective rainfall, ET
                    Reservoirwater = xr.open_dataarray(output_path_in + "/" + variables_daily[0] + "_" + "daily.nc")
                    Groundwater = xr.open_dataarray(output_path_in + "/" + variables_daily[1] + "_" + "daily.nc")
                    Riverwater = xr.open_dataarray(output_path_in + "/" + variables_daily[2] + "_" + "daily.nc")
                    Inflow = xr.open_dataarray(PathOut + "/" + variables_daily[3] + "_" + "daily.nc")
                    Outflow = xr.open_dataarray(PathOut + "/" + variables_daily[4] + "_" + "daily.nc")
                    Leakage = xr.open_dataarray(PathOut + "/" + variables_daily[5] + "_" + "daily.nc")
                    Evap = xr.open_dataarray(PathOut + "/" + variables_daily[6] + "_" + "daily.nc")

                    ref_date = add_months(currentDate_in, count)
                    ET = xr.open_dataarray(ET_path, decode_times=False)
                    reference_date = ref_date.strftime(
                        "%Y-%m-%d")  # the start date of the simulation results from CWatM
                    ET['time'] = pd.date_range(start=reference_date, periods=ET.sizes['time'], freq='M')
                    ET = ET / 30

                    Effective_rain = xr.open_dataarray(Rainfall_path, decode_times=False)
                    reference_date = ref_date.strftime("%Y-%m-%d")  # the start date of the simulation results from CWatM
                    Effective_rain['time'] = pd.date_range(start=reference_date, periods=Effective_rain.sizes['time'],
                                                           freq='M')
                    count += 1
                    # ---------------------------------available land for ag module------------------------------------
                    irr_file = basepath + r"/modules/hydro/hydro_inputs_external/landuses/frac_irrigated.nc"
                    irr_nc = xr.open_dataarray(irr_file)

                    paddy_file = basepath + r"/modules/hydro/hydro_inputs_external/landuses/frac_paddy.nc"
                    paddy_nc = xr.open_dataarray(paddy_file)

                    land_nc = irr_nc + paddy_nc

                    Irr = land_nc.isel(time=0)
                    Irr.name = 'Pix_AvailableFrac_Irr'
                    merger = xr.merge([adminsegs, Irr, cellarea], join='override')
                    merger_df = merger.drop('time').to_dataframe().reset_index()
                    merger_df['AvailableArea_1000ha_pixel'] = \
                        merger_df.Pix_AvailableFrac_Irr * merger_df.cellArea_totalend * 0.0001 / 1000
                    ava_land = merger_df.groupby('Agent').sum().reset_index()[['Agent', 'AvailableArea_1000ha_pixel']]
                    ava_land.rename(columns={'AvailableArea_1000ha_pixel': 'Land_avai'}, inplace=True)

                    # -------------------------------------groundwater depth-------------------------------------------
                    file_combine_GW = Groundwater.mean(dim='time')
                    file_combine_GW.name = 'groundwater_depth'
                    merger = xr.merge([adminsegs, file_combine_GW], join='override')
                    merger_df = merger.to_dataframe().reset_index()
                    gw_file[t_in] = merger_df.groupby(['Agent']).mean().reset_index()

                    # ------------------------------surface water (reservoir + river)----------------------------------
                    file_combine_SW = Reservoirwater[0]  # first day of reservoir water
                    Inflow_m = Inflow.sum(dim='time')
                    Outflow_m = Outflow.sum(dim='time')
                    Leakage_m = Leakage.sum(dim='time')
                    Evap_m = Evap.sum(dim='time')

                    if t_in == 0:
                        (
                            res_initial_out_season[t_in], release_out_season[t_in], sw_urban_season, sw_ag_season) = \
                            sw_interaction(t0_year, pp, file_combine_SW, Inflow_m, Outflow_m, Leakage_m, Evap_m, {},
                                           ref_date.month)
                    else:
                        (
                            res_initial_out_season[t_in], release_out_season[t_in], sw_urban_season, sw_ag_season) = \
                            sw_interaction(t0_year, pp, file_combine_SW, Inflow_m, Outflow_m, Leakage_m, Evap_m,
                                           res_initial_out_season[t_in - 1], ref_date.month)

                    sw_file[t_in] = sw_ag_season.rename(columns={'AgentID': 'Agent'})

                    year_month_idx = pd.MultiIndex.from_arrays([Riverwater['time.year'], Riverwater['time.month']])
                    Riverwater.coords['year_month'] = ('time', year_month_idx)

                    file_combine_RIVER = Riverwater[0]  # first day of river water
                    file_combine_RIVER.name = 'river'
                    merger_river = xr.merge([adminsegs, file_combine_RIVER, cellarea], join='override')
                    merger_river_df = merger_river.to_dataframe().reset_index()
                    merger_river_df[
                        'river_volume'] = merger_river_df.river * merger_river_df.cellArea_totalend * 1E-9 * 0.2
                    river_file[t_in] = merger_river_df.groupby(['Agent']).sum().reset_index()

                    sw_river_combine[t_in] = sw_file[t_in].set_index('Agent').add(river_file[t_in].set_index('Agent'),
                                                                                  fill_value=0).reset_index()
                    sw_river_combine[t_in]['sw_river'] = \
                        sw_river_combine[t_in]['SW_avai_ag_km3'] + sw_river_combine[t_in]['river_volume']

                    # -----------------------------------------------------ET-------------------------------------------
                    ET_planning = ET
                    ET_planning.name = 'm' + str(t_in + 6)
                    merger = xr.merge([adminsegs, ET_planning], join='override')
                    merger_df = merger.to_dataframe().reset_index()
                    ET_file[t_in] = merger_df.groupby(['Agent']).mean().reset_index()

                    # ----------------------------------------------Effective rainfall----------------------------------
                    Effrain_planning = Effective_rain
                    Effrain_planning.name = 'm' + str(t_in + 6)
                    merger = xr.merge([adminsegs, Effrain_planning], join='override')
                    merger_df = merger.to_dataframe().reset_index()
                    Effrain_file[t_in] = merger_df.groupby(['Agent']).mean().reset_index()

                    # -----------------------------------grs ir water requirement---------------------------------------
                    rainfall = Effrain_file[t_in].rename(columns={'m' + str(t_in + 6): 'crop_wat_app'})[
                        ['Agent', 'crop_wat_app']]
                    ref_evap = ET_file[t_in].rename(columns={'m' + str(t_in + 6): 'crop_wat_app'})[
                        ['Agent', 'crop_wat_app']]
                    crop_coeff = ag_input_df['crop_coeff']

                    full_irr = {}
                    for crop in range(len(crop_coeff['Crop'])):
                        full_irr[crop_coeff['Crop'][crop]] = np.maximum(0,
                                                                        (((ref_evap.mul(crop_coeff.iloc[crop][t_in + 1],
                                                                                        axis=1)
                                                                           .mul(month_days.iloc[0][t_in],
                                                                                axis=1)) - rainfall)) / 0.6 / 100)
                        full_irr['Rice1'] = np.maximum(0, (((ref_evap.mul(crop_coeff.iloc[0][t_in + 1], axis=1)
                                                             .mul(month_days.iloc[0][t_in],
                                                                  axis=1)) - rainfall)) / 0.6 / 100)
                        full_irr['Rice2'] = np.maximum(0, (((ref_evap.mul(crop_coeff.iloc[1][t_in + 1], axis=1)
                                                             .mul(month_days.iloc[0][t_in],
                                                                  axis=1)) - rainfall)) / 0.6 / 100)
                        # unit conversion 1km3/1000hac = 100meter

                    final_combine = pd.concat([df for df in full_irr.values()], ignore_index=True, axis=0).reset_index()
                    final_combine = final_combine.fillna(0)

                    grs_file[t_in] = final_combine[['crop_wat_app']]

                # ---------------------------------------AG TO HYDRO ---------------------------------------------------
                # replace the land_avai (input) with ava land (excel)
                ag_input_df['land_avai'] = ava_land

                # replace the gw_avai (input) with ag_gw_avai (excel)
                ag_gw_write = ag_input_df['gw_depth'].set_index('Agent')
                gw_avai_ag = pd.concat([df.set_index('Agent')['groundwater_depth'] for df in gw_file.values()],
                                       ignore_index=False, axis=1)
                ag_gw_write.loc[:, ag_gw_write.columns[index]] = gw_avai_ag.mean(axis=1)
                ag_input_df['gw_depth'] = ag_gw_write.reset_index().fillna(0)

                # replace the sw_avai (input) with ag_sw_avai (excel)
                ag_sw_write = ag_input_df['sw_avai'].set_index('Agent')
                sw_avai_ag = pd.concat([df.set_index('Agent')['sw_river'] for df in sw_river_combine.values()],
                                       ignore_index=False, axis=1)
                ag_sw_write.loc[:, ag_sw_write.columns[index]] = sw_avai_ag.sum(axis=1)
                ag_input_df['sw_avai'] = ag_sw_write.reset_index().fillna(0)

                # replace the eff_rainfall (input) with effrainfall (excel)
                ag_effrain_write = ag_input_df['eff_rainfall'].set_index('Agent')
                effrain_ag = pd.concat([df.set_index('Agent').iloc[:, -1] for df in Effrain_file.values()],
                                       ignore_index=False, axis=1)
                ag_effrain_write.loc[:, ag_effrain_write.columns[season_range]] = effrain_ag
                ag_input_df['eff_rainfall'] = ag_effrain_write.reset_index().fillna(0)

                # replace the ref_evapotransp (input) with ET (excel)
                ag_evap_write = ag_input_df['ref_evapotransp'].set_index('Agent')
                ET_ag = pd.concat([df.set_index('Agent').iloc[:, -1] for df in ET_file.values()],
                                  ignore_index=False, axis=1)
                ag_evap_write.loc[:, ag_evap_write.columns[season_range]] = ET_ag
                ag_input_df['ref_evapotransp'] = ag_evap_write.reset_index().fillna(0)

                # replace the grs ir water requirement (input) with gir file (excel)
                grs_ag = pd.concat([df['crop_wat_app'] for df in grs_file.values()],
                                   ignore_index=False, axis=1).fillna(0)
                ag_input_df['crop_wat_app'].loc[:, ag_input_df['crop_wat_app'].columns[index + 2]] = grs_ag.sum(
                    axis=1) * 1000000

                # adjust the full irrigation water requirement by deficit coefficient
                deficit_efficiency = pd.read_csv(basepath + r'/modules/ag_seasonal/ag_inputs/deficit_efficiency.csv')
                water_deficit_ratio = ag_input_df['crop_wat_app'].merge(deficit_efficiency, how='left',
                                                                        on=['Crop', 'Agent'])
                water_deficit_ratio = water_deficit_ratio.fillna(0)

                if index == 0:
                    water_deficit_ratio['khy1'] = water_deficit_ratio['khy1'] * (1 - water_deficit_ratio['khy1_ratio'])
                else:
                    water_deficit_ratio['rby1'] = water_deficit_ratio['rby1'] * (1 - water_deficit_ratio['rby1_ratio'])

                ag_input_df['crop_wat_app'].iloc[:, 2:] = water_deficit_ratio[['khy1', 'rby1', 'khy2', 'rby2']]

                # change the data type of agent from float64 to int64
                ag_input_df['land_avai'] = ag_input_df['land_avai'].astype({'Agent': 'int64'})
                ag_input_df['gw_depth'] = ag_input_df['gw_depth'].astype({'Agent': 'int64'})
                ag_input_df['sw_avai'] = ag_input_df['sw_avai'].astype({'Agent': 'int64'})
                ag_input_df['eff_rainfall'] = ag_input_df['eff_rainfall'].astype({'Agent': 'int64'})
                ag_input_df['ref_evapotransp'] = ag_input_df['ref_evapotransp'].astype({'Agent': 'int64'})


                with pd.ExcelWriter(filepath_sim + 
                                '/ag_implementation_input_' + str(time_counter) + "_" + str(count_imple) + '.xlsx') as writer:
                    for sheetname in ag_input_df.keys():
                        ag_input_df[sheetname].to_excel(writer, index=False, sheet_name=sheetname)

                if time_counter == 4:
                    share.ag_input_kharif = ag_input_df

                # ----------------------------------------------------------
                #
                #      _   ___ ___  ___
                #     /_\ / __| _ \/ _ \
                #    / _ | (_ |   | (_) |
                #   /_/ \_\___|_|_\\___/
                #
                # ----------------------------------------------------------
                ag_opt = FarmSeasonalDecisionMaker('ag land test')

                # Create each agents inputs
                full_agent_list = ag_input_df['sw_avai'].Agent.unique()  # change agents list to your agent codes
                agent_list = full_agent_list

                # Create each agents inputs
                input_df = ag_opt.create_seasonal_agent_input_subset(agent_list, ag_input_df)

                # Modify the Inputs as Necessary
                modified_agent_inputs = {}

                print('Start ag loop')
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
                agent_mod_obj_dict = ag_opt.ag_seasonal_optimization_loop(agent_list, modified_agent_inputs, interventionID_tanker, scenarioID,solar_input)

                # Extract the results # see the ag_looping_functions for description
                ag_opt_output_in = ag_opt.extract_seasonal_ag_outputs(agent_list, agent_mod_obj_dict, filepath_sim +
                                                                      '/ag_implementation_' + str(time_counter) +
                                                                      "_" + str(count_imple) + '.xlsx', toExcel=True)
                share.ag_output_implementation = ag_opt_output_in

                # Ag output for LAND
                crop_agent_land_df_in = ag_opt_output_in['Land']

                # Ag output for surface water and groundwater
                season_crop_agent_surfacewater_df = ag_opt_output_in['SW']
                season_crop_agent_groundwater_df = ag_opt_output_in['GW']
                season_crop_agent_totalwater_df = ag_opt_output_in['Water']

                # Correct the data types
                season_crop_agent_surfacewater_df = season_crop_agent_surfacewater_df.astype(
                    {'Month': 'string', 'Crop': 'string', 'Agent': 'float64'})
                season_crop_agent_groundwater_df = season_crop_agent_groundwater_df.astype(
                    {'Month': 'string', 'Crop': 'string', 'Agent': 'float64'})
                season_crop_agent_totalwater_df = season_crop_agent_totalwater_df.astype(
                    {'Month': 'string', 'Crop': 'string', 'Agent': 'float64'})

                ##########################################################################################
                # AG Seasonal output dis aggregate to monthly output
                # start with total water use (in proportion with the crop water requirement)
                # agent within CCA uses SW (and GW), if sw_use <= sw_avai, only sw (sw=total water use)
                # if sw_use > sw_avai, use sw and gw
                # gw use = total water use - surface water use
                ##########################################################################################
                # total
                # read the monthly crop water requirement from the planning module
                if index == 0:
                    water_percent = share.ag_water_percent_planning
                else:
                    water_percent = share.ag_water_percent_implementation

                # replace the crop water requirement with that from current season (perfect information)
                for t_p in season_range:
                    water_percent.loc[:, Months[t_p]] = grs_ag.iloc[:, np.where(season_range == t_p)[0][0]]

                water_percent['khy1'] = 0
                water_percent['rby1'] = 0
                water_percent['khy2'] = 0
                water_percent['rby2'] = 0
                for i in range(6, 11):
                    water_percent['khy1'] += water_percent.loc[:, "m" + str(i)] * 1000000  # 1km3/1000ha = 1000000m3/ha
                for i in range(11, 18):
                    water_percent['rby1'] += water_percent.loc[:, "m" + str(i)] * 1000000
                for i in range(18, 23):
                    water_percent['khy2'] += water_percent.loc[:, "m" + str(i)] * 1000000
                for i in [23, 24, 1, 2, 3, 4, 5]:
                    water_percent['rby2'] += water_percent.loc[:, "m" + str(i)] * 1000000

                share.ag_water_percent_implementation = water_percent

                # read the monthly crop water requirement from the planning module
                if index == 0:
                    sw_river_combine_planning = share.ag_sw_planning
                else:
                    sw_river_combine_planning = share.ag_sw_implementation

                # replace the crop water requirement with that from current season (perfect information)
                sw_avai_ag = sw_avai_ag.reset_index().iloc[:, 1:]
                for t_p in season_range:
                    # for j in np.arange(water_percent.shape[0]):
                    sw_river_combine_planning.loc[:, Months[t_p]] = sw_avai_ag.iloc[:,
                                                                    np.where(season_range == t_p)[0][0]]

                share.ag_sw_implementation = sw_river_combine_planning

                Z1 = AgOutput_season_to_mo()
                mo_agent_totalwater_df = Z1.totalwater_season_to_mo(water_percent, season_crop_agent_totalwater_df)
                mo_agent_surfacewater_df = Z1.surfacewater_season_to_mo(sw_river_combine_planning,
                                                                        mo_agent_totalwater_df,
                                                                        season_crop_agent_surfacewater_df)
                mo_agent_groundwater_df = Z1.groundwater_season_to_mo(mo_agent_totalwater_df, mo_agent_surfacewater_df)

                # -----------------------------------------------------------------
                #     _______          __  _______ __  __
                #    / ____\ \        / /\|__   __|  \/  |
                #   | |     \ \  /\  / /  \  | |  | \  / |
                #   | |      \ \/  \/ / /\ \ | |  | |\/| |
                #   | |____   \  /\  / ____ \| |  | |  | |
                #    \_____|   \/  \/_/    \_\_|  |_|  |_|
                # -----------------------------------------------------------------
                for time_counter_in in season_range:

                    year_in = currentDate_in.year
                    month_in = currentDate_in.month
                    day_in = currentDate_in.day

                    # set number of Months
                    numMonths_in = 1

                    if time_counter_in == 0:
                        priorDate_in = currentDate_in
                    else:
                        priorDate_in = currentDate_in - timedelta(days=1)
                    nextDate_in = add_months(currentDate_in, numMonths_in)

                    # Use the last month's information
                    if time_counter_in == 0:
                        priorDate_month_in = currentDate_in
                    else:
                        priorDate_month_in = add_months(currentDate_in, -numMonths_in)

                    print('time_counter_inside_second_loop: ' + str(time_counter_in))
                    print('year_inside_second_loop: ' + str(year_in))
                    print('month_inside_second_loop: ' + str(month_in))
                    print('day_inside_second_loop: ' + str(day_in))

                    # ----------------------------------------- CWATM SETTINGS  ---------------------------------------
                    cwatm_test_in = HydroCWATM('cwatm land test')

                    PathOut_folder_in = filepath_final
                    PathRoot = basepath + r'/modules/hydro'

                    PathOut_in = PathOut_folder_in + "_t" + str(time_counter_in + 1)
                    PathOut_prior_in = PathOut_folder_in + "_t" + str(time_counter_in)

                    if not os.path.isdir(PathOut_in):
                        os.makedirs(PathOut_in)
                        print("created hydro-out folder : ", PathOut_in)
                    else:
                        print(PathOut_in, "hydro-out folder already exists.")

                    # File inputs
                    excel_settings_file_in = cwatm_settings
                    settings_template_in = basepath + r'/modules/hydro/hydro_inputs/settings' \
                                                      r'/settings_CWatM_template_Agent.ini'
                    new_settings_file_in = filepath_sim + '/implementation_in_CWatM_new.ini'

                    StepStart_in = currentDate_in.strftime('%d/%m/%Y')
                    StepEnd_in = nextDate_in.strftime('%d/%m/%Y')  # the start of the next month
                    StepInit_in = (nextDate_in - timedelta(days=1)).strftime('%d/%m/%Y')

                    if update_years == 'historical':
                        urbanfrac_file = urbanfrac_filepath + r'/UBB_built-up_res_' + str(year_in) + '.nc' # str(year) + '.nc'
                    else:
                        urbanfrac_file = urbanfrac_filepath + r'/UBB_built-up_res_' + str(2050) + '.nc' # str(year) + '.nc'

                    if time_counter_in == 0:
                        initLoadDate_in = priorDate_in.strftime('%Y%m%d')  # the end of the prior month
                        src_file_in = filepath_sim + '/init_planning_' + initLoadDate_in + '.nc'
                        dest_file_in = filepath_sim + '/init_implementation_in_' + initLoadDate_in + ".nc"
                        shutil.copy2(src_file_in, dest_file_in)

                    # init date
                    else:
                        initLoadDate_in = priorDate_in.strftime('%Y%m%d')  # StepEnd

                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # + Getting Agents + Land Available from Hydro
                    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # available land from cwatm lat,long, avail land for segments and for pixels [changes with each run]
                    if time_counter_in == 0:  # initial conditions
                        pix_available_frac_Irr_file_in = xr.open_dataarray(
                            basepath + r'/modules/hydro/hydro_inputs_external/initial_condition/fallowIrr_Time0_daily.nc')

                    else:
                        pix_available_frac_Irr_file_in = xr.open_dataarray(PathOut_prior_in + r'/fallowIrr_daily.nc')
                        reference_date_in = priorDate_month_in.strftime("%Y-%m-%d")
                        pix_available_frac_Irr_file_in['time'] = pd.date_range(start=reference_date_in,
                                                                               periods=
                                                                               pix_available_frac_Irr_file_in.sizes[
                                                                                   'time'], freq='D')
                    pix_available_frac_Irr_file_in.name = 'Pix_AvailableFrac_Irr'


                    if time_counter_in == 0:
                        sel_date_in = priorDate_in.strftime('%Y-%m-%d')
                        pix_available_frac_Irr_in = pix_available_frac_Irr_file_in[0]
                    else:
                        sel_date_in = currentDate_in.strftime('%Y-%m-%d')
                        pix_available_frac_Irr_in = pix_available_frac_Irr_file_in.loc[sel_date_in]

                    # print(sel_date_in)

                    # merge and create a data frame
                    merger_in = xr.merge([adminsegs, pix_available_frac_Irr_in, cellarea], join='override')
                    merger_in = merger_in.drop('time')
                    merger_df_in = merger_in.to_dataframe()

                    # calculate the actual available area equipped with irrigation by pixel
                    merger_df_in[
                        'Pix_AvailableArea_m2_Irr'] = merger_df_in.Pix_AvailableFrac_Irr * merger_df_in.cellArea_totalend
                    merger_df_in['AvailableArea_1000ha_Irr'] = merger_df_in['Pix_AvailableArea_m2_Irr'] * 0.0001 / 1000
                    seg_available_area_Irr_in = merger_df_in.groupby('Agent').sum().reset_index()[
                        ['Agent', 'AvailableArea_1000ha_Irr']]

                    main_df_in = merger_df_in.reset_index()
                    main_df_in.rename(columns={'AvailableArea_1000ha_Irr': 'AvailableArea_1000ha_pixel'}, inplace=True)

                    # -------------------------------------------------------------------------------
                    #                                                                  _
                    #      _   ___ ___  ___        _____  ___          _   _ _   _  __| |_ -- ---
                    #     /_\ / __| _ \/ _ \      |_   _|/ _ \        | |_| | | | |/ _` | '__/ _ \
                    #    / _ | (_ |   | (_) |       | | | (_) |       |  _  | |_| | (_| | | | (_) |
                    #   /_/ \_\___|_|_\\___/        |_|  \___/        |_| |_|\__, |\__,_|_|  \___/
                    #                                                        |___/
                    # --------------------------------------------------------------------------------

                    # ---------------------------------------------------LAND---------------------------
                    # Correct the datatypes
                    crop_agent_land_df_in = crop_agent_land_df_in.astype(
                        {'Crop': 'string', 'Agent': 'float64'})

                    # Cropland
                    cropland_df_in = crop_agent_land_df_in[['Crop', 'Agent', 'Land_1000ha']].groupby(
                        ['Agent', 'Crop']).sum().reset_index()  # check length = number of crops x number agents

                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # + Creating a netcdf of land area with calculation
                    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    outpath_ag_in = filepath_sim + '/ag_to_hydro_in'
                    if not os.path.exists(outpath_ag_in):
                        os.makedirs(outpath_ag_in)

                    # To use this, the dataframe must have a lat and lon component
                    selected_croplist_agro_in = croplist_agro  # selected all the crops

                    for cropname in selected_croplist_agro_in:
                        # print('Processing Crop: ' + cropname)

                        # --------------------------------- CALCULATION ----------------------------------------------
                        crop_df = cropland_df_in[cropland_df_in['Crop'] == cropname]
                        full_df = main_df_in.merge(crop_df, on='Agent')
                        full_df = full_df.merge(seg_available_area_Irr_in, on='Agent')

                        full_df['frac'] = (full_df['Land_1000ha'] / full_df['AvailableArea_1000ha_Irr']) * (
                            full_df['Pix_AvailableFrac_Irr'])  
                        cropfrac = full_df[['lat', 'lon', 'Crop', 'frac']]
                        # --------------------------------------------------------------------------------------------

                        df_in = cropfrac

                        if ('lat' in df_in.columns) and ('lon' in df_in.columns):
                            array_in = create_array(df_in, lat_lon, 'frac')
                            outfile = outpath_ag_in + "/" + 'implementation_in_' + str(time_counter_in) + \
                                      "_" + cropname + "_" + str(count_imple) + '.nc'

                            create_netcdf_from_array(nc_template_file, outfile, array_in, cropname + "_area")
                        else:
                            print('lat or lon is not a column name -- check your input dataframe')
                        # print(cropname + " netcdf file created")

                    # ---------------------------------------------WATER--------------------------------------------------
                    # --------------------------------------------GROUNDWATER---------------------------------------------
                    print('generating gw use file for' + Months[time_counter_in])
                    var_gw = mo_agent_groundwater_df[['Agent', Months[time_counter_in] + '_gw_use']]
                    merger_gw = adminsegs_df.merge(var_gw, on='Agent')
                    merger_gw['gw_m3'] = merger_gw[Months[time_counter_in] + '_gw_use'] * 1e+9 * 0.6 / irrigation_efficiency_nonpaddy 
                    # change the unit from km3 to m3

                    df_in_gw = merger_gw

                    if ('lat' in df_in_gw.columns) and ('lon' in df_in_gw.columns):
                        array_in_gw = create_array(df_in_gw, lat_lon, 'gw_m3')
                        outfile_gw = outpath_ag_in + r'/implementation_in_gw_' + Months[time_counter_in] + "_" + str(count_imple) + '.nc'
                        create_netcdf_from_array(nc_template_file, outfile_gw, array_in_gw, 'gw_m3')
                    else:
                        print('lat or lon is not a column name -- check your input dataframe')
                    # print("netcdf file created")

                    # --------------------------------------------SURFACE WATER-----------------------------------------
                    print('generating sw use file for' + Months[time_counter_in])
                    var_sw = mo_agent_surfacewater_df[['Agent', Months[time_counter_in] + '_sw_use']]
                    merger_sw = adminsegs_df.merge(var_sw, on='Agent')
                    merger_sw['sw_m3'] = merger_sw[Months[time_counter_in] + '_sw_use'] * 1e+9 * 0.6 / irrigation_efficiency_nonpaddy 
                    # change the unit from km3 to m3

                    df_in_sw = merger_sw

                    if ('lat' in df_in_sw.columns) and ('lon' in df_in_sw.columns):
                        array_in_sw = create_array(df_in_sw, lat_lon, 'sw_m3')
                        outfile_sw = outpath_ag_in + r'/implementation_in_sw_' + Months[time_counter_in] + "_" + str(count_imple) + '.nc'
                        create_netcdf_from_array(nc_template_file, outfile_sw, array_in_sw, 'sw_m3')

                    else:
                        print('lat or lon is not a column name -- check your input dataframe')
                    # print("netcdf file created")

                    # --------------------------------
                    #    _   _           _
                    #   | | | |_   _  __| |_ __ ___
                    #   | |_| | | | |/ _` | '__/ _ \
                    #   |  _  | |_| | (_| | | | (_) |
                    #   |_| |_|\__, |\__,_|_|  \___/
                    #          |___/
                    # --------------------------------

                    # ----------------------------------------- CWATM SETTINGS DICTIONARY -----------------------------
                    print('Creating Settings for CWATM')
                    # Create settings dictionary
                    settings_dict_in = cwatm_test_in.create_cwatm_settings_dict_obj(excel_settings_file_in)

                    # Time Settings
                    settings_dict_in['StepStart'] = StepStart_in  # 6
                    settings_dict_in['StepEnd'] = StepEnd_in  # 7
                    settings_dict_in[
                        'initLoad'] = filepath_sim + '/init_implementation_in_' +\
                                      initLoadDate_in + '.nc'  # 5
                    settings_dict_in['initSave'] = filepath_sim + '/init_implementation_in'
                    settings_dict_in['StepInit'] = StepInit_in

                    # paths
                    settings_dict_in['PathRoot'] = PathRoot
                    settings_dict_in['PathOut'] = PathOut_in
                    settings_dict_in['Excel_settings_file'] = excel_settings_file_in

                    # LAND USES/ LAND COVERS
                    settings_dict_in['leftoverIrrigatedCropIsRainfed'] = 0
                    settings_dict_in['irrPaddy_efficiency'] = irrigation_efficiency_paddy
                    settings_dict_in['irrNonPaddy_efficiency'] = irrigation_efficiency_nonpaddy

                    # urban_module
                    settings_dict_in['sealed_fracVegCover'] = urbanfrac_file

                    settings_dict_in['domestic_agent_GW_request_month_m3'] = outpath_urban + "/" + \
                                                                             'urban_CWatM_implementation_gw_t' + str(
                        time_counter_in) + "_" + str(count_imple) + '.nc'

                    settings_dict_in['domestic_agent_SW_request_month_m3'] = outpath_urban + "/" + \
                                                                             'urban_CWatM_implementation_sw_t' + str(
                        time_counter_in) + "_" + str(count_imple) + '.nc'
                        
                             # Climate
                    settings_dict_in['PathMeteo'] = PathMeteo
                
                    settings_dict_in['WindMaps'] = WindMaps 
                    settings_dict_in['PrecipitationMaps'] = PreciMaps
                    settings_dict_in['TavgMaps'] = TavgMaps 
                    settings_dict_in['TminMaps'] = TminMaps
                    settings_dict_in['TmaxMaps'] = TmaxMaps
                    settings_dict_in['PSurfMaps'] = PSurfMaps
                    settings_dict_in['RSDSMaps'] = RSDSMaps
                    settings_dict_in['RSDLMaps'] = RSDLMaps
                    settings_dict_in['QAirMaps'] = QAirMaps

                    settings_dict_in['reservoir_command_areas'] = reservoir_cca_file

                    settings_dict_in['water_table_limit_for_pumping'] = water_table_limit_for_pumping


                    if time_counter_in == 0:
                        if year == initial_year+interval:
                            settings_dict_in[
                                'init_water_table'] = basepath + r'/modules/hydro/hydro_inputs/Modflow/modflow_inputs' \
                                                                 r'/modflow_watertable_totalend.nc'
                        else:
                            settings_dict_in[
                                'init_water_table'] = filepath_sim + '/final_y' + \
                                                      str(count_imple) + '_t12' + '/modflow_watertable_totalend.nc'
                    else:
                        settings_dict_in[
                            'init_water_table'] = filepath_final + '_t' + str(
                            time_counter_in) + r'/modflow_watertable_totalend.nc'

                        # to pass ag to cwatm variable name
                    sum_irr_data = np.empty([320, 370])

                    croplist_hydro.sort()
                    croplist_agro.sort()
                    special_crop = ['Rice1', 'Rice2']
                    agro_list_in = [ele for ele in croplist_agro if ele not in special_crop]
                    cropname_dict_in = dict(zip(croplist_hydro, agro_list_in))

                    for cropname in croplist_hydro:  # hydro
                        zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                        settings_dict_in[cropname + "_Irr"] = zeros_array
                        # settings_dict_in[cropname + "_nonIrr"] = zeros_array
                        settings_dict_in['irrPaddy_fracVegCover'] = zeros_array

                    if time_counter_in == 0:  # June
                        for cropname_kharif in Kharif_crop_list:  # hydro
                            if 'implementation_in_' + str(time_counter_in) + "_" + cropname_dict[
                                cropname_kharif] + "_" + \
                                    str(count_imple) + ".nc" in os.listdir(outpath_ag_in):
                                print(cropname_kharif)

                                nc_path = outpath_ag_in + '/' + 'implementation_in_' + str(time_counter_in) + "_" + \
                                          cropname_dict[cropname_kharif] + "_" + str(count_imple) + '.nc'

                                settings_dict_in[cropname_kharif + "_Irr"] = nc_path
                                # settings_dict_in[cropname_kharif + "_nonIrr"] = nc_path

                                nc = xr.open_dataset(nc_path)
                                varname = list(nc.data_vars)
                                data = nc.variables[varname[0]].data
                                sum_irr_data = sum_irr_data + data
                            else:
                                # need a zero's netcdf
                                zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                                settings_dict_in[cropname_kharif + "_Irr"] = zeros_array
                                # settings_dict_in[cropname_kharif + "_nonIrr"] = zeros_array

                    if time_counter_in in range(6):
                        settings_dict_in['irrPaddy_fracVegCover'] = outpath_ag_in + "/" + 'implementation_in_' + str(
                            time_counter_in) + "_" + "Rice1" + "_" + str(count_imple) + ".nc"

                    if time_counter_in == 5:  # November
                        for cropname_rabi in Rabi_crop_list:  # hydro
                            if 'implementation_in_' + str(time_counter_in) + "_" + cropname_dict[cropname_rabi] + "_" + \
                                    str(count_imple) + ".nc" in os.listdir(outpath_ag_in):
                                print(cropname_rabi)

                                nc_path = outpath_ag_in + '/' + 'implementation_in_' + str(time_counter_in) + "_" + \
                                          cropname_dict[cropname_rabi] + "_" + str(count_imple) + '.nc'

                                settings_dict_in[cropname_rabi + "_Irr"] = nc_path
                                # settings_dict_in[cropname_rabi + "_nonIrr"] = nc_path

                                nc = xr.open_dataset(nc_path)
                                varname = list(nc.data_vars)
                                data = nc.variables[varname[0]].data
                                sum_irr_data = sum_irr_data + data
                            else:
                                # need a zero's netcdf
                                zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                                settings_dict_in[cropname_rabi + "_Irr"] = zeros_array
                                # settings_dict_in[cropname_rabi + "_nonIrr"] = zeros_array

                    if time_counter_in == 7:  # Jan
                        for cropname_jan in Jan_crop_list:  # hydro
                            if 'implementation_in_' + str(time_counter_in) + "_" + cropname_dict[cropname_jan] + "_" + \
                                    str(count_imple) + ".nc" in os.listdir(outpath_ag_in):
                                print(cropname_jan)

                                nc_path = outpath_ag_in + '/' + 'implementation_in_' + str(time_counter_in) + "_" + \
                                          cropname_dict[cropname_jan] + "_" + str(count_imple) + '.nc'

                                settings_dict_in[cropname_jan + "_Irr"] = nc_path
                                # settings_dict_in[cropname_jan + "_nonIrr"] = nc_path

                                nc = xr.open_dataset(nc_path)
                                varname = list(nc.data_vars)
                                data = nc.variables[varname[0]].data
                                sum_irr_data = sum_irr_data + data
                            else:
                                # need a zero's netcdf
                                zeros_array = basepath + r'/modules/hydro/hydro_files/netcdfs' + '/' + 'zeros_array.nc'

                                settings_dict_in[cropname_jan + "_Irr"] = zeros_array
                                # settings_dict_in[cropname_jan + "_nonIrr"] = zeros_array

                    settings_dict_in[
                        'irrigation_agent_SW_request_month_m3'] = outpath_ag_in + r'/implementation_in_sw_' + Months[
                        time_counter_in] + "_" + str(count_imple) + '.nc'
                    settings_dict_in[
                        'irrigation_agent_GW_request_month_m3'] = outpath_ag_in + r'/implementation_in_gw_' + Months[
                        time_counter_in] + "_" + str(count_imple) + '.nc'

                    # OUTPUTS Settings

                    # select outputs --------------------------------------------------------------------------------
                    settings_dict_in['save_initial'] = 1

                    settings_dict_in['OUT_Map_MonthEnd'] = open(basepath + r"/OUT_MAP_MonthEnd.txt", "r").readline()
                    settings_dict_in['OUT_MAP_Daily'] = open(basepath + r"/OUT_MAP_Daily.txt", "r").readline()
                    settings_dict_in['OUT_Map_MonthTot'] = open(basepath + r"/OUT_MAP_MonthTot.txt", "r").readline()
                    settings_dict_in['OUT_Map_MonthAvg'] = open(basepath + r"/OUT_MAP_MonthAvg.txt", "r").readline()
                    settings_dict_in['OUT_MAP_AnnualTot'] = ""
                    settings_dict_in['OUT_MAP_TotalEnd'] = open(basepath + r"/OUT_MAP_TotalEnd.txt", "r").readline()

                    # ----------------------------------------- RUNNING CWATM  --------------------------------------
                    # create the new settings file
                    cwatm_test_in.define_cwatm_settings(settings_template_in, new_settings_file_in, settings_dict_in)

                    # show the settings file
                    cwatm_test_in.save_cwatm_settings_file(new_settings_file_in, settings_dict_in['PathOut'],
                                                           '_settings')

                    # run cwatm
                    cwatm_test_in.run_cwatm(new_settings_file_in)

                    currentDate_in = nextDate_in
                    print(currentDate_in)

                    # close the netcdf file every time when open it
                    for cropname in croplist_hydro:
                        file = outpath_ag_in + '/' + 'implementation_in_' + str(time_counter_in) + "_" + \
                               cropname_dict[cropname] + "_" + str(count_imple) + '.nc'
                        open_Chickpea = xr.open_dataset(file)
                        open_Chickpea.close()

            else:
                pass

            currentDate = nextDate
            print(currentDate)





