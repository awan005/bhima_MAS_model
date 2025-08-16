__author__ = 'ankunwang'

from basepath_file import basepath
import xarray as xr
import numpy as np
import pandas as pd
from helper_objects.helper_functions import data_selection, add_months
from datetime import date


def first_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, Variable_name, path, period,interval):
    PathOut = basepath + r'/modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2'
    SWvariable = xr.open_dataarray(PathOut + "/" + Variable_name + "_" + "daily.nc")

    if t0_year > initial_year+interval:
        SWvariable_test = {}
        SWvariable_new = xr.open_dataarray(path + '/final_y1_t1' +
                                           "/" + Variable_name + '_daily.nc')
        num = 0
        years_run = t0_year - initial_year - interval
        for j in np.arange(years_run):
            Path_final = path + '/final_y' + str(j+1)
            for i in np.arange(12):
                basepath_SW = Path_final + "_t" + str(i + 1) + "/" + Variable_name + "_" + "daily.nc"
                SWvariable_test[num] = xr.open_dataarray(basepath_SW)
                num += 1
        for n in np.arange(1, num):
            SWvariable_new = SWvariable_test[n].combine_first(SWvariable_new)

        if t0_year - initial_year < (period+interval):
            SWvariable_selected_his_yr1 = data_selection(SWvariable, t0_year_prior-interval, t0_month, t0_day,
                                                        initial_year, t0_month, t0_day)
            SWvariable_selected_his_yr2 = data_selection(SWvariable, t0_year_prior-interval + 1, t0_month, t0_day,
                                                        initial_year, t0_month, t0_day)
            if t0_year - initial_year == (1+interval):
                SWvariable_selected_new_yr1 = None
            else:
                SWvariable_selected_new_yr1 = data_selection(SWvariable_new, initial_year+interval, t0_month, t0_day,
                                                                 t0_year - 1, t0_month, t0_day)
            SWvariable_selected_new_yr2 = data_selection(SWvariable_new, initial_year+interval, t0_month, t0_day,
                                                             t0_year, t0_month, t0_day)
            if t0_year - initial_year == (1+interval):
                SWvariable_selected_yr1 = SWvariable_selected_his_yr1
            else:
                SWvariable_selected_yr1 = xr.concat([SWvariable_selected_his_yr1, SWvariable_selected_new_yr1],
                                                        dim='time')
            SWvariable_selected_yr2 = xr.concat([SWvariable_selected_his_yr2, SWvariable_selected_new_yr2],
                                                    dim='time')
        else:
            SWvariable_selected_yr1 = data_selection(SWvariable_new, t0_year_prior, t0_month, t0_day,
                                                         t0_year - 1, t0_month, t0_day)
            SWvariable_selected_yr2 = data_selection(SWvariable_new, t0_year_prior + 1, t0_month, t0_day,
                                                         t0_year, t0_month, t0_day)
    else:
        SWvariable_selected_yr1 = data_selection(SWvariable, t0_year_prior-interval, t0_month, t0_day,
                                                     t0_year-interval-1, t0_month, t0_day)
        SWvariable_selected_yr2 = data_selection(SWvariable, t0_year_prior-interval+1, t0_month, t0_day,
                                                     t0_year-interval, t0_month, t0_day)

    SWvariable_selected_yr1.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
                [SWvariable_selected_yr1['time.year'], SWvariable_selected_yr1['time.month']]))

    SWvariable_selected_yr2.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
                [SWvariable_selected_yr2['time.year'], SWvariable_selected_yr2['time.month']]))

    SWvariable_year_month_yr1 = SWvariable_selected_yr1.groupby('year_month').first() # storage on the first day
    SWvariable_year_month_yr2 = SWvariable_selected_yr2.groupby('year_month').first()

    SWvariable_month_yr1 = SWvariable_year_month_yr1.groupby('year_month_level_1').mean()
    SWvariable_month_yr2 = SWvariable_year_month_yr2.groupby('year_month_level_1').mean()

    SWvariable_month_yr12 = xr.concat([SWvariable_month_yr1, SWvariable_month_yr2], dim='year_month_level_1')

    return SWvariable_month_yr12


def sum_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, Variable_name, path, period,interval):
    PathOut = basepath + r'/modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2'
    SWvariable = xr.open_dataarray(PathOut + "/" + Variable_name + "_" + "daily.nc")

    if t0_year > initial_year+interval:
        SWvariable_test = {}
        SWvariable_new = xr.open_dataarray(path + '/final_y1_t1' +
                                           "/" + Variable_name + '_daily.nc')
        num = 0
        years_run = t0_year - initial_year - interval
        for j in np.arange(years_run):
            Path_final = path + '/final_y' + str(j+1)
            for i in np.arange(12):
                basepath_SW = Path_final + "_t" + str(i + 1) + "/" + Variable_name + "_" + "daily.nc"
                SWvariable_test[num] = xr.open_dataarray(basepath_SW)
                num += 1
        for n in np.arange(1, num):
            SWvariable_new = SWvariable_test[n].combine_first(SWvariable_new)

        if t0_year - initial_year < (period+interval):
            SWvariable_selected_his_yr1 = data_selection(SWvariable, t0_year_prior-interval, t0_month, t0_day,
                                                        initial_year, t0_month, t0_day)
            SWvariable_selected_his_yr2 = data_selection(SWvariable, t0_year_prior-interval + 1, t0_month, t0_day,
                                                        initial_year, t0_month, t0_day)
            if t0_year - initial_year == (1+interval):
                SWvariable_selected_new_yr1 = None
            else:
                SWvariable_selected_new_yr1 = data_selection(SWvariable_new, initial_year+interval, t0_month, t0_day,
                                                                 t0_year - 1, t0_month, t0_day)
            SWvariable_selected_new_yr2 = data_selection(SWvariable_new, initial_year+interval, t0_month, t0_day,
                                                             t0_year, t0_month, t0_day)
            if t0_year - initial_year == (1+interval):
                SWvariable_selected_yr1 = SWvariable_selected_his_yr1
            else:
                SWvariable_selected_yr1 = xr.concat([SWvariable_selected_his_yr1, SWvariable_selected_new_yr1],
                                                        dim='time')
            SWvariable_selected_yr2 = xr.concat([SWvariable_selected_his_yr2, SWvariable_selected_new_yr2],
                                                    dim='time')
        else:
            SWvariable_selected_yr1 = data_selection(SWvariable_new, t0_year_prior, t0_month, t0_day,
                                                         t0_year - 1, t0_month, t0_day)
            SWvariable_selected_yr2 = data_selection(SWvariable_new, t0_year_prior + 1, t0_month, t0_day,
                                                         t0_year, t0_month, t0_day)
    else:
        SWvariable_selected_yr1 = data_selection(SWvariable, t0_year_prior-interval, t0_month, t0_day,
                                                     t0_year-interval-1, t0_month, t0_day)
        SWvariable_selected_yr2 = data_selection(SWvariable, t0_year_prior-interval+1, t0_month, t0_day,
                                                     t0_year-interval, t0_month, t0_day)

    SWvariable_selected_yr1.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
                [SWvariable_selected_yr1['time.year'], SWvariable_selected_yr1['time.month']]))

    SWvariable_selected_yr2.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
                [SWvariable_selected_yr2['time.year'], SWvariable_selected_yr2['time.month']]))

    SWvariable_year_month_yr1 = SWvariable_selected_yr1.groupby('year_month').sum() # storage on the first day
    SWvariable_year_month_yr2 = SWvariable_selected_yr2.groupby('year_month').sum()

    SWvariable_month_yr1 = SWvariable_year_month_yr1.groupby('year_month_level_1').mean()
    SWvariable_month_yr2 = SWvariable_year_month_yr2.groupby('year_month_level_1').mean()

    SWvariable_month_yr12 = xr.concat([SWvariable_month_yr1, SWvariable_month_yr2], dim='year_month_level_1')

    return SWvariable_month_yr12


def average_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, Variable_name, path, period,interval):
    PathOut = basepath + r'/modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2'
    SWvariable = xr.open_dataarray(PathOut + "/" + Variable_name + "_" + "daily.nc")

    if t0_year > initial_year+interval:
        SWvariable_test = {}
        SWvariable_new = xr.open_dataarray(path + '/final_y1_t1' +
                                           "/" + Variable_name + '_daily.nc')
        num = 0
        years_run = t0_year - initial_year - interval
        for j in np.arange(years_run):
            Path_final = path + '/final_y' + str(j+1)
            for i in np.arange(12):
                basepath_SW = Path_final + "_t" + str(i + 1) + "/" + Variable_name + "_" + "daily.nc"
                SWvariable_test[num] = xr.open_dataarray(basepath_SW)
                num += 1
        for n in np.arange(1, num):
            SWvariable_new = SWvariable_test[n].combine_first(SWvariable_new)

        if t0_year - initial_year < (period+interval):
            SWvariable_selected_his_yr1 = data_selection(SWvariable, t0_year_prior-interval, t0_month, t0_day,
                                                        initial_year, t0_month, t0_day)
            SWvariable_selected_his_yr2 = data_selection(SWvariable, t0_year_prior-interval + 1, t0_month, t0_day,
                                                        initial_year, t0_month, t0_day)
            if t0_year - initial_year == (1+interval):
                SWvariable_selected_new_yr1 = None
            else:
                SWvariable_selected_new_yr1 = data_selection(SWvariable_new, initial_year+interval, t0_month, t0_day,
                                                                 t0_year - 1, t0_month, t0_day)
            SWvariable_selected_new_yr2 = data_selection(SWvariable_new, initial_year+interval, t0_month, t0_day,
                                                             t0_year, t0_month, t0_day)
            if t0_year - initial_year == (1+interval):
                SWvariable_selected_yr1 = SWvariable_selected_his_yr1
            else:
                SWvariable_selected_yr1 = xr.concat([SWvariable_selected_his_yr1, SWvariable_selected_new_yr1],
                                                        dim='time')
            SWvariable_selected_yr2 = xr.concat([SWvariable_selected_his_yr2, SWvariable_selected_new_yr2],
                                                    dim='time')
        else:
            SWvariable_selected_yr1 = data_selection(SWvariable_new, t0_year_prior, t0_month, t0_day,
                                                         t0_year- 1, t0_month, t0_day)
            SWvariable_selected_yr2 = data_selection(SWvariable_new, t0_year_prior + 1, t0_month, t0_day,
                                                         t0_year, t0_month, t0_day)
    else:
        SWvariable_selected_yr1 = data_selection(SWvariable, t0_year_prior-interval, t0_month, t0_day,
                                                     t0_year-interval -1, t0_month, t0_day)
        SWvariable_selected_yr2 = data_selection(SWvariable, t0_year_prior-interval+1, t0_month, t0_day,
                                                     t0_year-interval, t0_month, t0_day)

    SWvariable_selected_yr1.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
                [SWvariable_selected_yr1['time.year'], SWvariable_selected_yr1['time.month']]))

    SWvariable_selected_yr2.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
                [SWvariable_selected_yr2['time.year'], SWvariable_selected_yr2['time.month']]))

    SWvariable_year_month_yr1 = SWvariable_selected_yr1.groupby('year_month').mean() # storage on the first day
    SWvariable_year_month_yr2 = SWvariable_selected_yr2.groupby('year_month').mean()

    SWvariable_month_yr1 = SWvariable_year_month_yr1.groupby('year_month_level_1').mean()
    SWvariable_month_yr2 = SWvariable_year_month_yr2.groupby('year_month_level_1').mean()

    SWvariable_month_yr12 = xr.concat([SWvariable_month_yr1, SWvariable_month_yr2], dim='year_month_level_1')

    return SWvariable_month_yr12


def average_monthly_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, path, period,interval):
    PathOut = basepath + r'/modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2'

    ET_path = PathOut + r'/ETRefAverage_segments_monthavg.nc'

    ET = xr.open_dataarray(ET_path, decode_times=False)
    reference_date = '1995-06-30'  # the start date of the simulation results from CWatM
    ET['time'] = pd.date_range(start=reference_date, periods=ET.sizes['time'], freq='M')

    numMonths = 1
    currentDate = date(t0_year, t0_month, t0_day)
    nextDate = add_months(currentDate, numMonths)
    last_day = (nextDate - currentDate).days

    if t0_year > initial_year+interval:

        ET_test = {}
        ET_new = xr.open_dataarray(path + '/final_y1_t1' + "/" +
                                   "ETRefAverage_segments_monthtot.nc", decode_times=False)
        reference_date = date(initial_year+interval, t0_month, last_day).strftime("%Y-%m-%d")
        ET_new['time'] = pd.date_range(start=reference_date, periods=ET_new.sizes['time'], freq='M')
        ET_new = ET_new / 30
        num = 0
        years_run = t0_year - initial_year - interval
        for j in np.arange(years_run):
            Path_final = path + '/final_y' + str(j + 1)
            for i in np.arange(12):
                basepath_ET = Path_final + "_t" + str(i + 1) + "/" + "ETRefAverage_segments_monthtot.nc"
                ET_test[num] = xr.open_dataarray(basepath_ET, decode_times=False)
                reference_date = add_months(date(initial_year+interval, t0_month, last_day), num).strftime("%Y-%m-%d")
                ET_test[num]['time'] = pd.date_range(start=reference_date, periods=ET_test[num].sizes['time'], freq='M')
                ET_test[num] = ET_test[num] / 30
                num += 1
        for n in np.arange(1, num):
            ET_new = ET_test[n].combine_first(ET_new)

        if t0_year - initial_year < (period+interval):
            ET_selected_his_yr1 = data_selection(ET, t0_year_prior-interval, t0_month, last_day,
                                                 initial_year, t0_month - 1, last_day + 1)
            ET_selected_his_yr2 = data_selection(ET, t0_year_prior -interval+ 1, t0_month, last_day,
                                                 initial_year, t0_month - 1, last_day + 1)
            if t0_year - initial_year == (1+interval):
                ET_selected_new_yr1 = None
            else:
                ET_selected_new_yr1 = data_selection(ET_new, initial_year+interval, t0_month, last_day,
                                                     t0_year - 1, t0_month - 1, last_day + 1)
            ET_selected_new_yr2 = data_selection(ET_new, initial_year+interval, t0_month, last_day,
                                                 t0_year, t0_month - 1, last_day + 1)
            if t0_year - initial_year == (1+interval):
                ET_selected_yr1 = ET_selected_his_yr1
            else:
                ET_selected_yr1 = xr.concat([ET_selected_his_yr1, ET_selected_new_yr1],
                                            dim='time')
            ET_selected_yr2 = xr.concat([ET_selected_his_yr2, ET_selected_new_yr2],
                                        dim='time')
        else:
            ET_selected_yr1 = data_selection(ET_new, t0_year_prior, t0_month, last_day,
                                             t0_year - 1, t0_month - 1, last_day + 1)
            ET_selected_yr2 = data_selection(ET_new, t0_year_prior + 1, t0_month, last_day,
                                             t0_year, t0_month - 1, last_day + 1)
    else:
        ET_selected_yr1 = data_selection(ET, t0_year_prior-interval, t0_month, last_day,
                                         t0_year-interval - 1, t0_month - 1, last_day + 1)
        ET_selected_yr2 = data_selection(ET, t0_year_prior-interval + 1, t0_month, last_day,
                                         t0_year-interval, t0_month - 1, last_day + 1)

    ET_selected_yr1.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
        [ET_selected_yr1['time.year'], ET_selected_yr1['time.month']]))

    ET_selected_yr2.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
        [ET_selected_yr2['time.year'], ET_selected_yr2['time.month']]))

    ET_year_month_yr1 = ET_selected_yr1.groupby('year_month').mean()
    ET_year_month_yr2 = ET_selected_yr2.groupby('year_month').mean()

    ET_month_yr1 = ET_year_month_yr1.groupby('year_month_level_1').mean()
    ET_month_yr2 = ET_year_month_yr2.groupby('year_month_level_1').mean()

    ET_month_yr12 = xr.concat([ET_month_yr1, ET_month_yr2], dim='year_month_level_1')

    return ET_month_yr12


def sum_monthly_historical_climate(t0_year, t0_year_prior, t0_month, t0_day, initial_year, path, period,interval):
    PathOut = basepath + r'/modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2'

    Rainfall_path = PathOut + r'/precipEffectiveAverage_segments_monthtot.nc'

    Effective_rain = xr.open_dataarray(Rainfall_path, decode_times=False)
    reference_date = '1995-06-30'  # the start date of the simulation results from CWatM
    Effective_rain['time'] = pd.date_range(start=reference_date, periods=Effective_rain.sizes['time'], freq='M')

    numMonths = 1
    currentDate = date(t0_year, t0_month, t0_day)
    nextDate = add_months(currentDate, numMonths)
    last_day = (nextDate - currentDate).days

    if t0_year > initial_year+interval:

        Effrain_test = {}
        Effective_rain_new = xr.open_dataarray(path + '/final_y1_t1' +
                                               "/" + "precipEffectiveAverage_segments_monthtot.nc", decode_times=False)
        reference_date = date(initial_year+interval, t0_month, last_day).strftime("%Y-%m-%d")
        Effective_rain_new['time'] = pd.date_range(start=reference_date, periods=Effective_rain_new.sizes['time'], freq='M')

        num = 0
        years_run = t0_year - initial_year - interval
        for j in np.arange(years_run):
            Path_final = path + '/final_y' + str(j + 1)
            for i in np.arange(12):
                basepath_Effrain = Path_final + "_t" + str(i + 1) + "/" + "precipEffectiveAverage_segments_monthtot.nc"
                Effrain_test[num] = xr.open_dataarray(basepath_Effrain, decode_times=False)
                reference_date = add_months(date(initial_year+interval, t0_month, last_day), num).strftime("%Y-%m-%d")
                Effrain_test[num]['time'] = pd.date_range(start=reference_date, periods=Effrain_test[num].sizes['time'], freq='M')
                num += 1
        for n in np.arange(1, num):
            Effective_rain_new = Effrain_test[n].combine_first(Effective_rain_new)

        if t0_year - initial_year < (period+interval):
            Effrain_selected_his_yr1 = data_selection(Effective_rain, t0_year_prior-interval, t0_month, last_day,
                                                      initial_year, t0_month-1, last_day+1)
            Effrain_selected_his_yr2 = data_selection(Effective_rain, t0_year_prior-interval + 1, t0_month, last_day,
                                                      initial_year, t0_month-1, last_day+1)
            if t0_year - initial_year == (1+interval):
                Effrain_selected_new_yr1 = None
            else:
                Effrain_selected_new_yr1 = data_selection(Effective_rain_new, initial_year+interval, t0_month, last_day,
                                                     t0_year - 1, t0_month-1, last_day+1)
            Effrain_selected_new_yr2 = data_selection(Effective_rain_new, initial_year+interval, t0_month, last_day,
                                                 t0_year, t0_month-1, last_day+1)
            if t0_year - initial_year == (1+interval):
                Effrain_selected_yr1 = Effrain_selected_his_yr1
            else:
                Effrain_selected_yr1 = xr.concat([Effrain_selected_his_yr1, Effrain_selected_new_yr1],
                                            dim='time')
            Effrain_selected_yr2 = xr.concat([Effrain_selected_his_yr2, Effrain_selected_new_yr2],
                                        dim='time')
        else:
            Effrain_selected_yr1 = data_selection(Effective_rain_new, t0_year_prior, t0_month, last_day,
                                                  t0_year- 1, t0_month-1, last_day+1)
            Effrain_selected_yr2 = data_selection(Effective_rain_new, t0_year_prior + 1, t0_month, last_day,
                                                  t0_year, t0_month-1, last_day+1)
    else:
        Effrain_selected_yr1 = data_selection(Effective_rain, t0_year_prior-interval, t0_month, last_day,
                                              t0_year -interval - 1, t0_month-1, last_day+1)
        Effrain_selected_yr2 = data_selection(Effective_rain, t0_year_prior -interval+ 1, t0_month, last_day,
                                              t0_year-interval, t0_month-1, last_day+1)

    Effrain_selected_yr1.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
        [Effrain_selected_yr1['time.year'], Effrain_selected_yr1['time.month']]))

    Effrain_selected_yr2.coords['year_month'] = ('time', pd.MultiIndex.from_arrays(
        [Effrain_selected_yr2['time.year'], Effrain_selected_yr2['time.month']]))

    Effrain_year_month_yr1 = Effrain_selected_yr1.groupby('year_month').sum()
    Effrain_year_month_yr2 = Effrain_selected_yr2.groupby('year_month').sum()

    Effrain_month_yr1 = Effrain_year_month_yr1.groupby('year_month_level_1').mean()
    Effrain_month_yr2 = Effrain_year_month_yr2.groupby('year_month_level_1').mean()

    Effrain_month_yr12 = xr.concat([Effrain_month_yr1, Effrain_month_yr2], dim='year_month_level_1')

    return Effrain_month_yr12