__author__ = 'ankun wang'

import xarray as xr
import pandas as pd
from basepath_file import basepath


# monthly sum of the daily sw and gw values in the format of netcdf files
def dailync_monthlysum_ak(fileIn, yearSelected, monthSelected):
    data = xr.open_dataarray(fileIn)
    monthlysum = data[data.indexes["time"].year == yearSelected].groupby('time.month').sum("time")
    select_month = monthlysum.sel(month=monthSelected)
    data_month_sel = select_month.drop('month')
    return data_month_sel


# monthly average of the daily sw and gw values in the format of netcdf files
def dailync_monthlymean_ak(fileIn, yearSelected, monthSelected):
    data = xr.open_dataarray(fileIn)
    monthlysum = data[data.indexes["time"].year == yearSelected].groupby('time.month').mean("time")
    select_month = monthlysum.sel(month=monthSelected)
    data_month_sel = select_month.drop('month')
    return data_month_sel


# first day within a month of the daily sw and gw values in the format of netcdf files
def dailync_monthlyfirst_ak(fileIn, yearSelected, monthSelected):
    data = xr.open_dataarray(fileIn)
    monthlysum = data[data.indexes["time"].year == yearSelected].groupby('time.month').first()
    select_month = monthlysum.sel(month=monthSelected)
    data_month_sel = select_month.drop('month')
    return data_month_sel

