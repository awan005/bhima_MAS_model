__author__ = 'ankun wang'

from basepath_file import basepath
from datetime import date
import pandas as pd
import xarray as xr
import geopandas as gpd

adminsegs = xr.open_dataarray(
    basepath + r'/modules/hydro/hydro_inputs/landsurface/waterDemand/2020-10-14_Agents.nc')
adminsegs.name = 'Agent'
adminsegs_df = adminsegs.to_dataframe().reset_index()
cellarea = xr.open_dataarray(basepath + r'/modules/hydro/hydro_files/netcdfs/cellArea.nc')

# adminsegs for urban_module
adminsegs_df_urban = pd.read_csv(basepath + r'/modules/hydro/hydro_inputs/landsurface/waterDemand/'
                                            r'adminsegs_Pune_Solapur.csv')

Pune_shp = gpd.read_file(
    basepath + r'/modules/hydro/hydro_files/shapefiles/PMC_Boundary/PMC_PCMC_CleanBoundary.shp')
            
# SW and GW water use
lat_lon = pd.read_excel(basepath + r'/modules/hydro/hydro_files/lat_lon.xlsx', sheet_name='lat_lon')
nc_template_file = basepath + r'/modules/hydro/hydro_inputs_external/nc_template.nc'

# read the ag module original input file
ag_input_file = basepath + r'/modules/ag_seasonal/ag_inputs/' \
                           r'2020-11-06 Ag Input Data Bhima.xlsx'
ag_input_df = pd.read_excel(ag_input_file, sheet_name=None)
Months = ['m6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm16', 'm17', 'm18', 'm19', 'm20',
          'm21', 'm22', 'm23', 'm24', 'm1', 'm2', 'm3', 'm4', 'm5']
month_days = pd.DataFrame(
    [{'m6': 30, 'm7': 31, 'm8': 31, 'm9': 30,
      'm10': 31, 'm11': 30, 'm12': 31, 'm13': 31, 'm14': 28, 'm15': 31, 'm16': 30, 'm17': 31, 'm18': 30,
      'm19': 31, 'm20': 31, 'm21': 30, 'm22': 31, 'm23': 30, 'm24': 31,
      'm1': 31, 'm2': 28, 'm3': 31, 'm4': 30, 'm5': 31}])
Agent_months = ['Agent'] + Months

croplist_hydro = ['SugarAdsali', 'SugarPreseasonal', 'SugarSuru', 'Wheat', 'FruitVegK', 'FruitVegR','SpicesK',
                  'SpicesR', 'Cotton', 'Soybean', 'Chickpea', 'Groundnut', 'SorghumK', 'SorghumR', 'MaizeK',
                  'MaizeR', 'GeneralK', 'GeneralR', 'Solar']

Kharif_crop_list = ['SugarAdsali', 'FruitVegK', 'SpicesK', 'Cotton', 'Soybean', 'Groundnut', 'SorghumK',
                    'MaizeK', 'GeneralK']
Rabi_crop_list = ['SugarPreseasonal', 'Wheat', 'FruitVegR', 'SpicesR', 'Chickpea', 'SorghumR', 'MaizeR',
                  'GeneralR']
Jan_crop_list = ['SugarSuru', 'Solar']

croplist_agro = ['Chickpea1', 'Cotton1', 'FruitVegK1', 'FruitVegR1', 'GeneralK1', 'GeneralR1', 'Groundnut1',
                 'MaizeK1', 'MaizeR1', 'Rice1', 'Solar', 'SorghumK1', 'SorghumR1', 'Soybean1', 'SpicesK1',
                 'SpicesR1', 'SugarAdsali', 'SugarPreseasonal', 'SugarSuru2', 'Wheat1']


