__author__ = 'ankunwang'

from geopandas import gpd
from basepath_file import basepath
import xarray as xr
import numpy as np
from shapely.geometry import Point
from global_var import cellarea


def urban_df_to_cwatm(urban_module_output, urbanfrac_file, year, pipe_loss):

    urban_output_df = urban_module_output
    urban_output_df_no_duplicate = urban_output_df.groupby(['x', 'y']).sum().reset_index()
    nonIrrDem = urban_output_df_no_duplicate[
        ['x', 'y', 'non_piped_m3/mo', 'piped_m3/mo', 'well_m3/mo', 'tanker_m3/mo', 'total_water_m3/mo']] \
        .reset_index().rename(columns={'x': 'lon', 'y': 'lat'})

    # cellarea - pixels
    cellarea_df = cellarea.to_dataframe().reset_index()
    cellarea_df_urban = cellarea_df.round({'lon': 4, 'lat': 4})  # # different number of decimal place don't merge
    merger_urban_mid = nonIrrDem.merge(cellarea_df_urban, on=['lat', 'lon'])

    # consumption over withdrawal ratio
    # Urban water consumption = urban_module water withdrawal * (1 - (urban_module fraction – recycling ratio))
    recycling_ratio = 0.3
    urban_frac = xr.open_dataarray(urbanfrac_file)
    urban_frac.name = str(year)
    urbanfrac_recycle = urban_frac - recycling_ratio
    urbanfrac_recycle_masked = urbanfrac_recycle.where(urbanfrac_recycle > 0, 0)
    consumption_over_withdrawal = (1 - urbanfrac_recycle_masked).to_dataframe().reset_index().round(
        {'lon': 4, 'lat': 4})
    consumption_over_withdrawal = consumption_over_withdrawal.rename(columns={str(year): 'consump_over_withdrwl'})
    merger_urban = merger_urban_mid.merge(consumption_over_withdrawal, on=['lat', 'lon'])

    # change monthly urban_module sectoral water use from m3/month to m/day
    merger_urban['gw_consumption'] = merger_urban['non_piped_m3/mo'] * merger_urban['consump_over_withdrwl']

    merger_urban['piped_m3/month_with_loss'] = merger_urban['piped_m3/mo'] / (1 - pipe_loss)
    merger_urban['piped_consumption'] = merger_urban['piped_m3/month_with_loss'] * merger_urban['consump_over_withdrwl']

    # Tanker water demand sourced from outside the city

    tanker_shp = gpd.read_file(basepath + r"/modules/hydro/hydro_files/shapefiles/tanker/tanker.shp")
    merger_urban['geometry'] = merger_urban.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    merger_urban_gpd = gpd.GeoDataFrame(merger_urban, crs=tanker_shp.crs, geometry="geometry")
    tanker_Pune = gpd.sjoin(merger_urban_gpd, tanker_shp[['geometry']], op="within")

    tanker_0 = np.sum(tanker_Pune[tanker_Pune['index_right'] == 0])['tanker_m3/mo']
    tanker_1 = np.sum(tanker_Pune[tanker_Pune['index_right'] == 1])['tanker_m3/mo']
    tanker_2 = np.sum(tanker_Pune[tanker_Pune['index_right'] == 2])['tanker_m3/mo']
    tanker_6 = np.sum(tanker_Pune[tanker_Pune['index_right'] == 6])['tanker_m3/mo']

    tanker_Pune.loc[tanker_Pune.index_right == 3, 'tanker_m3/mo_new'] = tanker_Pune.loc[tanker_Pune.index_right == 3,
                                                                                        'tanker_m3/mo'] + tanker_0 / \
                                                                        tanker_Pune[
                                                                            tanker_Pune['index_right'] == 3].shape[0]
    tanker_Pune.loc[tanker_Pune.index_right == 4, 'tanker_m3/mo_new'] = tanker_Pune.loc[tanker_Pune.index_right == 4,
                                                                                        'tanker_m3/mo'] + tanker_1 / \
                                                                        tanker_Pune[
                                                                            tanker_Pune['index_right'] == 4].shape[0]
    tanker_Pune.loc[tanker_Pune.index_right == 7, 'tanker_m3/mo_new'] = tanker_Pune.loc[tanker_Pune.index_right == 7,
                                                                                        'tanker_m3/mo'] + tanker_2 / \
                                                                        tanker_Pune[
                                                                            tanker_Pune['index_right'] == 7].shape[0]
    tanker_Pune.loc[tanker_Pune.index_right == 5, 'tanker_m3/mo_new'] = tanker_Pune.loc[tanker_Pune.index_right == 5,
                                                                                        'tanker_m3/mo'] + tanker_6 / \
                                                                        tanker_Pune[
                                                                            tanker_Pune['index_right'] == 5].shape[0]

    tanker_Pune.loc[tanker_Pune.index_right == 0, 'tanker_m3/mo_new'] = 0
    tanker_Pune.loc[tanker_Pune.index_right == 1, 'tanker_m3/mo_new'] = 0
    tanker_Pune.loc[tanker_Pune.index_right == 2, 'tanker_m3/mo_new'] = 0
    tanker_Pune.loc[tanker_Pune.index_right == 6, 'tanker_m3/mo_new'] = 0

    merger_urban.loc[tanker_Pune.index, 'tanker_m3/mo'] = tanker_Pune['tanker_m3/mo_new']

    merger_urban['non_piped_m3/mo_new'] = merger_urban['well_m3/mo'] + merger_urban['tanker_m3/mo']

    return merger_urban
