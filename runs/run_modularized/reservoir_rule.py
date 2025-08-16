__author__ = 'ankunwang'

from helper_objects.helper_functions import geo_idx
from basepath_file import basepath
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from global_var import Pune_shp, cellarea
import numpy as np

waterbody = pd.read_csv(basepath + r'/modules/hydro/hydro_files/excel/Copy of Waterbodies.csv')
waterbody_water = waterbody[waterbody['lat'] > 0]

Res = {}
Res_downstream = {}
for i in range(waterbody_water.shape[0]):
    Res[waterbody_water['Lake_name'][waterbody_water.index[i]]] = (
        waterbody_water['lat'][waterbody_water.index[i]], waterbody_water['long'][waterbody_water.index[i]])

waterbody_downstream = waterbody_water[waterbody_water['CCA_code']>0]
waterbody_downstream = waterbody_downstream[waterbody_downstream['Downstream'] != 1]

for i in range(waterbody_downstream.shape[0]):
    Res_downstream[waterbody_downstream['Lake_name'][waterbody_downstream.index[i]]] = (
        waterbody_downstream['lat'][waterbody_downstream.index[i]],waterbody_downstream['long'][waterbody_downstream.index[i]])

reservoir_rule_file = basepath + r'/modules/hydro/hydro_files/excel/reservoir_wet_dry_rule.xlsx'
rule = pd.read_excel(reservoir_rule_file, sheet_name=None)
rule['percentage'] = rule['percentage'].fillna(0)

# wet season, release is a function of storage(previous month) and expected inflow in this month
itcp = {}
co_inflow = {}
co_storage = {}
for i in range(5):
    itcp[i+6] = {}
    co_inflow[i+6] = {}
    co_storage[i+6] = {}
    for keys in Res:
        itcp[i+6][keys] = rule['intercept'][rule['intercept']['Unnamed: 0'] == keys][i+6].values.item()
        co_inflow[i+6][keys] = rule['co_inflow'][rule['co_inflow']['Unnamed: 0'] == keys][i+6].values.item()
        co_storage[i+6][keys] = rule['co_storage'][rule['co_storage']['Unnamed: 0'] == keys][i+6].values.item()

for i in range(waterbody_water.shape[0]):
    Res[waterbody_water['Lake_name'][waterbody_water.index[i]]] = (
        waterbody_water['lat'][waterbody_water.index[i]], waterbody_water['long'][waterbody_water.index[i]])

# dry season, release is a function of october 31st storage (fixed percentage)
perc = {}
ls = [11, 12, 1, 2, 3, 4, 5]
for i in range(7):
    perc[ls[i]] = {}
    for keys in Res:
        perc[ls[i]][keys] = rule['percentage'][rule['percentage']['Unnamed: 0'] == keys][ls[i]].values.item()


PathOut = basepath + r'/modules/hydro/hydro_outputs/HydroTest_2021-11-01_v2'
# files -- daily sw, gw, river; monthly effective rainfall, ET
Reservoirwater = xr.open_dataarray(PathOut + "/lakeResStorage_daily.nc")
lats = Reservoirwater[0]['lat'][:]
lons = Reservoirwater[0]['lon'][:]

lat_idx = {}
lon_idx = {}

for keys in Res:
    outlet = Res[keys]

    in_lat = outlet[0]
    in_lon = outlet[1]

    lat_idx[keys] = geo_idx(in_lat, lats)
    lon_idx[keys] = geo_idx(in_lon, lons)


def sw_interaction(t0_year, population, reservoirstorage_va, inflow_va,
                   outflow_va, leakage_va, evap_va, res_ini_intermediate, m):
    res = {}
    inflow = {}
    for keys in Res:
        res[keys] = reservoirstorage_va[lat_idx[keys], lon_idx[keys]].values.item()
        inflow[keys] = inflow_va[lat_idx[keys], lon_idx[keys]].values.item() * cellarea[lat_idx[keys], lon_idx[keys]].values.item()

    Outflow_res = {}
    for keys in Res:
        if keys in Res_downstream:
            Outflow_res[keys] = outflow_va[lat_idx[keys], lon_idx[keys]].values.item() * cellarea[
                lat_idx[keys], lon_idx[keys]].values.item()
        else:
            Outflow_res[keys] = 0

    Leakage_res = {}
    Evaporation_res = {}
    for keys in Res:
        Leakage_res[keys] = leakage_va[lat_idx[keys], lon_idx[keys]].values.item()
        Evaporation_res[keys] = evap_va[lat_idx[keys], lon_idx[keys]].values.item() * cellarea[
            lat_idx[keys], lon_idx[keys]].values.item()

    ## dry season
    release = {}
    res_ini = res_ini_intermediate.copy()
    for keys in Res:
        if m == 11:
            release[keys] = res[keys] * perc[m][keys]
            res_ini[keys] = res[keys]

            release['Khadakwasla'] = (res['Khadakwasla'] + res['Panshet'] + res['Warasgaon'] + res['Temghar']) * \
                                     perc[m]['Khadakwasla']
            release['Veer'] = (res['Gunjwani'] + res['Niradevghar'] + res['Bhatghar'] + res['Veer']) * perc[m]['Veer']
            release['Yedgaon'] = (res['Yedgaon'] + res['Manikdoh']) * perc[m]['Yedgaon']
            release['Chaskaman'] = (res['Chaskaman'] + res['Kalmodi']) * perc[m]['Chaskaman']

        elif m in [12, 1, 2, 3, 4, 5]:
            if perc[list(perc)[list(perc).index(m)]][keys] == perc[list(perc)[list(perc).index(m)-1]][keys]:
                res[keys] = res_ini[keys]
            else:
                res_ini[keys] = res[keys]
            release[keys] = res[keys] * perc[m][keys]

            release['Khadakwasla'] = (res['Khadakwasla'] + res['Panshet'] + res['Warasgaon'] + res['Temghar']) * \
                                     perc[m]['Khadakwasla']
            release['Veer'] = (res['Gunjwani'] + res['Niradevghar'] + res['Bhatghar'] + res['Veer']) * perc[m]['Veer']
            release['Yedgaon'] = (res['Yedgaon'] + res['Manikdoh']) * perc[m]['Yedgaon']
            release['Chaskaman'] = (res['Chaskaman'] + res['Kalmodi']) * perc[m]['Chaskaman']
        else:
            release[keys] = np.exp(res[keys] * co_storage[m][keys] + inflow[keys] * co_inflow[m][keys] + itcp[m][keys])
            release[keys] -= Outflow_res[keys]

        release[keys] -= Leakage_res[keys]
        release[keys] -= Evaporation_res[keys]

    release_df = pd.DataFrame.from_dict(release, orient='index', columns=[m])

    # Surface water availability
    reservoirstorage = reservoirstorage_va.to_dataframe().reset_index()
    inflow_df = inflow_va.to_dataframe().reset_index()

    restorage_df = reservoirstorage.rename(columns={'lakeResStorage': 'reservoirStorage'})
    inflow_restorage = inflow_df.merge(restorage_df, on=['lat', 'lon'])

    restorage_df_261wbs = inflow_restorage[inflow_restorage['reservoirStorage'] != 0].round({'lon': 4, 'lat': 4})
    # 261 reservoirs has water

    res_data_in_file = basepath + r'/modules/hydro/hydro_files/excel/Copy of Waterbodies.csv'
    res_water_use_data = pd.read_csv(res_data_in_file).rename(columns={'long': 'lon'}).round({'lon': 4, 'lat': 4})
    res_cca = res_water_use_data[res_water_use_data['CCA_code'] != 0]
    # res_CCA has 32 reservoirs with connection to CCA, also has lat lon info

    # Big reservoirs that are connected to a CCA （CCA-reservoir storage）
    res_merge = res_cca.merge(restorage_df_261wbs, on=['lat', 'lon'])  # this is to connect res storage with its ID
    res_merge = res_merge[
        ['CCA_name', 'CCA_code', 'reservoirStorage', 'lakeResInflowM', 'lat', 'lon', 'Lake_name']].rename(
        columns={'CCA_code': 'CCA code', 'reservoirStorage': 'BigreservoirStorage'})

    res_merge_release = res_merge.set_index('Lake_name').join(release_df).reset_index()

    # --------------------------
    # for urban_module module
    # --------------------------

    # get the agent list 174 agents
    delineation_data_in_file = basepath + r'/modules/hydro/hydro_files/excel/2020-11-18 FUSE_Codes.xlsx'
    delineation_water_use_data = pd.read_excel(delineation_data_in_file, '174_AgAgents')
    agent_list174 = delineation_water_use_data.sort_values('Agent Code').rename(
        columns={'Agent Code': 'Agent'})[['Agent']].reset_index()

    # extend to 177 agents including PMC, PCMC, Solapur
    agent_list177 = agent_list174.append({'Agent': 1}, ignore_index=True)
    agent_list177 = agent_list177.append({'Agent': 2}, ignore_index=True)
    agent_list177 = agent_list177.append({'Agent': 3}, ignore_index=True)
    agent_list177 = agent_list177.sort_values(['Agent']).reset_index()[['Agent']]

    # Connect reservoir storage file (CCA-reservoir storage) with Res_CCA_Agent_Pixel
    # create a dataframe containing agent, corresponding cca code, CCA name, Reservoir ID and reservoir name
    res_cca_agent_pixel = pd.read_csv(basepath + r'/modules/hydro/hydro_files/excel/Res_CCA_Agent_Pixel_updated_urban.csv')

    # connect res, cca, agent, pixel lat lon, with reservoir storage
    restorage_connect = pd.merge(res_cca_agent_pixel, res_merge_release[['Lake_name', 'CCA code', m]], on=['CCA code'])
    # restorage_connect.loc[restorage_connect[m] < 0, m] = 0
    restorage_connect = restorage_connect[restorage_connect[m] > 0]

    res_pp = restorage_connect.merge(population, on=['lat', 'lon'])
    
    if res_pp.shape[0] > 0: 
        print('res has water')

        # print(res_pp.head())
        res_pp['geometry'] = res_pp.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
        res_pp_gpd = gpd.GeoDataFrame(res_pp, crs=Pune_shp.crs, geometry="geometry")
        Pune_res_pp = gpd.sjoin(res_pp_gpd, Pune_shp[['geometry']], op="within")
        res_pp_gpd.loc[~res_pp_gpd.index.isin(Pune_res_pp.index), 'lpcd'] = 100
        res_pp_gpd.loc[Pune_res_pp.index, 'lpcd'] = 250
        
        ratio_ind_total=0.25
        res_pp_gpd['res supply'] = res_pp_gpd['lpcd'] * res_pp_gpd[t0_year] * 30.44 * 1e-3 / (1-ratio_ind_total)
    
        test = res_pp_gpd[['lat', 'lon', 'Agent', m, 'res supply']].groupby('Agent').agg({'lat': 'mean', 'lon': 'mean',
                                                                                          'res supply': 'sum', m: 'mean'})
        test.loc[test['res supply'] > test[m], 'res supply'] = test[m]
    
        urban_sw_avai_agent177 = agent_list177.join(test, on='Agent').reset_index().rename(
            columns={'Agent': 'AgentID', 'res supply': 'sw_m3_normal'}).fillna(0)
    
        # --------------------------
        # for ag module
        # --------------------------
    
        # agent_cca_lat_lon_file
        res_cca = pd.read_csv(basepath + r'/modules/hydro/hydro_files/excel/Res_CCA_Agent_Pixel_updated_ag.csv')
        res_cca = res_cca.fillna(0)
    
        res_pp_gpd['SW_avai_ag'] = res_pp_gpd[m] - res_pp_gpd['res supply']
        res_pp_gpd.loc[res_pp_gpd['SW_avai_ag'] < 0, 'SW_avai_ag'] = 0
    
        test_agent_ag = res_pp_gpd[['lat', 'lon', m, 'res supply', 'SW_avai_ag']].merge(res_cca, on=['lat', 'lon'])
        ag_sw_avai = test_agent_ag[['Agent', 'SW_avai_ag']].groupby('Agent').mean()  # unit m3 only 70 agents
    
        ag_sw_avai_agent174 = agent_list174.join(ag_sw_avai, on='Agent').reset_index()
    
        frac_irr_nc = xr.open_dataarray(
            basepath + r"/modules/hydro/hydro_inputs_external/landuses/frac_irrigated.nc")
        frac_irr_df = frac_irr_nc.to_dataframe().reset_index().round({'lon': 4, 'lat': 4})
    
        frac_irr_agent = frac_irr_df.merge(res_cca, on=['lat', 'lon'])
        merge_frac_irr_agent = frac_irr_agent.groupby(['Agent']).sum().reset_index()[
            ['Agent', 'fracVegCover[3]_monthend']]
    
        # merge_pop_cca_agent['CCA code'] = abs(merge_pop_cca_agent['Agent']) % 100
        merge_frac_irr_agent['CCA code'] = frac_irr_agent.groupby('Agent')['CCA code'].apply(
            lambda x: x.value_counts().index[0]).reset_index()['CCA code']
        merge_frac_irr_agent_weight = merge_frac_irr_agent.groupby(['CCA code']).sum().reset_index().merge(
            merge_frac_irr_agent, on=['CCA code'])
    
        merge_frac_irr_agent_weight['weight'] = merge_frac_irr_agent_weight['fracVegCover[3]_monthend_y'] / \
                                                merge_frac_irr_agent_weight['fracVegCover[3]_monthend_x']
        merge_frac_irr_agent_weight = merge_frac_irr_agent_weight[['Agent_y', 'weight']].rename(
            columns={'Agent_y': 'Agent'})
    
        ag_sw_avai_agent174 = ag_sw_avai_agent174.merge(merge_frac_irr_agent_weight, on='Agent')
        ag_sw_avai_agent174['SW_avai_ag_km3'] = ag_sw_avai_agent174["SW_avai_ag"] * ag_sw_avai_agent174['weight'] * 1E-9
        ag_sw_avai_agent174 = ag_sw_avai_agent174[['Agent', 'SW_avai_ag_km3']].rename(
            columns={'Agent': 'AgentID'}).fillna(0)
    else:
        print('no res has water')
        # urban_sw_avai_agent177 = pd.read_csv(basepath + '/swurban.csv')
        # ag_sw_avai_agent174 = pd.read_csv(basepath + '/swag.csv')
        
    return res_ini, release_df, urban_sw_avai_agent177, ag_sw_avai_agent174