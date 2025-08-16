__author__ = 'ajainf'
# =====================================================
# Imports
# =====================================================

from basepath_file import basepath
import numpy as np
import pandas as pd

# =====================================================
# Helper Functions
# =====================================================
# NOTES
# This file contains several functions to transform files between different formats
#
# USEFUL WEBSITES
# https://rasterio.readthedocs.io/en/latest/quickstart.html
# https://gis.stackexchange.com/questions/328247/get-row-column-of-latitude-longitude
# https://stackoverflow.com/questions/27861197/how-to-i-get-the-coordinates-of-a-cell-in-a-geotif
# =====================================================


# Excel Data to Pandas Dataframe 
# =====================================================
def get_excel_data(filename):
    """Return a pandas object of the given excel file. Looks by default in:
        ./domdem_filename

    """
    import pandas as pd
    import os
    excel_data = pd.ExcelFile(os.path.join(basepath, filename))
    return excel_data


# NetCDF Data to Pandas Dataframe 
# =====================================================
def netcdf_to_dataframe(netcdf_file_path):
    """Return a pandas dataframe object of a given netcdf file."""

    import xarray as xr
    dataset = xr.open_dataset(netcdf_file_path)
    dataframe = dataset.to_dataframe()
    return dataframe


# Pandas Dataframe to NetCDF 
# =====================================================
def dataframe_to_netcdf(dataframe):
    """Return a netcdf object of the given dataframe.
        See: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html

    """

    array = dataframe.to_xarray()
    nc = array.Dataset.to_netcdf()

    return nc


# Pandas Dataframe to NetCDF File 
# =====================================================
def dataframe_to_netcdf_file(dataframe, ncfile):
    """Return a netcdf file of the given dataframe.
    """
    import xarray as xr
    xr.Dataset.from_dataframe(dataframe).to_netcdf(path=ncfile)
    return


# Tiff Raster Band 1 to Dictionary 
# =====================================================
def tiff_to_dict(tiff_file_path):
    """Returns a dictionary from the 1st band of a tiff file
        See rasterio: https://rasterio.readthedocs.io/en/latest/
    """
    import rasterio
    tif = rasterio.open(tiff_file_path)
    band_num = 1
    tif_band = tif.read(band_num)
    tif_dict = {}
    count = 0
    for i in range(0, tif_band.shape[0]):
        for j in range(0, tif_band.shape[1]):
            key = tif.xy(i, j)
            value = count  # tif_band[i, j]
            tif_dict[key] = value
            count = count + 1
    # df = pd.DataFrame.from_dict(tif_dict)
    return tif_dict  # Should we also return a dataframe to view


# Lat Long to Index (eventually make into Row Col) 
# =====================================================
def find_row_col_of_lat_lon_points(latlon_tiff_file, points):
    """Return index of latlon points
        See # https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates

        :argument
        2D array points
        tiff file path
    """

    import rasterio
    from scipy import spatial
    import numpy as np

    # read image
    image_data = rasterio.open(latlon_tiff_file)
    print("Image Coordinate System- " + str(image_data.crs))  # coordinate system
    lons = image_data.read(1)  # band number 1
    lats = image_data.read(2)  # band number 2

    yidx = np.linspace(0, len(np.unique(lats)), len(np.unique(lats)) + 1)
    xidx = np.linspace(0, len(np.unique(lons)), len(np.unique(lons)) + 1)

    # meshgrid
    combined_lat_lon_array = np.dstack([lats.ravel(), lons.ravel()])[0]
    points_list = list(points.transpose())

    def do_kdtree(combined_lat_lon_array_in, points_in):
        mytree = spatial.cKDTree(combined_lat_lon_array_in)
        dist, indexes = mytree.query(points_in)
        return indexes

    result = do_kdtree(combined_lat_lon_array, points_list)

    return result


# Base Case NetCDF from hydro module 
# =====================================================
def create_hydro_netcdf_basefile():
    """Return a netcdf to use as a base for alignment"""

    import xarray as xr
    # This particular file was suggested by hydro module developer as good for basecase
    nc_test = xr.open_dataset(basepath + r'\modules\hydro\hydro_inputs\landsurface\fractionLandcover_UB_1km.nc')
    nc_small = nc_test.fracforest
    nc_small.attrs['standard_name'] = 'variable'
    nc_small.attrs['long_name'] = 'variable'
    nc_base = nc_small.isel(time=slice(1))  # base file will only have one time step
    nc_base.name = 'variable'
    nc_base.to_netcdf(basepath + '/modules/hydro/hydro_files/nc_base.nc')


# To use this, the dataframe must have a lat and lon component
###################################################################
def create_array(df, lat_lon, val):
    # path_cwatm_lat_lon_grid = basepath + r'\tests\lat_lon.xlsx'
    # lat_lon = pd.read_excel(path_cwatm_lat_lon_grid)
    # from helper_objects.helper_functions import find_nearest
    import numpy as np

    bhima_array = np.flipud(df.pivot(index='lat', columns='lon', values=val))
    # this array is located somewhere in the larger 320x370 box. place it in correct location of the 320x370

    # To find offset --------------
    minlat = df[['lat', 'lon']].lat.min()
    maxlat = df[['lat', 'lon']].lat.max()
    minlon = df[['lat', 'lon']].lon.min()

    # CWATM lats/lons--------------

    lats = np.array(lat_lon['lat'])  # list of CWatM lats-- lats.shape =320, from highest to lowest
    lats = lats[~np.isnan(lats)]
    lons = np.array(lat_lon['lon'])  # list of CwatM lons --- lons.shape=370

    val_x, ix_x = find_nearest(lats, minlat)
    val_x2, ix_x2 = find_nearest(lats, maxlat)
    val_y, ix_y = find_nearest(lons, minlon)
    # ----------------------------------
    # To position the smaller array in correct location of the 320x370 array ------
    a = bhima_array
    # quickplot
    # plt.imshow(bhima_array)
    result = np.empty((320, 370))  # create an empty array of the correct size
    result[:] = np.nan  # set all values to nan

    x_offset = result.shape[0] - ix_x2 -1  # 11 #result.shape[0] - bhima_array.shape[0]-(ix_x+1)  # 320-306-ix_x +1
    y_offset = ix_y   # 23  # 0 in your case # plus one b/c index starts counting at 0

    result[x_offset:a.shape[0] + x_offset, y_offset:a.shape[1] + y_offset] = a
    print('successfully updated')
    return result


# df to netcdf 
# =====================================================
def get_netcdf_from_dataframe(dataframe_in, varname, lat_name, lon_name, nc_template_file_path,
                              nc_var_name='nc_template'):
    # from netCDF4 import Dataset
    import xarray as xr
    import numpy as np

    def geo_idx(dd, dd_array):
        geo_idx = (np.abs(dd_array - dd)).argmin()
        return geo_idx

    # read in base netcdf
    nc_file_path = nc_template_file_path  # ;basepath + r'\modules\hydro\hydro_files\nc_base.nc'
    # ncfile_base = Dataset(nc_file_path, 'r', format='NETCDF4_CLASSIC') #using NetCDF4 package
    ncfile_base = xr.open_dataset(nc_file_path)  # using xarray package

    # find lats and lons
    lats = ncfile_base.variables['lat'][:].data
    lons = ncfile_base.variables['lon'][:].data

    # Dataset(nc_file_path, 'w', format='NETCDF4_CLASSIC') # no writing permission

    nc_out = ncfile_base
    len_time_dim = ncfile_base.variables[nc_var_name].data.shape[0]

    if len(ncfile_base.dims) > 2:
        nc_write = ncfile_base.variables[nc_var_name][:].data[0, :, :]
    else:
        nc_write = ncfile_base.variables[nc_var_name][:].data[:, :]

    for i in range(len(dataframe_in)):
        in_lat = dataframe_in.iloc[i][lat_name]
        in_lon = dataframe_in.iloc[i][lon_name]
        lat_idx = geo_idx(in_lat, lats)
        lon_idx = geo_idx(in_lon, lons)
        new_value = dataframe_in.iloc[i][varname]
        nc_write[lat_idx, lon_idx] = new_value

    if len(ncfile_base.dims) > 2:
        nc_out.variables[nc_var_name][:].data = np.tile(nc_write, (len_time_dim, 1, 1))
    else:
        nc_out.variables[nc_var_name][:].data = nc_write

    return nc_out


# Base Case NetCDF from hydro module 
# =====================================================
def write_netcdf_to_file(netcdf_in, filename, varname):
    # create a file to write to
    from netCDF4 import Dataset
    ncfile_new = Dataset(filename, 'w', format='NETCDF4_CLASSIC')

    lat_dd = []  # decimal degree
    lon_dd = []
    lat_dd[:] = netcdf_in.variables['lat'][:]
    lon_dd[:] = netcdf_in.variables['lon'][:]
    numlat = len(lat_dd)
    numlon = len(lon_dd)
    numtime = 1

    # create dimensions
    ncfile_new.createDimension('lat', numlat)
    ncfile_new.createDimension('lon', numlon)
    # ncfile_new.createDimension('time', numtime)

    # define variables
    # time = ncfile_new.createVariable('Time', 'd', 'time')
    latitude = ncfile_new.createVariable('Latitude', 'd', 'lat')
    longitude = ncfile_new.createVariable('Longitude', 'd', 'lon')

    # var = ncfile_new.createVariable(varname, 'd', ('time', 'lat', 'lon'))
    var = ncfile_new.createVariable(varname, 'd', ('lat', 'lon'))

    longitude[:] = lon_dd
    latitude[:] = lat_dd
    var[:] = netcdf_in['variable'][:]
    # time[:] = timein # works when this line is commented out to have no time

    # close ncfile
    ncfile_new.close()


# =====================================================
# GIS in pandas 

def transform_from_latlon(lat, lon):
    import numpy as np
    from affine import Affine

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, lat_name='latitude', lon_name='longitude'):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    import xarray as xr
    from rasterio import features

    transform = transform_from_latlon(coords[lat_name], coords[lon_name])
    out_shape = (len(coords[lat_name]), len(coords[lon_name]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=-9999, transform=transform,
                                dtype=float)

    return xr.DataArray(raster, coords=coords, dims=(lat_name, lon_name))  # mask


# For pyomo 
# =====================================================
def nc_data_aggregate_to_agent_area(netcdf_file_path, shapefile_path, agent_id_name, nc_var_name, lat_name, lon_name,
                                    plotcheck=False):
    ''' This function conducts a spatial join to aggregate points to an area '''

    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import Point

    # Shapefile holding agent id areas
    shapefile = gpd.read_file(shapefile_path)  # read in as lon, lat (x, y)

    # Read full netcdf file
    all_nc_data = xr.open_dataset(netcdf_file_path)
    nc_data = all_nc_data[nc_var_name]

    # Plot to check shapefile and netcdf data
    if plotcheck:
        ax = shapefile.plot(alpha=0.2, ec='black', fc="none")
        nc_data.plot(ax=ax, zorder=-1, alpha=0.5)

    # Convert nc to dataframe
    nc_df = nc_data.to_dataframe()
    nc_df.reset_index(inplace=True)

    # Create geometry points
    geometry_points = nc_df.apply(lambda x: Point([x[lon_name], x[lat_name]]),
                                  axis=1)  # note it has to be lon, lat (x,y)
    nc_df['geometry'] = geometry_points

    # Create goedataframe and set coordinate system
    nc_gdf = gpd.GeoDataFrame(nc_df, geometry=geometry_points)
    nc_gdf.crs = "EPSG:4326"

    # Spatial Join
    join = gpd.GeoDataFrame()
    # Check coordinate system
    if (shapefile.crs == nc_gdf.crs):
        print('Coordinate systems both EPSG:4326 - ready to do spatial join')
        join = gpd.sjoin(nc_gdf, shapefile, how="inner",
                         op="within")  # points first, operation = within; if shapefile first, operation = contains

    else:
        print('Shapefile Coordinate system does not match EPSG:4326')

    if plotcheck:
        print(join.head)
        join.plot()

    # Calculate the mean by group
    df_out = join.groupby(agent_id_name).mean()[[nc_var_name]]

    return df_out


# ---------------


# Tic Toc Generator to time function  
# =====================================================
# https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
import time


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


# Find index and nearest value in an array  
# =====================================================
def find_nearest(a, a0):
    import numpy as np
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], idx


# Merge and Joining dataframes 
# =====================================================
def merge_dataframes(list_of_dataframes, on_header):
    import pandas as pd
    from functools import reduce
    df_final = reduce(lambda left, right: pd.merge(left, right, on=on_header, how='outer'), list_of_dataframes)


# From dataframe to geodataframe 
# # ===================================================
def df_to_gdf(df_in, lon_name, lat_name, ):
    import geopandas as gpd
    from shapely.geometry import Point

    df = df_in.reset_index()
    geometry_points = df.apply(lambda x: Point([x[lon_name], x[lat_name]]), axis=1)
    df['geometry'] = geometry_points

    # Create goedataframe and set coordinate system
    gdf = gpd.GeoDataFrame(df, geometry=geometry_points)
    gdf.crs = "EPSG:4326"

    # gdf.plot(column=name, cmap='OrRd')
    return gdf


# Joining two geodataframes spatially on lat and long
# # ===================================================
def join_gdfs_on_latlon(gdf1, gdf2):
    gdf1['join_point'] = [(x, y) for x, y in zip(round(gdf1['lat'], 6), round(gdf1['lon'], 6))]
    gdf2['join_point'] = [(x, y) for x, y in zip(round(gdf2['lat'], 6), round(gdf2['lon'], 6))]
    joined_gdf = gdf1.merge(gdf2, on='join_point', how='outer')
    return joined_gdf


# Converting a geodataframe to tif
# # ===================================================
def gdf_to_tif(gdf, tif_path_out, lat_name, lon_name, z_value):
    import numpy as np
    import rasterio

    np_array = np.flipud(gdf.pivot(index=lat_name, columns=lon_name, values=z_value))
    # plt.imshow(np_array)

    tif_path = tif_path_out

    with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=np_array.shape[0],
            width=np_array.shape[1],
            count=1,
            dtype=np_array.dtype,
            crs='+proj=latlong',
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),  
    ) as data:
        data.write(np_array, 1)

        return np_array


# Converting tiffs to netcdf
def tiff_to_nc_old(folder):
    # # ===================================================
    import os
    import netCDF4 as nc
    import numpy as np
    import xarray as xr

    from osgeo import gdal
    from basepath_file import basepath

    # tiff file has bands -- need fraction -- band 1

    for each_tiff_file in os.listdir(folder + '\\tiffs\\'):
        name, ext = os.path.splitext(each_tiff_file)
        print(name)
        # convert tiff to nc
        # open tiff file
        # ds = gdal.Open(folder + '\\tiffs\\' + each_tiff_file)
        # ds = None  # empty dataset
        ds = gdal.Open(folder + '\\tiffs\\' + each_tiff_file)
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()

        # save as nc in folder
        # create netcdf to write to
        outfile = folder + "\\netcdfs\\" + name + ".nc"
        nc_file = nc.Dataset(outfile, 'w', format='NETCDF4')

        lat = nc_file.createDimension('lat', np.shape(array)[0])
        lon = nc_file.createDimension('lon', np.shape(array)[1])

        lats = nc_file.createVariable('lat', 'f4', ('lat',))
        lons = nc_file.createVariable('lon', 'f4', ('lon',))
        value = nc_file.createVariable('urban_fraction', 'f4', ('lat', 'lon',))
        value.units = 'Unknown'

        # cwatm template file
        nc_template_file = basepath + r'\modules\hydro\hydro_files\bhima_cellArea_m2_30s.nc'
        template = xr.open_dataset(nc_template_file)

        lats[:] = template.coords['lat'].__array__()
        lons[:] = template.coords['lon'].__array__()

        # value[:, :] = np.flipud(array)  # check if this has to be flipped up down
        value[:, :] = array

        nc_file.close()
        return


# nc_template_file = basepath + r'\modules\hydro\hydro_files\bhima_cellArea_m2_30s.nc'
# Note template file and tiff file to convert must be same size 
def tiff_to_nc(nc_template_file, tiff_file, out_nc_file, varname, units):
    # # ===================================================
    import netCDF4 as nc
    import numpy as np
    import xarray as xr
    from osgeo import gdal

    ds = gdal.Open(tiff_file)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()

    print('Tiff Shape')
    print(array.shape)

    # save as nc in folder
    # create netcdf to write to
    nc_file = nc.Dataset(out_nc_file, 'w', format='NETCDF4')

    lat = nc_file.createDimension('lat', np.shape(array)[0])
    lon = nc_file.createDimension('lon', np.shape(array)[1])

    lats = nc_file.createVariable('lat', 'f4', ('lat',))
    lons = nc_file.createVariable('lon', 'f4', ('lon',))
    value = nc_file.createVariable(varname, 'f4', ('lat', 'lon',))
    value.units = units

    # cwatm template file
    template = xr.open_dataset(nc_template_file)

    lats[:] = template.coords['lat'].__array__()
    lons[:] = template.coords['lon'].__array__()

    # value[:, :] = np.flipud(array)  # check if this has to be flipped up down
    value[:, :] = array

    nc_file.close()
    return


def create_fixed_grid():
    # # ===================================================
    import numpy as np
    import netCDF4 as nc
    import xarray as xr

    # cwatm template file
    nc_template_file = basepath + r'\modules\hydro\hydro_files\bhima_cellArea_m2_30s.nc'
    template = xr.open_dataset(nc_template_file)

    # create a grid of appropriate size
    xlen = template['lon'].shape[0]  # 370
    ylen = template['lat'].shape[0]  # 320

    gridx = np.array([[i for i in range(xlen)] for j in range(ylen)])
    gridy = np.array([[j for i in range(xlen)] for j in range(ylen)])

    # create netcdf
    outfile = r"C:\Users\ajainf1\Documents\GitHub\nira_repo\modules\hydro\hydro_files\cwatm_template_09-28-2020.nc"

    nc_file = nc.Dataset(outfile, 'w', format='NETCDF4')

    lat = nc_file.createDimension('lat', template['lat'].shape[0])
    lon = nc_file.createDimension('lon', template['lon'].shape[0])

    lats = nc_file.createVariable('lat', 'f4', ('lat',))
    lons = nc_file.createVariable('lon', 'f4', ('lon',))
    value1 = nc_file.createVariable('x_index', 'f4', ('lat', 'lon',))
    value2 = nc_file.createVariable('y_index', 'f4', ('lat', 'lon',))
    value1.units = 'Unknown'
    value2.units = 'Unknown'

    lats[:] = template.coords['lat'].__array__()
    lons[:] = template.coords['lon'].__array__()

    value1[:, :] = np.flipud(gridx)  # check if this has to be flipped up down
    value2[:, :] = np.flipud(gridy)  # check if this has to be flipped up down

    nc_file.close()


def create_netcdf_from_array_urban(nc_template_file_path, nc_outfile_path, array_in1, array_in2, varname1,
                                   varname2, time_urban):
    # # ===================================================
    import netCDF4 as nc
    import xarray as xr
    import numpy as np

    # cwatm template file
    template = xr.open_dataset(nc_template_file_path)

    # create a grid of appropriate size
    xlen = template['lon'].shape[0]  # 370
    ylen = template['lat'].shape[0]  # 320

    nc_file = nc.Dataset(nc_outfile_path, 'w', format='NETCDF4')

    lat = nc_file.createDimension('lat', template['lat'].shape[0])
    lon = nc_file.createDimension('lon', template['lon'].shape[0])
    time = nc_file.createDimension('time', 1)
    lats = nc_file.createVariable('lat', 'f4', ('lat',))
    lons = nc_file.createVariable('lon', 'f4', ('lon',))
    times = nc_file.createVariable('time', 'int', ('time'))
    array1 = nc_file.createVariable(varname1, 'f4', ('time', 'lat', 'lon',))
    array2 = nc_file.createVariable(varname2, 'f4', ('time', 'lat', 'lon',))

    array1.units = 'Unknown'
    array2.units = 'Unknown'

    lats[:] = template.coords['lat'].__array__()
    lons[:] = template.coords['lon'].__array__()
    times[0] = 0

    times.setncattr('units', 'days since ' + time_urban)
    times.setncattr('standard_name', 'time')
    times.setncattr('calendar', 'standard')

    # array1[:, :] = np.flipud(array_in)  # check if this has to be flipped up down
    array1[0, :, :] = np.nan_to_num(array_in1)
    array2[0, :, :] = np.nan_to_num(array_in2)

    nc_file.close()


def create_netcdf_from_array(nc_template_file_path, nc_outfile_path, array_in, varname):
    # # ===================================================
    import netCDF4 as nc
    import xarray as xr
    import numpy as np

    # cwatm template file
    # nc_template_file = basepath + r'\modules\hydro\hydro_files\bhima_cellArea_m2_30s.nc'
    template = xr.open_dataset(nc_template_file_path)

    # create a grid of appropriate size
    xlen = template['lon'].shape[0]  # 370
    ylen = template['lat'].shape[0]  # 320
    # print('Grid Shape:' + str(ylen) + " x " + str(xlen))
    # print('Array In Shape:' + str(array_in.shape))

    # create netcdf
    # outfile = r"C:\Users\ajainf1\Documents\GitHub\nira_repo\modules\hydro\hydro_files\cwatm_template_09-28-2020.nc"

    nc_file = nc.Dataset(nc_outfile_path, 'w', format='NETCDF4')

    lat = nc_file.createDimension('lat', template['lat'].shape[0])
    lon = nc_file.createDimension('lon', template['lon'].shape[0])
    lats = nc_file.createVariable('lat', 'f4', ('lat',))
    lons = nc_file.createVariable('lon', 'f4', ('lon',))
    array1 = nc_file.createVariable(varname, 'f4', ('lat', 'lon',))

    array1.units = 'Unknown'

    lats[:] = template.coords['lat'].__array__()
    lons[:] = template.coords['lon'].__array__()

    # array1[:, :] = np.flipud(array_in)  # check if this has to be flipped up down
    array1[:, :] = np.nan_to_num(array_in)

    nc_file.close()


def add_months(sourcedate, months):
    # =====================================================
    import datetime
    import calendar

    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def dict_to_csv(my_dict, csv_file):
    # =====================================================
    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}
    # csv_file = 'test.csv'
    with open(csv_file, 'w') as f:
        for key in my_dict.keys():
            f.write("%s,%s\n" % (key, my_dict[key]))


def time_idx(da, year_selected, month_selected, day_selected):
    mask = np.logical_and(np.logical_and((da.indexes['time'].year == year_selected),
                                         (da.indexes['time'].month == month_selected)),
                          (da.indexes['time'].day == day_selected))
    idx = np.where(mask == True)[0].item()
    return idx


def data_selection(da, start_year, start_month, start_day, end_year, end_month, end_day):
    da_mask_start = time_idx(da, start_year, start_month, start_day)
    da_mask_end = time_idx(da, end_year, end_month, end_day)

    da_selected = da[da_mask_start: da_mask_end + 1]

    year_month_idx = pd.MultiIndex.from_arrays(
        [da_selected['time.year'], da_selected['time.month']])
    da_selected.coords['year_month'] = ('time', year_month_idx)
    return da_selected


def geo_idx(dd, dd_array):
   """
     search for nearest decimal degree in an array of decimal degrees and return the index.
     np.argmin returns the indices of minium value along an axis.
     so subtract dd from all values in dd_array, take absolute value and find index of minium.
    """
   geo_idx = (np.abs(dd_array - dd)).argmin()
   return geo_idx