#!/usr/bin/env python
# coding: utf-8

# Input data ............................................................................................................

## Hydraulic parameters ranges [min, max]
'''
Values were taken from different reports and research
Where:
   VAB = Vesicular amygdaloidal basalts
   CB  = Consolidated basalts
   AF  = Alluvial formation
   Txy = Transmissivity in 'x' and 'y' axes
   Sy  = Specific yield
'''
### Transmissivities (m2/d)
Txy_VAB = [50,190]
Txy_CB  = [20,60]
Txy_AF  = [97,248]

### Storativity or Specific yield (unitless)
Sy_VAB = [0.01,0.03]
Sy_CB  = [0.008,0.01]
Sy_AF  = [0.05,0.09]

### Alluvial formation
alluv_shp_dir = r'.\SHP\alluvium_formation.shp'    # Directory where Alluvium shapefile is located
alluv_depth   = 25            # Alluvium thickness (meters)

## Aquifer thickness
dem_dir = r'.\RST\DEM.tif'    # Directory where DEM is located
thickness_type = 2            # Aquifer thickness: [Constant = 1], [Decreasing with DEM = 2]
depth_ave   = 100             # If thickness_type = 1, indicate average aquifer thickness (meters)
depth_top   = 50              # If thickness_type = 2, indicate thickness at the top of the hills/mountains (meters)

## Raster properties template
template = r'.\RST\DEM.tif'

# Importing modules .....................................................................................................
from osgeo import gdal, gdal_array
import numpy, xarray, sys, os

# Pre-processing ........................................................................................................

## Getting template raster properties
src = gdal.Open(template)
if src is None:
    print('Could not open template file')
    sys.exit(1)

xmin, dx, xskew, ymax, yskew, dy = src.GetGeoTransform()
ncol, nrow = src.RasterXSize, src.RasterYSize
xmax, ymin = xmin + ncol*dx, ymax - nrow*-dy

## Creating xarray definition
def raster(array, xmin, xmax, dx, ymin, ymax, dy):
    raster = xarray.DataArray(array,
        coords={
            "y": numpy.arange(ymin + (0.5 * -dy), ymax, -dy),
            "x": numpy.arange(xmin + (0.5 *  dx), xmax,  dx),
            "dx": dx,
            "dy": dy,},
        dims=("y", "x"),)
    return raster

# Aquifer thickness .....................................................................................................
'''
- Constant aquifer thickness (thickness_type == 1)
     Assumes an aquifer with a constant thickness. Needs an average aquifer thickness.
- Decreasing with DEM (thickness_type == 2)
     Assumes an aquifer thinner at the top of the hills and thicker at alluvial plains 
     due to weathering, deposition and sedimentation processes. Needs the aquifer thickness 
     at the top of the hills and considers the average aquifer thickness the depth at the bottom
'''
if thickness_type == 1:
    thickness = raster(numpy.full((nrow, ncol), depth_ave), xmin, xmax, dx, ymin, ymax, dy)
elif thickness_type == 2:
    dem = gdal_array.LoadFile(dem_dir)
    dem_min, dem_max = numpy.where(dem > 0, dem, numpy.inf).min(), numpy.where(dem < 8000, dem, -numpy.inf).max()
    dec_depth = depth_ave - (depth_ave - depth_top) / (dem_max - dem_min) * (dem - dem_min)
    dec_depth = numpy.where((dec_depth == dec_depth.max()) | (dec_depth == dec_depth.min()), 100, dec_depth)
    thickness = raster(numpy.flip(dec_depth, 0), xmin, xmax, dx, ymin, ymax, dy)

# Hydraulic properties ..................................................................................................
## Randomizing hydrogeological units
'''
dist -> raster of [0,1] to indicate the type of hydrogeological unit accross the study area
Where:
   0 = Vesicular amygdaloidal basalts
   1 = Consolidated basalts
'''
dist = numpy.random.randint(2, size=[nrow, ncol])
dist = raster(dist, xmin, xmax, dx, ymin, ymax, dy)

## Rasterizing alluvial formation shapefile
'''
Lithology information retrived from Geological Survey of India
alluv -> raster of [0,1] created from shapefile that only contains alluvium formations to identify their location
Where:
   0 = Non-alluvial formation
   1 = Alluvial formation
'''
os.system('gdal_rasterize -l alluvium_formation -burn 1.0 -tr ' + 
          str(dx) + ' ' + str(-dy) + ' -a_nodata 0.0 -te ' + 
          str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + 
          ' -ot Int32 -of GTiff ' + str(alluv_shp_dir) + ' "./RST/alluvial_rst.tif"')
alluv = numpy.flip(gdal_array.LoadFile('./RST/alluvial_rst.tif'), 0)
if alluv is None:
    print('Could not open alluvial file')
    sys.exit(1)
alluv = raster(alluv, xmin, xmax, dx, ymin, ymax, dy)

## Randomizing hydraulic parameters of hydrogeological units
'''
Randomizes the hydraulic parameters of the VAB, CB, AF hydrogeological units,
based on the given thresholds [min, max]
using a Gaussian distribution of '0' mean and '1' standard deviation
Assumes that the entire raster extend is covered by this unit
'''
### Creating a normally distributed raster with template extensions
gauss = numpy.random.normal(loc=0.0, scale=1.0, size=[nrow,ncol])

### Randomizing hydraulic parameters
hyd_prop = ['Txy_VAB', 'Txy_CB', 'Txy_AF', 'Sy_VAB', 'Sy_CB', 'Sy_AF']
for i in hyd_prop:
    globals()[i+'_rst'] = (globals()[i][1]-globals()[i][0])/(gauss.max()-gauss.min())*(gauss-gauss.min())+globals()[i][0]
    globals()[i+'_rst'] = raster(globals()[i+'_rst'], xmin, xmax, dx, ymin, ymax, dy)

## Unifying results
'''
Uses the randomized hydraulic properties and combine them in a
Hydraulic conductivity (m/s) and Specific yield (unitless) arrays
An harmonic mean of the hydraulic property was calculated where alluvial formations matched with VAB or CB
'''
## Hydraulic conductivity (m/s)
Kxy = Txy_VAB_rst.where((alluv == 0) & (dist == 0), Txy_CB_rst)
Kxy = Kxy.where(alluv == 0, thickness/((alluv_depth/Txy_AF_rst)+((thickness-alluv_depth)/Txy_VAB_rst)))
Kxy = Kxy.where((alluv == 0) | ((alluv == 1) & (dist == 0)), thickness/((alluv_depth/Txy_AF_rst)+((thickness-alluv_depth)/Txy_CB_rst)))
Kxy = Kxy / thickness / 86400

## Specific yield (unitless)
Sy = Sy_VAB_rst.where((alluv == 0) & (dist == 0), Sy_CB_rst)
Sy = Sy.where(alluv == 0, thickness/((alluv_depth/Sy_AF_rst)+((thickness-alluv_depth)/Sy_VAB_rst)))
Sy = Sy.where((alluv == 0) | ((alluv == 1) & (dist == 0)), thickness/((alluv_depth/Sy_AF_rst)+((thickness-alluv_depth)/Sy_CB_rst)))

# Exporting netcdf ......................................................................................................
thickness.to_netcdf('./_outputs/thickness.nc')
Kxy.to_netcdf('./_outputs/Kxy.nc')
Sy.to_netcdf('./_outputs/Sy.nc')