__author__ = 'Anjui Jain Figueroa'

# import gdal
import matplotlib.pyplot as plt
import netCDF4 as nc
# import os
import numpy as np

from basepath_file import basepath
from netsim.model_components.institutions._pune_institution import PuneInstitution


class HydroCWATM(PuneInstitution):
    """
    functions to run cwatm
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.target = None

    def create_cwatm_settings_dict_obj(self, xl_settings_file_path):
        import xlrd
        import pandas as pd
        # Create dictionary from excel file
        settings_dict = {}
        df = pd.read_excel(xl_settings_file_path)
        wb = xlrd.open_workbook(xl_settings_file_path)
        sh = wb.sheet_by_index(0)
        for i in range(len(df) + 1):
            cell_key = sh.cell(i, 0).value.strip()
            cell_value = sh.cell(i, 1).value
            settings_dict[cell_key] = cell_value
        return settings_dict

    def show_cwatm_settings_file(self, settings_file_path):
        file = open(settings_file_path, "r")
        for line in file:
            print(line)
        file.close()

    def save_cwatm_settings_file(self, settings_file_path, save_path, new_name):
        import shutil
        source = settings_file_path
        destination = save_path + "/" + new_name
        newPath = shutil.copy(source, destination)
        print("Saved a copy of settings file : ", newPath)

    def run_cwatm(self, settings_file_path):
        import os
        cwatm_source_path = basepath + r"/CWatM/run_cwatm.py"
        # settings_file_path = basepath + '\modules\hydro\hydro_inputs\settings\settings_bhima_template.ini'
        command_str = 'python ' + cwatm_source_path + " " + settings_file_path + ' -l'
        flag = os.system(command_str)
        if flag == 0:
            print('Successfully Finished Running CWATM\n')

    def define_cwatm_settings(self, settings_template, new_settings_file, settings_dictionary):
        print('Defining cwatm settings')
        sourcefile = open(settings_template, "r")
        newfile = open(new_settings_file, "wt")
        keylist = list(settings_dictionary.keys())
        for line in sourcefile:
            lineparts = list(line.split())  # cannot have spaces
            boolcheck = [elem in lineparts for elem in keylist]
            match = any(elem in lineparts for elem in keylist)
            ix = boolcheck.index(match)
            if match:
                key = keylist.pop(ix)
                value = settings_dictionary[key]
                newstring = key + ' = ' + str(value) + '\n'
                newfile.write(newstring)
            else:
                newfile.write(line)
        sourcefile.close()
        newfile.close()

    def find_all_instances_in_file(self, file_name, text):
        with open(file_name, "r") as file_handle:
            file_contents = file_handle.read()
        for i in range(len(file_contents.splitlines())):
            line = file_contents.splitlines()[i]
            if text in line:
                print(str(i) + " " + line)

    def replace_line(self, file_name, line_num, text):
        lines = open(file_name, mode='r').readlines()
        lines[line_num] = text
        out = open(file_name, mode='w')
        out.writelines(lines)
        out.close()

    def modify_cwatm_fracland(self, area_mask):
        fracland = nc.Dataset('../hydro_inputs/landsurface/fractionLandcover_UB_1km.nc')
        excluded = ['lat', 'lon', 'time']
        vars = list(fracland.variables.keys())
        land_cover_categories = [elem for elem in vars if elem not in excluded]
        lon_array = fracland.variables['lon'][:]
        lat_array = fracland.variables['lat'][:]
        grid = np.meshgrid(lat_array, lon_array)

        # Original fracland
        time0 = 0  #
        for land_cat in land_cover_categories:
            old_fracland = fracland[land_cat][time0, :, :].data
            fracland_map = np.ma.masked_where(np.ma.getmask(area_mask), old_fracland)  # applies the area_mask
            plt.imshow(fracland_map)
            print(np.shape(old_fracland))

            new_fracland = old_fracland.deepcopy()

            if land_cat == 'fracsealed':  # Urban condition
                lat_bnds, lon_bnds = [40, 43], [-96, -89]
                lat_inds = np.where((lat_array > lat_bnds[0]) & (lat_array < lat_bnds[1]))
                lon_inds = np.where((lon_array > lon_bnds[0]) & (lon_array < lon_bnds[1]))


                greencityfactor = 0.1  # the higher the factor the less green
                new_fracland[:, lat_inds,lon_inds] = 1  # from URBAN MAP #(1-greencityfactor)*1  #new_val  # this new value depends on the land category; read from different files

            else:  # farmer condition
                lat_bnds, lon_bnds = [40, 43], [-96, -89]
                lat_inds = np.where((lat_array > lat_bnds[0]) & (lat_array < lat_bnds[1]))
                lon_inds = np.where((lon_array > lon_bnds[0]) & (lon_array < lon_bnds[1]))
                new_fracland[:, lat_inds, lon_inds] = 0  # read corresponding value from farmer files

            new_fracland_map = np.ma.masked_where(np.ma.getmask(area_mask), new_fracland)  # applies the area_mask
            plt.imshow(new_fracland_map)

    def create_area_mask(self):
        area_map = basepath + r"/modules/hydro/hydro_inputs/areamaps/Sarati_UB.map"
        ds = gdal.Open(area_map)
        roi = ds.ReadAsArray()  # region of interest
        area_mask = np.ma.masked_where(roi > 1, roi)
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        plt.imshow(area_mask)
        return area_mask

    def create_netcdf_from_array(nc_template_file_path, nc_outfile_path, array_in, varname):
        import numpy as np
        import netCDF4 as nc
        import xarray as xr

        # cwatm template file
        # nc_template_file = basepath + r'\modules\hydro\hydro_files\bhima_cellArea_m2_30s.nc'
        template = xr.open_dataset(nc_template_file_path)

        # create a grid of appropriate size
        xlen = template['lon'].shape[0]  # 370
        ylen = template['lat'].shape[0]  # 320
        print('Grid Shape:' + str(ylen) + " x " + str(xlen))
        print('Array In Shape:' + str(array_in.shape))

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

        array1[:, :] = np.flipud(array_in)  # check if this has to be flipped up down

        nc_file.close()
