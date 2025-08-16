# -------------------------------------------------------------------------
# Name:        Water demand module
#
# Author:      PB, MS, LG, JdeB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import numpy as np
from cwatm.management_modules import globals

from cwatm.management_modules.replace_pcr import npareatotal, npareamaximum
from cwatm.management_modules.data_handling import returnBool, binding, cbinding, loadmap, divideValues, checkOption, npareaaverage, readnetcdf2
from cwatm.hydrological_modules.water_demand.domestic import waterdemand_domestic
from cwatm.hydrological_modules.water_demand.industry import waterdemand_industry
from cwatm.hydrological_modules.water_demand.livestock import waterdemand_livestock
from cwatm.hydrological_modules.water_demand.irrigation import waterdemand_irrigation
from cwatm.hydrological_modules.water_demand.environmental_need import waterdemand_environmental_need


#PB1507
from cwatm.management_modules.data_handling import *


class water_demand:
    """
    WATERDEMAND

    Calculating water demand and attributing sources to satisfy demands
    Industrial, domestic, and livestock are based on precalculated maps
    Agricultural water demand based on water need by plants
    
    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    domesticDemand                                                                                                   
    pot_domesticConsumpt                                                                                             
    envFlow                                                                                                          
    readAvlStorGroundwat  same as storGroundwater but equal to 0 when inferior to a treshold                m        
    industryDemand                                                                                                   
    efficiencyPaddy                                                                                                  
    efficiencyNonpaddy                                                                                               
    returnfractionIrr                                                                                                
    irrDemand                                                                                                        
    irrPaddyDemand                                                                                                   
    irrNonpaddyDemand                                                                                                
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --       
    waterBodyBuffer                                                                                                  
    compress_LR           boolean map as mask map for compressing lake/reservoir                            --       
    decompress_LR         boolean map as mask map for decompressing lake/reservoir                          --       
    MtoM3C                conversion factor from m to m3 (compressed map)                                   --       
    waterBodyTypCTemp                                                                                                
    livestockDemand                                                                                                  
    pot_livestockConsump                                                                                             
    MtoM3                 Coefficient to change units                                                       --       
    InvDtSec                                                                                                         
    cellArea              Area of cell                                                                      m2       
    M3toM                 Coefficient to change units                                                       --       
    channelStorage                                                                                                   
    fracVegCover          Fraction of specific land covers (0=forest, 1=grasslands, etc.)                   %        
    nonFossilGroundwater  groundwater abstraction which is sustainable and not using fossil resources       m        
    lakeVolumeM3C         compressed map of lake volume                                                     m3       
    lakeStorageC                                                                                            m3       
    reservoirStorageM3C                                                                                              
    lakeResStorageC                                                                                                  
    lakeResStorage                                                                                                   
    smalllakeVolumeM3                                                                                                
    smalllakeStorage                                                                                                 
    act_SurfaceWaterAbst                                                                                             
    addtoevapotrans                                                                                                  
    act_irrWithdrawal                                                                                                
    act_nonIrrConsumptio                                                                                             
    returnFlow                                                                                                       
    act_irrConsumption    actual irrigation water consumption                                               m        
    unmetDemand                                                                                                      
    act_nonIrrWithdrawal                                                                                             
    returnflowIrr                                                                                                    
    nonIrrReturnFlowFrac                                                                                             
    unmet_lost                                                                                                       
    act_totalWaterWithdr                                                                                             
    act_bigLakeResAbst                                                                                               
    act_smallLakeResAbst                                                                                             
    modflowPumpingM                                                                                                  
    modflowTopography                                                                                                
    modflowDepth2                                                                                                    
    leakageC                                                                                                         
    dom_efficiency                                                                                                   
    demand_unit                                                                                                      
    pot_industryConsumpt                                                                                             
    ind_efficiency                                                                                                   
    unmetDemandPaddy                                                                                                 
    unmetDemandNonpaddy                                                                                              
    totalIrrDemand                                                                                                   
    liv_efficiency                                                                                                   
    waterdemandFixed                                                                                                 
    waterdemandFixedYear                                                                                             
    allocSegments                                                                                                    
    swAbstractionFractio                                                                                             
    allocation_zone                                                                                                  
    modflowPumping                                                                                                   
    leakage                                                                                                          
    pumping                                                                                                          
    nonIrruse                                                                                                        
    act_indDemand                                                                                                    
    act_domDemand                                                                                                    
    act_livDemand                                                                                                    
    nonIrrDemand                                                                                                     
    totalWaterDemand                                                                                                 
    act_indConsumption                                                                                               
    act_domConsumption                                                                                               
    act_livConsumption                                                                                               
    act_totalIrrConsumpt                                                                                             
    act_totalWaterConsum                                                                                             
    pot_GroundwaterAbstr                                                                                             
    pot_nonIrrConsumptio                                                                                             
    readAvlChannelStorag                                                                                             
    act_channelAbst                                                                                                  
    reservoir_command_ar                                                                                             
    leakageC_daily                                                                                                   
    leakageC_daily_segme                                                                                             
    act_irrNonpaddyWithd                                                                                             
    act_irrPaddyWithdraw                                                                                             
    act_irrPaddyDemand                                                                                               
    act_irrNonpaddyDeman                                                                                             
    act_indWithdrawal                                                                                                
    act_domWithdrawal                                                                                                
    act_livWithdrawal                                                                                                
    act_paddyConsumption                                                                                             
    act_nonpaddyConsumpt                                                                                             
    returnflowNonIrr                                                                                                 
    unmet_lostirr                                                                                                    
    unmet_lostNonirr                                                                                                 
    waterabstraction                                                                                                 
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.var
        self.model = model

        self.domestic = waterdemand_domestic(model)
        self.industry = waterdemand_industry(model)
        self.livestock = waterdemand_livestock(model)
        self.irrigation = waterdemand_irrigation(model)
        self.environmental_need = waterdemand_environmental_need(model)

    def initial(self):
        """
        Initial part of the water demand module
        """

        self.var.includeIndusDomesDemand = True
        # True if all demands are taken into account,
        # False if not only irrigation is considered
        # This variable has no impact if includeWaterDemand is False
        if "includeIndusDomesDemand" in option:
            self.var.includeIndusDomesDemand = checkOption('includeIndusDomesDemand')


        # 🕵 Variables related to agents =========================

        self.var.activate_domestic_agents = False
        if 'activate_domestic_agents' in option:
            if checkOption('activate_domestic_agents'):
                self.var.activate_domestic_agents = True

        self.var.activate_irrigation_agents = False
        if 'activate_irrigation_agents' in option:
            if checkOption('activate_irrigation_agents'):
                self.var.activate_irrigation_agents = True

        self.var.relaxGWagent = globals.inZero.copy()
        self.var.relaxSWagent = globals.inZero.copy()
        self.var.irrWithdrawalSW_max = globals.inZero.copy()
        self.var.irrWithdrawalGW_max = globals.inZero.copy()

        self.var.relax_irrigation_agents = False
        if 'relax_irrigation_agents' in option:
            if checkOption('relax_irrigation_agents'):
                self.var.relax_irrigation_agents = True

        if 'adminSegments' in binding:
            # adminSegments (administrative segment) are collections of cells that are associated together, ie Agents
            # Cells within the same administrative segment have the same positive integer value.
            # Cells with non-positive integer values are all associated together.
            # Irrigation agents use adminSegments

            self.var.adminSegments = loadmap('adminSegments').astype(int)
            self.var.adminSegments = np.where(self.var.adminSegments>0, self.var.adminSegments, 0)

            if 'irrigation_agent_SW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                self.var.irrWithdrawalSW_max = npareaaverage(
                    loadmap('irrigation_agent_SW_request_month_m3') + globals.inZero.copy(),
                    self.var.adminSegments)

                if 'relax_sw_agent' in binding:
                    if self.var.loadInit:
                        self.var.relaxSWagent = self.var.load_initial('relaxSWagent')
                    else:
                        self.var.relaxSWagent = loadmap('relax_sw_agent')

            if 'irrigation_agent_GW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                self.var.irrWithdrawalGW_max = npareaaverage(
                    loadmap('irrigation_agent_GW_request_month_m3') + globals.inZero.copy(),
                    self.var.adminSegments)

                if 'relax_gw_agent' in binding:
                    if self.var.loadInit:
                        self.var.relaxGWagent = self.var.load_initial('relaxGWagent')
                    else:
                        self.var.relaxGWagent = loadmap('relax_gw_agent')

            if 'relax_abstraction_fraction_initial' in binding:
                self.var.relax_abstraction_fraction_initial = loadmap('relax_abstraction_fraction_initial') + globals.inZero.copy()
            else:
                self.var.relax_abstraction_fraction_initial = 0.5 + globals.inZero.copy()

        # =======================================================

        if checkOption('includeWaterDemand'):

            if self.var.includeIndusDomesDemand:  # all demands are taken into account

                self.domestic.initial()
                self.industry.initial()
                self.livestock.initial()
                self.irrigation.initial()
                self.environmental_need.initial()
            else:  # only irrigation is considered

                self.irrigation.initial()
                self.environmental_need.initial()

            # if waterdemand is fixed it means it does not change between years.
            self.var.waterdemandFixed = False
            if "waterdemandFixed" in binding:
                if returnBool('waterdemandFixed'):
                    self.var.waterdemandFixed = True
                    self.var.waterdemandFixedYear = loadmap('waterdemandFixedYear')

            self.var.sectorSourceAbstractionFractions = False
            # Sector-,source-abstraction fractions facilitate designating the specific source for the specific sector
            # Sources: River, Lake, Reservoir, Groundwater
            # Sectors: Domestic, Industry, Livestock, Irrigation
            # Otherwise, one can distinguish only between surface and groundwater, irrigation and non-irrigation

            if 'sectorSourceAbstractionFractions' in option:
                if checkOption('sectorSourceAbstractionFractions'):
                    print('Sector- and source-specific abstraction fractions are activated')
                    self.var.sectorSourceAbstractionFractions = True

                    self.var.swAbstractionFraction_Channel_Domestic = loadmap(
                        'swAbstractionFraction_Channel_Domestic')
                    self.var.swAbstractionFraction_Channel_Livestock = loadmap(
                        'swAbstractionFraction_Channel_Livestock')
                    self.var.swAbstractionFraction_Channel_Industry = loadmap(
                        'swAbstractionFraction_Channel_Industry')
                    self.var.swAbstractionFraction_Channel_Irrigation = loadmap(
                        'swAbstractionFraction_Channel_Irrigation')

                    self.var.swAbstractionFraction_Lake_Domestic = loadmap(
                        'swAbstractionFraction_Lake_Domestic')
                    self.var.swAbstractionFraction_Lake_Livestock = loadmap(
                        'swAbstractionFraction_Lake_Livestock')
                    self.var.swAbstractionFraction_Lake_Industry = loadmap(
                        'swAbstractionFraction_Lake_Industry')
                    self.var.swAbstractionFraction_Lake_Irrigation = loadmap(
                        'swAbstractionFraction_Lake_Irrigation')

                    self.var.swAbstractionFraction_Res_Domestic = loadmap(
                        'swAbstractionFraction_Res_Domestic')
                    self.var.swAbstractionFraction_Res_Livestock = loadmap(
                        'swAbstractionFraction_Res_Livestock')
                    self.var.swAbstractionFraction_Res_Industry = loadmap(
                        'swAbstractionFraction_Res_Industry')
                    self.var.swAbstractionFraction_Res_Irrigation = loadmap(
                        'swAbstractionFraction_Res_Irrigation')

                    self.var.gwAbstractionFraction_Domestic = loadmap(
                        'gwAbstractionFraction_Domestic')
                    self.var.gwAbstractionFraction_Livestock = loadmap(
                        'gwAbstractionFraction_Livestock')
                    self.var.gwAbstractionFraction_Industry = loadmap(
                        'gwAbstractionFraction_Industry')
                    self.var.gwAbstractionFraction_Irrigation = loadmap(
                        'gwAbstractionFraction_Irrigation')

            self.var.using_reservoir_command_areas = False
            if 'using_reservoir_command_areas' in option:
                if checkOption('using_reservoir_command_areas'):

                    self.var.using_reservoir_command_areas = True
                    self.var.reservoir_command_areas = loadmap('reservoir_command_areas').astype(int)


                    # Lakes within command areas are removed from the command area
                    self.var.reservoir_command_areas = np.where(loadmap('waterBodyTyp').astype(int) == 1,
                                                                0, self.var.reservoir_command_areas)
                    self.var.segmentArea = np.where(self.var.reservoir_command_areas > 0,
                                                    npareatotal(self.var.cellArea,
                                                                self.var.reservoir_command_areas), self.var.cellArea)

                    # Water abstracted from reservoirs leaks along canals related to conveyance efficiency.
                    # Canals are a map where canal cells have the number of the command area they are associated with
                    # Command areas without canals experience leakage equally throughout the command area

                    if 'canals' in binding:
                        self.var.canals = loadmap('canals').astype(int)
                    else:
                        self.var.canals = globals.inZero.copy()

                    self.var.canals = np.where(self.var.canals != self.var.reservoir_command_areas, 0, self.var.canals)

                    #When there are no set canals, the entire command area expereinces leakage
                    self.var.canals = np.where(npareamaximum(self.var.canals, self.var.reservoir_command_areas) == 0,
                                               self.var.reservoir_command_areas, self.var.canals)

                    self.var.canalsArea = np.where(self.var.canals>0, npareatotal(self.var.cellArea, self.var.canals), 0)
                    self.var.canalsAreaC = np.compress(self.var.compress_LR, self.var.canalsArea)

            self.var.swAbstractionFraction_Lift_Domestic = globals.inZero.copy()
            self.var.swAbstractionFraction_Lift_Livestock = globals.inZero.copy()
            self.var.swAbstractionFraction_Lift_Industry = globals.inZero.copy()
            self.var.swAbstractionFraction_Lift_Irrigation = globals.inZero.copy()

            self.var.using_lift_areas = False
            if 'using_lift_areas' in option:
                if checkOption('using_lift_areas'):

                    self.var.using_lift_areas = True
                    self.var.lift_command_areas = loadmap('lift_areas').astype(int)

                    if self.var.sectorSourceAbstractionFractions:
                        self.var.swAbstractionFraction_Lift_Domestic = loadmap(
                            'swAbstractionFraction_Lift_Domestic')
                        self.var.swAbstractionFraction_Lift_Livestock = loadmap(
                            'swAbstractionFraction_Lift_Livestock')
                        self.var.swAbstractionFraction_Lift_Industry = loadmap(
                            'swAbstractionFraction_Lift_Industry')
                        self.var.swAbstractionFraction_Lift_Irrigation = loadmap(
                            'swAbstractionFraction_Lift_Irrigation')

            # -------------------------------------------
            # partitioningGroundSurfaceAbstraction
            # partitioning abstraction sources: groundwater and surface water
            # partitioning based on local average baseflow (m3/s) and upstream average discharge (m3/s)
            # estimates of fractions of groundwater and surface water abstractions
            swAbstractionFraction = loadmap('swAbstractionFrac')

            if swAbstractionFraction < 0:

                averageBaseflowInput = loadmap('averageBaseflow')
                averageDischargeInput = loadmap('averageDischarge')
                # convert baseflow from m to m3/s
                if returnBool('baseflowInM'):
                    averageBaseflowInput = averageBaseflowInput * self.var.cellArea * self.var.InvDtSec

                if checkOption('usingAllocSegments'):
                    averageBaseflowInput = np.where(self.var.allocSegments > 0,
                                                    npareaaverage(averageBaseflowInput, self.var.allocSegments),
                                                    averageBaseflowInput)

                    # averageUpstreamInput = np.where(self.var.allocSegments > 0,
                    #                                npareamaximum(averageDischargeInput, self.var.allocSegments),
                    #                                averageDischargeInput)

                swAbstractionFraction = np.maximum(0.0, np.minimum(1.0, averageDischargeInput / np.maximum(1e-20, averageDischargeInput + averageBaseflowInput)))
                swAbstractionFraction = np.minimum(1.0, np.maximum(0.0, swAbstractionFraction))

            self.var.swAbstractionFraction = globals.inZero.copy()
            for No in range(4):
                self.var.swAbstractionFraction += self.var.fracVegCover[No] * swAbstractionFraction
            for No in range(4, 6):
                if self.var.modflow:
                    self.var.swAbstractionFraction += self.var.fracVegCover[No] * swAbstractionFraction
                else:
                    # The motivation is to avoid groundwater on sealed and water land classes
                    # TODO: Groundwater pumping should be allowed over sealed land
                    self.var.swAbstractionFraction += self.var.fracVegCover[No]

            # non-irrigation input maps have for each month or year the unit m/day (True) or million m3/month (False)
            self.var.demand_unit = True
            if "demand_unit" in binding:
                self.var.demand_unit = returnBool('demand_unit')

            # allocation zone
            # regular grid inside the 2d array
            # inner grid size
            inner = 1
            if "allocation_area" in binding:
                inner = int(loadmap('allocation_area'))

            latldd, lonldd, cell, invcellldd, rows, cols = readCoord(cbinding('Ldd'))
            filename = os.path.splitext(cbinding('Ldd'))[0] + '.nc'
            if os.path.isfile(filename):
                cut0, cut1, cut2, cut3 = mapattrNetCDF(filename, check=False)
            else:
                filename = os.path.splitext(cbinding('Ldd'))[0] + '.tif'
                if not(os.path.isfile(filename)):
                    filename = os.path.splitext(cbinding('Ldd'))[0] + '.map'
                nf2 = gdal.Open(filename, gdalconst.GA_ReadOnly)
                cut0, cut1, cut2, cut3 = mapattrTiff(nf2)

            arr = np.kron(np.arange(rows // inner * cols // inner).reshape((rows // inner, cols // inner)), np.ones((inner, inner)))
            arr = arr[cut2:cut3, cut0:cut1].astype(int)
            self.var.allocation_zone = compressArray(arr)

            self.var.modflowPumping = globals.inZero.copy()
            self.var.leakage = globals.inZero.copy()
            self.var.pumping = globals.inZero.copy()
            self.var.modfPumpingM = globals.inZero.copy()
            self.var.Pumping_daily = globals.inZero.copy()
            self.var.modflowDepth2 = 0
            self.var.modflowTopography = 0
            self.var.allowedPumping = globals.inZero.copy()
            self.var.leakageCanals_M = globals.inZero.copy()

            self.var.act_nonIrrWithdrawal = globals.inZero.copy()
            self.var.ratio_irrWithdrawalGW_month = globals.inZero.copy()
            self.var.ratio_irrWithdrawalSW_month = globals.inZero.copy()
            self.var.act_irrWithdrawalSW_month = globals.inZero.copy()
            self.var.act_irrWithdrawalGW_month = globals.inZero.copy()

            self.var.Channel_Domestic = globals.inZero.copy()
            self.var.Channel_Industry = globals.inZero.copy()
            self.var.Channel_Livestock = globals.inZero.copy()
            self.var.Channel_Irrigation = globals.inZero.copy()

            self.var.Lake_Domestic = globals.inZero.copy()
            self.var.Lake_Industry = globals.inZero.copy()
            self.var.Lake_Livestock = globals.inZero.copy()
            self.var.Lake_Irrigation = globals.inZero.copy()

            self.var.Res_Domestic = globals.inZero.copy()
            self.var.Res_Industry = globals.inZero.copy()
            self.var.Res_Livestock = globals.inZero.copy()
            self.var.Res_Irrigation = globals.inZero.copy()

            self.var.GW_Domestic = globals.inZero.copy()
            self.var.GW_Industry = globals.inZero.copy()
            self.var.GW_Livestock = globals.inZero.copy()
            self.var.GW_Irrigation = globals.inZero.copy()
            self.var.abstractedLakeReservoirM3 = globals.inZero.copy()

            self.var.swAbstractionFraction_domestic = 1 + globals.inZero.copy()
            self.var.ind_efficiency = 1.
            self.var.dom_efficiency = 1.
            self.var.liv_efficiency = 1


        else:  # no water demand
            self.var.nonIrrReturnFlowFraction = globals.inZero.copy()
            self.var.nonFossilGroundwaterAbs = globals.inZero.copy()
            self.var.nonIrruse = globals.inZero.copy()
            self.var.act_indDemand = globals.inZero.copy()
            self.var.act_domDemand = globals.inZero.copy()
            self.var.act_livDemand = globals.inZero.copy()
            self.var.nonIrrDemand = globals.inZero.copy()
            self.var.totalIrrDemand = globals.inZero.copy()
            self.var.totalWaterDemand = globals.inZero.copy()
            self.var.act_irrWithdrawal = globals.inZero.copy()
            self.var.act_nonIrrWithdrawal = globals.inZero.copy()
            self.var.act_totalWaterWithdrawal = globals.inZero.copy()
            self.var.act_indConsumption = globals.inZero.copy()
            self.var.act_domConsumption = globals.inZero.copy()
            self.var.act_livConsumption = globals.inZero.copy()
            self.var.act_nonIrrConsumption = globals.inZero.copy()
            self.var.act_totalIrrConsumption = globals.inZero.copy()
            self.var.act_totalWaterConsumption = globals.inZero.copy()
            self.var.unmetDemand = globals.inZero.copy()
            self.var.addtoevapotrans = globals.inZero.copy()
            self.var.returnflowIrr = globals.inZero.copy()
            self.var.returnFlow = globals.inZero.copy()
            self.var.unmetDemandPaddy = globals.inZero.copy()
            self.var.unmetDemandNonpaddy = globals.inZero.copy()
            self.var.ind_efficiency = 1.
            self.var.dom_efficiency = 1.
            self.var.liv_efficiency = 1
            self.var.modflowPumping = 0
            self.var.modflowDepth2 = 0
            self.var.modflowTopography = 0
            self.var.act_bigLakeResAbst = globals.inZero.copy()

            self.var.leakage = globals.inZero.copy()
            self.var.pumping = globals.inZero.copy()
            self.var.unmet_lost = globals.inZero.copy()
            self.var.pot_GroundwaterAbstract = globals.inZero.copy()

            self.var.WB_elec = globals.inZero.copy()
            self.var.relaxSWagent = globals.inZero.copy()
            self.var.relaxGWagent = globals.inZero.copy()

            self.var.Channel_Domestic = globals.inZero.copy()
            self.var.Channel_Industry = globals.inZero.copy()
            self.var.Channel_Livestock = globals.inZero.copy()
            self.var.Channel_Irrigation = globals.inZero.copy()

            self.var.Lake_Domestic = globals.inZero.copy()
            self.var.Lake_Industry = globals.inZero.copy()
            self.var.Lake_Livestock = globals.inZero.copy()
            self.var.Lake_Irrigation = globals.inZero.copy()

            self.var.Res_Domestic = globals.inZero.copy()
            self.var.Res_Industry = globals.inZero.copy()
            self.var.Res_Livestock = globals.inZero.copy()
            self.var.Res_Irrigation = globals.inZero.copy()

            self.var.GW_Domestic = globals.inZero.copy()
            self.var.GW_Industry = globals.inZero.copy()
            self.var.GW_Livestock = globals.inZero.copy()
            self.var.GW_Irrigation = globals.inZero.copy()

            self.var.abstractedLakeReservoirM3 = globals.inZero.copy()
            self.var.swAbstractionFraction_domestic = 1 + globals.inZero.copy()


    def dynamic(self):
        """
        Dynamic part of the water demand module

        * calculate the fraction of water from surface water vs. groundwater
        * get non-Irrigation water demand and its return flow fraction
        """

        if self.var.modflow:

            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.environmental_need.dynamic()
            else:
                self.var.envFlow = 0.00001  # 0.01mm

            # computing leakage from rivers to groundwater

            # self.var.readAvlChannelStorageM will be used in land covertype to compute leakage to groundwater if ModFlow coupling
            # to avoid small values and to avoid surface water abstractions from dry channels (>= 0.01mm)
            self.var.readAvlChannelStorageM = np.where(self.var.channelStorage < (0.0005 * self.var.cellArea), 0.,
                                                       self.var.channelStorage)  # in [m3]
            # coversersion m3 -> m # minus environmental flow
            self.var.readAvlChannelStorageM = self.var.readAvlChannelStorageM * self.var.M3toM  # in [m]
            self.var.readAvlChannelStorageM = np.maximum(0., self.var.readAvlChannelStorageM - self.var.envFlow)


        if checkOption('includeWaterDemand'):

            # for debugging of a specific date
            #if (globals.dateVar['curr'] >= 137):
            #    ii =1

            # ----------------------------------------------------
            # WATER DEMAND

            # Fix year of water demand on predefined year
            wd_date = globals.dateVar['currDate']
            if self.var.waterdemandFixed:
                wd_date = wd_date.replace(day = 1)
                wd_date = wd_date.replace(year = self.var.waterdemandFixedYear)

            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.domestic.dynamic(wd_date)
                self.industry.dynamic(wd_date)
                self.livestock.dynamic(wd_date)
                self.irrigation.dynamic()
                self.environmental_need.dynamic()
            else:
                self.irrigation.dynamic()
            #if not self.var.modflow:
                self.environmental_need.dynamic()

            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                if globals.dateVar['newStart'] or globals.dateVar['newMonth'] or 'basin_transfers_daily_operations' in option or 'reservoir_transfers' in option:
                    # total (potential) non irrigation water demand
                    self.var.nonIrrDemand = self.var.domesticDemand + self.var.industryDemand + self.var.livestockDemand
                    self.var.pot_nonIrrConsumption = np.minimum(self.var.nonIrrDemand, self.var.pot_domesticConsumption +
                                                                self.var.pot_industryConsumption + self.var.pot_livestockConsumption)
                    # fraction of return flow from domestic and industrial water demand
                    self.var.nonIrrReturnFlowFraction = divideValues((self.var.nonIrrDemand - self.var.pot_nonIrrConsumption), self.var.nonIrrDemand)

                # non-irrg fracs in nonIrrDemand
                frac_industry = divideValues(self.var.industryDemand, self.var.nonIrrDemand)
                frac_domestic = divideValues(self.var.domesticDemand, self.var.nonIrrDemand)
                frac_livestock = divideValues(self.var.livestockDemand, self.var.nonIrrDemand)

                # Sum up water demand
                # totalDemand [m]: total maximum (potential) water demand: irrigation and non irrigation
                totalDemand = self.var.nonIrrDemand + self.var.totalIrrDemand  # in [m]
            else:  # only irrigation is considered
                totalDemand = np.copy(self.var.totalIrrDemand)  # in [m]
                self.var.domesticDemand = globals.inZero.copy()
                self.var.industryDemand = globals.inZero.copy()
                self.var.livestockDemand = globals.inZero.copy()
                self.var.nonIrrDemand = globals.inZero.copy()
                self.var.pot_nonIrrConsumption = globals.inZero.copy()
                self.var.nonIrrReturnFlowFraction = globals.inZero.copy()
                frac_industry = globals.inZero.copy()
                frac_domestic = globals.inZero.copy()
                frac_livestock = globals.inZero.copy()
                #print('-----------------------------totalDemand---------: ', np.mean(totalDemand))

            # ----------------------------------------------------
            # WATER AVAILABILITY

            if not self.var.modflow:  # already done if ModFlow coupling
                # conversion m3 -> m # minus environmental flow
                self.var.readAvlChannelStorageM = np.maximum(0.,self.var.channelStorage * self.var.M3toM - self.var.envFlow) # in [m]


            #-------------------------------------
            # WATER DEMAND vs. WATER AVAILABILITY
            #-------------------------------------

            if dateVar['newStart'] or dateVar['newMonth']:
                self.var.act_irrWithdrawalSW_month = globals.inZero.copy()
                self.var.act_irrWithdrawalGW_month = globals.inZero.copy()

                if 'irrigation_agent_SW_request_month_m3' in binding and self.var.activate_irrigation_agents:

                    # These are read at the beginning of each month as they are updated by several relax functions
                    # and turned off once satisfying request
                    if self.var.sectorSourceAbstractionFractions:
                        self.var.swAbstractionFraction_Channel_Irrigation = loadmap(
                            'swAbstractionFraction_Channel_Irrigation')
                        if self.var.using_lift_areas:
                            self.var.swAbstractionFraction_Lift_Irrigation = loadmap(
                                'swAbstractionFraction_Lift_Irrigation')
                        self.var.swAbstractionFraction_Lake_Irrigation = loadmap(
                            'swAbstractionFraction_Lake_Irrigation')
                        self.var.swAbstractionFraction_Res_Irrigation = loadmap(
                            'swAbstractionFraction_Res_Irrigation')
                    else:
                        self.var.swAbstractionFraction_Channel_Irrigation = 1 + globals.inZero.copy()
                        if self.var.using_lift_areas:
                            self.var.swAbstractionFraction_Lift_Irrigation = 1 + globals.inZero.copy()
                        self.var.swAbstractionFraction_Lake_Irrigation = 1 + globals.inZero.copy()
                        self.var.swAbstractionFraction_Res_Irrigation = 1 + globals.inZero.copy()

                    if self.var.relax_irrigation_agents:
                        self.var.swAbstractionFraction_Channel_Irrigation = np.where(self.var.relaxSWagent > 0,
                                                                                     self.var.relax_abstraction_fraction_initial / self.var.relaxSWagent,
                                                                                     self.var.swAbstractionFraction_Channel_Irrigation)
                        if self.var.using_lift_areas:
                            self.var.swAbstractionFraction_Lift_Irrigation = np.where(self.var.relaxSWagent > 0,
                                                                                      self.var.relax_abstraction_fraction_initial / self.var.relaxSWagent,
                                                                                      self.var.swAbstractionFraction_Lift_Irrigation)
                        self.var.swAbstractionFraction_Lake_Irrigation = np.where(self.var.relaxSWagent > 0,
                                                                                  self.var.relax_abstraction_fraction_initial / self.var.relaxSWagent,
                                                                                     self.var.swAbstractionFraction_Lake_Irrigation)
                        self.var.swAbstractionFraction_Res_Irrigation = np.where(self.var.relaxSWagent > 0,
                                                                                 self.var.relax_abstraction_fraction_initial / self.var.relaxSWagent,
                                                                                     self.var.swAbstractionFraction_Res_Irrigation)

                if 'irrigation_agent_GW_request_month_m3' in binding and self.var.activate_irrigation_agents:

                    if self.var.sectorSourceAbstractionFractions:
                        self.var.gwAbstractionFraction_Irrigation = loadmap(
                            'gwAbstractionFraction_Irrigation')
                    else:
                        self.var.gwAbstractionFraction_Irrigation = 1 + globals.inZero.copy()

                    if self.var.relax_irrigation_agents:
                        self.var.gwAbstractionFraction_Irrigation = np.where(self.var.relaxGWagent > 0,
                                                                      self.var.relax_abstraction_fraction_initial / self.var.relaxGWagent,
                                                                      self.var.gwAbstractionFraction_Irrigation)

                if 'commandAreasRelaxGwAbstraction' in binding:
                    self.var.gwAbstractionFraction_Irrigation = np.where(self.var.reservoir_command_areas > 0,
                                                                         0.01,
                                                                         self.var.gwAbstractionFraction_Irrigation)

            if self.var.sectorSourceAbstractionFractions and 'commandAreasRelaxGwAbstraction' in binding and \
                    self.var.using_reservoir_command_areas:

                    if dateVar['currDate'].day > 15:
                        self.var.gwAbstractionFraction_Irrigation = np.where(self.var.reservoir_command_areas > 0,
                                                                             loadmap('commandAreasRelaxGwAbstraction'),
                                                                             self.var.gwAbstractionFraction_Irrigation)

            # surface water abstraction that can be extracted to fulfill totalDemand
            # - based on ChannelStorage and swAbstractionFraction * totalDemand
            # sum up potential surface water abstraction (no groundwater abstraction under water and sealed area)

            if self.var.sectorSourceAbstractionFractions:

                pot_Channel_Domestic = self.var.swAbstractionFraction_Channel_Domestic * self.var.domesticDemand
                pot_Channel_Livestock = self.var.swAbstractionFraction_Channel_Livestock * self.var.livestockDemand
                pot_Channel_Industry = self.var.swAbstractionFraction_Channel_Industry * self.var.industryDemand
                pot_Channel_Irrigation = self.var.swAbstractionFraction_Channel_Irrigation * self.var.totalIrrDemand

                pot_channelAbst = pot_Channel_Domestic + pot_Channel_Livestock + pot_Channel_Industry + pot_Channel_Irrigation

                self.var.act_SurfaceWaterAbstract = np.minimum(self.var.readAvlChannelStorageM, pot_channelAbst)
            else:
                pot_SurfaceAbstract = totalDemand * self.var.swAbstractionFraction
                # only local surface water abstraction is allowed (network is only within a cell)
                self.var.act_SurfaceWaterAbstract = np.minimum(self.var.readAvlChannelStorageM, pot_SurfaceAbstract)

            self.var.readAvlChannelStorageM -= self.var.act_SurfaceWaterAbstract
            self.var.act_channelAbst = self.var.act_SurfaceWaterAbstract.copy()
                # if surface water is not sufficient it is taken from groundwater

            if self.var.sectorSourceAbstractionFractions:
                self.var.Channel_Domestic = np.minimum(self.var.act_channelAbst,
                                                       self.var.swAbstractionFraction_Channel_Domestic * self.var.domesticDemand)
                self.var.Channel_Livestock = np.minimum(self.var.act_channelAbst - self.var.Channel_Domestic,
                                                        self.var.swAbstractionFraction_Channel_Livestock * self.var.livestockDemand)
                self.var.Channel_Industry = np.minimum(
                    self.var.act_channelAbst - self.var.Channel_Domestic - self.var.Channel_Livestock,
                    self.var.swAbstractionFraction_Channel_Industry * self.var.industryDemand)
                self.var.Channel_Irrigation = np.minimum(
                    self.var.act_channelAbst - self.var.Channel_Domestic - self.var.Channel_Livestock - self.var.Channel_Industry,
                    self.var.swAbstractionFraction_Channel_Irrigation * self.var.totalIrrDemand)

            # UNDER CONSTRUCTION
            if self.var.using_lift_areas:

                # Lift development
                # When there is sufficient water in the Segment to fulfill demand, the water is taken away proportionally from each cell's readAvlChannelStorageM in the Segment.
                # For example, if total demand can be filled with 50% of total availability, then 50% of the readAvlChannelStorageM from each cell is used.
                # Note that if a cell has too little Channel Storage, then no water will be taken from the cell as this was dealt with earlier:  readAvlChannelStorage = 0 if < (0.0005 * self.var.cellArea)
                # Note: Due to the shared use of abstracted channel storage, a cell may abstract more than its pot_SurfaceAbstract, as well as not necessarily satisfy its pot_SurfaceAbstract

                pot_liftAbst = self.var.swAbstractionFraction_Lift_Irrigation * self.var.totalIrrDemand + \
                                  self.var.swAbstractionFraction_Lift_Domestic * self.var.domesticDemand+ \
                                  self.var.swAbstractionFraction_Lift_Industry * self.var.industryDemand + \
                                  self.var.swAbstractionFraction_Lift_Livestock * self.var.livestockDemand

                remainNeed_afterLocal = np.maximum(0, np.minimum(pot_SurfaceAbstract_Lift,
                                                                 self.var.totalIrrDemand * self.var.swAbstractionFraction_Irr - self.var.act_SurfaceWaterAbstract_Irr))
                # The remaining demand within each command area [M3] is put into a map where each cell in the command area holds this total demand
                demand_Segment_lift = np.where(self.var.lift_command_areas > 0,
                                               npareatotal(remainNeed_afterLocal * self.var.cellArea,
                                                           self.var.lift_command_areas),
                                               0)  # [M3]

                available_Segment_lift = np.where(self.var.lift_command_areas > 0,
                                                  npareatotal(self.var.readAvlChannelStorageM * self.var.cellArea,
                                                              self.var.lift_command_areas),
                                                  0)  # [M3]

                frac_used_Segment_lift = np.where(available_Segment_lift > 0,
                                                  np.minimum(demand_Segment_lift / available_Segment_lift, 1.), 0.)

                self.var.act_channelAbstract += (frac_used_Segment_lift * self.var.readAvlChannelStorageM)

                metRemainSegment_lift = np.where(demand_Segment_lift > 0,
                                                 divideValues(frac_used_Segment_lift * available_Segment_lift,
                                                              demand_Segment_lift), 0)
                self.var.metRemainSegment_lift = metRemainSegment_lift.copy()
                self.var.act_SurfaceWaterAbstract_Irr += (metRemainSegment_lift * remainNeed_afterLocal)
                self.var.act_SurfaceWaterAbstract += (metRemainSegment_lift * remainNeed_afterLocal)
                # print('satisfied need minus abstracted', np.sum(self.var.cellArea * metRemainSegment_lift * remainNeed_afterLocal) - np.sum(
                #    self.var.cellArea * frac_used_Segment_lift * self.var.readAvlChannelStorageM))

                self.var.readAvlChannelStorageM -= (frac_used_Segment_lift * self.var.readAvlChannelStorageM)
                self.var.readAvlChannelStorageM = np.where(self.var.readAvlChannelStorageM < 0.02, 0,
                                                           self.var.readAvlChannelStorageM)
                # Used in landCover for riverbedExchange

                self.var.act_channelAbstract_Lift = frac_used_Segment_lift * self.var.readAvlChannelStorageM


            if checkOption('includeWaterBodies'):

                if self.var.sectorSourceAbstractionFractions:

                    pot_Lake_Domestic = np.minimum(
                        self.var.swAbstractionFraction_Lake_Domestic * self.var.domesticDemand,
                        self.var.domesticDemand * self.var.swAbstractionFraction_domestic.copy() - self.var.Channel_Domestic)

                    pot_Lake_Livestock = np.minimum(
                        self.var.swAbstractionFraction_Lake_Livestock * self.var.livestockDemand,
                        self.var.livestockDemand - self.var.Channel_Livestock)

                    pot_Lake_Industry = np.minimum(
                        self.var.swAbstractionFraction_Lake_Industry * self.var.industryDemand,
                        self.var.industryDemand - self.var.Channel_Industry)

                    pot_Lake_Irrigation = np.minimum(
                        self.var.swAbstractionFraction_Lake_Irrigation * self.var.totalIrrDemand,
                        self.var.totalIrrDemand - self.var.Channel_Irrigation)

                    remainNeed = pot_Lake_Domestic + pot_Lake_Livestock + pot_Lake_Industry + pot_Lake_Irrigation

                else:

                    # water that is still needed from surface water
                    remainNeed = np.maximum(pot_SurfaceAbstract - self.var.act_SurfaceWaterAbstract, 0)

                # first from big Lakes and reservoirs, big lakes cover several gridcells
                # collect all water demand from lake pixels of the same id

                #remainNeedBig = npareatotal(remainNeed, self.var.waterBodyID)
                # not only the lakes and reservoirs but the command areas around water bodies e.g. here a buffer
                remainNeedBig = npareatotal(remainNeed, self.var.waterBodyBuffer)
                remainNeedBigC = np.compress(self.var.compress_LR, remainNeedBig)

                # Storage of a big lake
                lakeResStorageC = np.where(self.var.waterBodyTypCTemp == 0, 0.,
                            np.where(self.var.waterBodyTypCTemp == 1, self.var.lakeStorageC, self.var.reservoirStorageM3C)) / self.var.MtoM3C

                minlake = np.maximum(0., 0.98*lakeResStorageC) #reasonable but arbitrary limit
                act_bigLakeAbstC = np.minimum(minlake , remainNeedBigC)
                # substract from both, because it is sorted by self.var.waterBodyTypCTemp
                self.var.lakeStorageC = self.var.lakeStorageC - act_bigLakeAbstC * self.var.MtoM3C
                self.var.lakeVolumeM3C = self.var.lakeVolumeM3C - act_bigLakeAbstC * self.var.MtoM3C
                self.var.reservoirStorageM3C = self.var.reservoirStorageM3C - act_bigLakeAbstC * self.var.MtoM3C
                # and from the combined one for waterbalance issues
                self.var.lakeResStorageC = self.var.lakeResStorageC - act_bigLakeAbstC * self.var.MtoM3C

                self.var.abstractedLakeReservoirM3C = act_bigLakeAbstC.copy() * self.var.MtoM3C


                self.var.lakeResStorage = globals.inZero.copy()
                np.put(self.var.lakeResStorage, self.var.decompress_LR, self.var.lakeResStorageC)
                bigLakesFactorC = divideValues(act_bigLakeAbstC, remainNeedBigC)


                # and back to the big array
                bigLakesFactor = globals.inZero.copy()
                np.put(bigLakesFactor, self.var.decompress_LR, bigLakesFactorC)

                #bigLakesFactorAllaroundlake = npareamaximum(bigLakesFactor, self.var.waterBodyID)
                bigLakesFactorAllaroundlake = npareamaximum(bigLakesFactor, self.var.waterBodyBuffer)

                # abstraction from big lakes is partioned to the users around the lake
                self.var.act_bigLakeResAbst = remainNeed * bigLakesFactorAllaroundlake

                # remaining need is used from small lakes
                remainNeed1 = remainNeed * (1 - bigLakesFactorAllaroundlake)
                #minlake = np.maximum(0.,self.var.smalllakeStorage - self.var.minsmalllakeStorage) * self.var.M3toM

                if returnBool('useSmallLakes'):
                    minlake = np.maximum(0.,0.98 * self.var.smalllakeStorage) * self.var.M3toM
                    self.var.act_smallLakeResAbst = np.minimum(minlake, remainNeed1)
                    #self.var.actLakeResAbst = np.minimum(0.5 * self.var.smalllakeStorageM3 * self.var.M3toM, remainNeed)
                    # act_smallLakesres is substracted from small lakes storage
                    self.var.smalllakeVolumeM3 = self.var.smalllakeVolumeM3 - self.var.act_smallLakeResAbst * self.var.MtoM3
                    self.var.smalllakeStorage = self.var.smalllakeStorage - self.var.act_smallLakeResAbst * self.var.MtoM3
                else:
                    self.var.act_smallLakeResAbst = 0

                # available surface water is from river network + large/small lake & reservoirs
                self.var.act_SurfaceWaterAbstract = self.var.act_SurfaceWaterAbstract + self.var.act_bigLakeResAbst + self.var.act_smallLakeResAbst
                self.var.act_lakeAbst = self.var.act_bigLakeResAbst + self.var.act_smallLakeResAbst


                # 📤 Transfer water between reservoirs
                # Send storage between reservoirs using the Excel sheet reservoir_transfers within cwatm_settings.xlsx
                # Using the waterBodyIDs defined in the settings, designate
                # the Giver, the Receiver, and the daily fraction of live storage the Giver sends to the Receiver.
                # If the Receiver is already at capacity, the Giver does not send any storage.
                # Reservoirs can only send to one reservoir. Reservoirs can receive from several reservoirs.

                if 'reservoir_transfers' in option:
                    if checkOption('reservoir_transfers'):

                        for transfer in self.var.reservoir_transfers:

                            self.var.inZero_C = np.compress(self.var.compress_LR, globals.inZero.copy())

                            if returnBool('dynamicLakesRes'):
                                year = dateVar['currDate'].year
                            else:
                                year = loadmap('fixLakesResYear')

                            if transfer[0] > 0:
                                index_giver = np.where(self.var.waterBodyID_C == transfer[0])[0][0]
                                giver_already_constructed = self.var.resYearC[index_giver] <= year
                            else:
                                giver_already_constructed = True

                            if transfer[1] > 0:
                                index_receiver = np.where(self.var.waterBodyID_C == transfer[1])[0][0]
                                receiver_already_constructed = self.var.resYearC[index_receiver] <= year
                            else:
                                receiver_already_constructed = True


                            if receiver_already_constructed and giver_already_constructed:

                                reservoir_unused = self.var.resVolumeC - self.var.reservoirStorageM3C
                                if transfer[1] > 0:
                                    reservoir_unused_receiver = reservoir_unused[index_receiver]
                                else:
                                    reservoir_unused_receiver = 10e12

                                if transfer[0] == 0:
                                    # In this case, the fraction refers to the fraction of the receiver,
                                    # as the giver is infinite
                                    reservoir_storage_giver = self.var.resVolumeC[index_receiver]
                                else:
                                    reservoir_storage_giver = self.var.reservoirStorageM3C[index_giver]

                                reservoir_transfer_actual = np.minimum(reservoir_unused_receiver * 0.95,
                                                                       #reservoir_storage_giver * transfer[2])
                                                                        np.where(transfer[2] <= 1,
                                                                        reservoir_storage_giver * transfer[2],
                                                                        np.minimum(transfer[2],reservoir_storage_giver)))

                                #print(transfer[0], 'donated', reservoir_transfer_actual, 'm3 to', transfer[1])

                                if transfer[0] > 0: # There is a giver, not the ocean
                                    self.var.inZero_C[index_giver] = -reservoir_transfer_actual    # giver
                                    self.var.reservoir_transfers_out_M3C[index_giver] += reservoir_transfer_actual
                                else:
                                    self.var.reservoir_transfers_from_outside_M3C[index_receiver] += reservoir_transfer_actual

                                if transfer[1] > 0: # There is a receiver, not the ocean
                                    self.var.inZero_C[index_receiver] = reservoir_transfer_actual  # receiver
                                    self.var.reservoir_transfers_in_M3C[index_receiver] += reservoir_transfer_actual
                                else:
                                    self.var.reservoir_transfers_to_outside_M3C[index_giver] += reservoir_transfer_actual

                                self.var.lakeStorageC += self.var.inZero_C
                                self.var.lakeVolumeM3C += self.var.inZero_C
                                self.var.lakeResStorageC += self.var.inZero_C
                                self.var.reservoirStorageM3C += self.var.inZero_C

                                self.var.reservoir_transfers_net_M3C += self.var.inZero_C
                                # Cancels out positive and negative if both receiving and giving

                                if transfer[1] == 0:
                                    to_outside_basin = globals.inZero.copy()
                                    np.put(to_outside_basin, self.var.decompress_LR, self.var.inZero_C)

                                    pot_Lake_Industry -= to_outside_basin * self.var.M3toM
                                    # self.var.Lake_Industry is updated below
                                    self.var.act_lakeAbst -= to_outside_basin * self.var.M3toM

                                    self.var.act_SurfaceWaterAbstract -= to_outside_basin * self.var.M3toM
                                    self.var.act_bigLakeResAbst -= to_outside_basin * self.var.M3toM

                                    self.var.industryDemand -= to_outside_basin * self.var.M3toM
                                    self.var.pot_industryConsumption -= to_outside_basin * self.var.M3toM
                                    self.var.ind_efficiency = divideValues(self.var.pot_industryConsumption,
                                                                           self.var.industryDemand)

                        np.put(self.var.reservoir_transfers_net_M3, self.var.decompress_LR,
                               self.var.reservoir_transfers_net_M3C)
                        self.var.reservoir_transfers_net_M3C = np.compress(self.var.compress_LR,
                                                                       globals.inZero.copy())

                        np.put(self.var.reservoir_transfers_in_M3, self.var.decompress_LR,
                               self.var.reservoir_transfers_in_M3C)
                        self.var.reservoir_transfers_in_M3C = np.compress(self.var.compress_LR,
                                                                           globals.inZero.copy())

                        np.put(self.var.reservoir_transfers_out_M3, self.var.decompress_LR,
                               self.var.reservoir_transfers_out_M3C)
                        self.var.reservoir_transfers_out_M3C = np.compress(self.var.compress_LR,
                                                                          globals.inZero.copy())

                        np.put(self.var.reservoir_transfers_from_outside_M3, self.var.decompress_LR,
                               self.var.reservoir_transfers_from_outside_M3C)
                        self.var.reservoir_transfers_from_outside_M3C = np.compress(self.var.compress_LR,
                                                                           globals.inZero.copy())

                        np.put(self.var.reservoir_transfers_to_outside_M3, self.var.decompress_LR,
                               self.var.reservoir_transfers_to_outside_M3C)
                        self.var.reservoir_transfers_to_outside_M3C = np.compress(self.var.compress_LR,
                                                                           globals.inZero.copy())

                        ###

                        if self.var.sectorSourceAbstractionFractions:

                            #np.put(self.var.reservoir_transfers_out_M3, self.var.decompress_LR,
                            #       self.var.reservoir_transfers_out_M3C)

                            self.var.swAbstractionFraction_Res_Industry = np.where(self.var.reservoir_transfers_to_outside_M3 > 0, 0,
                                                                                   self.var.swAbstractionFraction_Res_Industry)
                            self.var.gwAbstractionFraction_Industry = np.where(self.var.reservoir_transfers_to_outside_M3 > 0, 0,
                                                                               self.var.gwAbstractionFraction_Industry)
                        else:
                            pot_SurfaceAbstract -= to_outside_basin
                            #to avoid groundwater abstraction
                            self.var.swAbstractionFraction = np.where(self.var.reservoir_transfers_to_outside_M3 != 0, 1,
                                                                      self.var.swAbstractionFraction_nonIrr)

                # 📤 -------------------------------------

                if self.var.sectorSourceAbstractionFractions:

                    #A
                    self.var.Lake_Domestic = np.minimum(self.var.act_lakeAbst, pot_Lake_Domestic)
                    self.var.Lake_Livestock = np.minimum(self.var.act_lakeAbst - self.var.Lake_Domestic,
                                                            pot_Lake_Livestock)
                    self.var.Lake_Industry = np.minimum(
                        self.var.act_lakeAbst - self.var.Lake_Domestic - self.var.Lake_Livestock,
                        pot_Lake_Industry)
                    self.var.Lake_Irrigation = np.minimum(
                        self.var.act_lakeAbst - self.var.Lake_Domestic - self.var.Lake_Livestock - self.var.Lake_Industry,
                        pot_Lake_Irrigation)

                    #B

                    pot_Res_Domestic = np.minimum(
                        self.var.swAbstractionFraction_Res_Domestic * self.var.domesticDemand,
                        self.var.domesticDemand * self.var.swAbstractionFraction_domestic.copy() - self.var.Channel_Domestic - self.var.Lake_Domestic)

                    pot_Res_Livestock = np.minimum(
                        self.var.swAbstractionFraction_Res_Livestock * self.var.livestockDemand,
                        self.var.livestockDemand - self.var.Channel_Livestock - self.var.Lake_Livestock)

                    pot_Res_Industry = np.minimum(
                        self.var.swAbstractionFraction_Res_Industry * self.var.industryDemand,
                        self.var.industryDemand - self.var.Channel_Industry - self.var.Lake_Industry)

                    pot_Res_Irrigation = np.minimum(
                        self.var.swAbstractionFraction_Res_Irrigation * self.var.totalIrrDemand,
                        self.var.totalIrrDemand - self.var.Channel_Irrigation - self.var.Lake_Irrigation)

                    #remainNeed2 = pot_Res_Domestic + pot_Res_Livestock + pot_Res_Industry + pot_Res_Irrigation
                    remainNeed2 = pot_Res_Irrigation

                else:

                    remainNeed2 = pot_SurfaceAbstract - self.var.act_SurfaceWaterAbstract

                if self.var.using_reservoir_command_areas:

                    # The command area of a reservoir is the area that can receive water from this reservoir.
                    # Before this step, each cell has attempted to satisfy its demands with local water using in-cell channel, lift area, and lake.
                    # The remaining demand within each command area is totaled and requested from the associated reservoir.
                    # The reservoir offers this water up to a daily maximum relating to the available storage in the reservoir, defined in the Reservoir_releases_input_file.
                    #
                    # SETTINGS FILE AND INPUTS

                    # -Activating
                    # In the OPTIONS section towards the beginning of the settings file, add/set
                    # using_reservoir_command_areas = True

                    # - Command areas raster map
                    # Anywhere after the OPTIONS section (in WATERDEMAND, for example), add/set reservoir_command_areas to a path holding...
                    # information about the command areas. This Command areas raster map should assign the same positive integer coding to each cell within the same segment.
                    # All other cells must Nan values, or values <= 0.

                    # -Optional inputs
                    #
                    # Anywhere after the OPTIONS section, add/set Reservoir_releases_input_file to a path holding information about irrigation releases.
                    # This should be a raster map (netCDF) of 366 values determining the maximum fraction of available storage to be used for meeting water demand...
                    # in the associated command area on the day of the year. If this is not included, a value of 0.01 will be assumed,
                    # i.e. 1% of the reservoir storage can be at most released into the command area on each day.
                    self.var.act_ResAbst = globals.inZero.copy()
                    if self.var.sectorSourceAbstractionFractions:

                        remainNeedPre = pot_Res_Domestic + pot_Res_Livestock + pot_Res_Industry

                        demand_Segment = np.where(self.var.reservoir_command_areas > 0,
                                                  npareatotal(remainNeedPre * self.var.cellArea,
                                                              self.var.reservoir_command_areas),
                                                  0)  # [M3]

                        ## Reservoir associated with the Command Area
                        #
                        # If there is more than one reservoir in a command area, the storage of the reservoir with maximum storage in this time-step is chosen.
                        # The map resStorageTotal_alloc holds this maximum reservoir storage within a command area in all cells within that command area

                        ReservoirsThatAreCurrentlyReservoirs = np.where(self.var.waterBodyTypCTemp == 2,
                                                                        self.var.reservoirStorageM3C, 0)
                        reservoirStorageM3 = globals.inZero.copy()
                        #np.put(reservoirStorageM3, self.var.decompress_LR, self.var.reservoirStorageM3C)
                        np.put(reservoirStorageM3, self.var.decompress_LR, ReservoirsThatAreCurrentlyReservoirs)

                        resStorageTotal_alloc = np.where(self.var.reservoir_command_areas > 0,
                                                         npareamaximum(reservoirStorageM3,
                                                                       self.var.reservoir_command_areas), 0)  # [M3]

                        # In the map resStorageTotal_allocC, the maximum storage from each allocation segment is held in all reservoir cells within that allocation segment.
                        # We now correct to remove the reservoirs that are not this maximum-storage-reservoir for the command area.
                        resStorageTotal_allocC = np.compress(self.var.compress_LR, resStorageTotal_alloc)
                        resStorageTotal_allocC = np.multiply(resStorageTotal_allocC == self.var.reservoirStorageM3C,
                                                             resStorageTotal_allocC)

                        day_of_year = globals.dateVar['currDate'].timetuple().tm_yday

                        if 'Reservoir_releases' in binding:
                            #resStorage_maxFracForIrrigation = 0.5 + globals.inZero.copy()
                            resStorage_maxFracForIrrigation = readnetcdf2('Reservoir_releases', day_of_year,
                                                                          useDaily='DOY', value='Fraction of Volume')
                        else:
                            resStorage_maxFracForIrrigation = 0.05 + globals.inZero.copy()

                        # resStorage_maxFracForIrrigationC holds the fractional rules found for each reservoir, so we must null those that are not the maximum-storage reservoirs
                        resStorage_maxFracForIrrigationC = np.compress(self.var.compress_LR,
                                                                       resStorage_maxFracForIrrigation)
                        resStorage_maxFracForIrrigationC = np.multiply(
                            resStorageTotal_allocC == self.var.reservoirStorageM3C, resStorage_maxFracForIrrigationC)
                        np.put(resStorage_maxFracForIrrigation, self.var.decompress_LR,
                               resStorage_maxFracForIrrigationC)

                        resStorage_maxFracForIrrigation_CA = np.where(self.var.reservoir_command_areas > 0,
                                                                      npareamaximum(resStorage_maxFracForIrrigation,
                                                                                    self.var.reservoir_command_areas),
                                                                      0)

                        Water_conveyance_efficiency = 1.0 + globals.inZero

                        act_bigLakeResAbst_alloc = np.minimum(
                            resStorage_maxFracForIrrigation_CA * resStorageTotal_alloc,
                            demand_Segment / Water_conveyance_efficiency)  # [M3]

                        ResAbstractFactor = np.where(resStorageTotal_alloc > 0,
                                                     divideValues(act_bigLakeResAbst_alloc, resStorageTotal_alloc),
                                                     0)  # fraction of water abstracted versus water available for total segment reservoir volumes
                        # Compressed version needs to be corrected as above
                        ResAbstractFactorC = np.compress(self.var.compress_LR, ResAbstractFactor)
                        ResAbstractFactorC = np.multiply(resStorageTotal_allocC == self.var.reservoirStorageM3C,
                                                         ResAbstractFactorC)

                        self.var.lakeStorageC -= self.var.reservoirStorageM3C * ResAbstractFactorC
                        self.var.lakeVolumeM3C -= self.var.reservoirStorageM3C * ResAbstractFactorC
                        self.var.lakeResStorageC -= self.var.reservoirStorageM3C * ResAbstractFactorC
                        self.var.reservoirStorageM3C -= self.var.reservoirStorageM3C * ResAbstractFactorC

                        self.var.abstractedLakeReservoirM3C += self.var.reservoirStorageM3C * ResAbstractFactorC
                        np.put(self.var.abstractedLakeReservoirM3, self.var.decompress_LR,
                               self.var.abstractedLakeReservoirM3C)

                        self.var.lakeResStorage = globals.inZero.copy()
                        np.put(self.var.lakeResStorage, self.var.decompress_LR, self.var.lakeResStorageC)

                        metRemainSegment = np.where(demand_Segment > 0,
                                                    divideValues(act_bigLakeResAbst_alloc * Water_conveyance_efficiency,
                                                                 demand_Segment), 0)  # by definition <= 1


                        self.var.act_bigLakeResAbst += remainNeedPre * metRemainSegment
                        self.var.act_SurfaceWaterAbstract += remainNeedPre * metRemainSegment

                        self.var.act_ResAbst = remainNeedPre * metRemainSegment

                        self.var.Res_Domestic = np.minimum(self.var.act_ResAbst,
                                                           pot_Res_Domestic)
                        self.var.Res_Livestock = np.minimum(self.var.act_ResAbst - self.var.Res_Domestic,
                                                            pot_Res_Livestock)
                        self.var.Res_Industry = np.minimum(
                            self.var.act_ResAbst - self.var.Res_Domestic - self.var.Res_Livestock,
                            pot_Res_Industry)

                    # If sector- and source-specific abstractions are activated, then domestic, industrial, and
                    #  livestock demands were attempted to be satisfied in the previous step. Otherwise, total demands
                    #  not satisfied by previous sources is attempted.
                    #
                    # The remaining demand within each command area [M3] is put into a map where each cell in the command area holds this total demand
                    demand_Segment = np.where(self.var.reservoir_command_areas > 0,
                                              npareatotal(remainNeed2 * self.var.cellArea,
                                                          self.var.reservoir_command_areas),
                                              0)  # [M3]

                    ## Reservoir associated with the Command Area
                    #
                    # If there is more than one reservoir in a command area, the storage of the reservoir with maximum storage in this time-step is chosen.
                    # The map resStorageTotal_alloc holds this maximum reservoir storage within a command area in all cells within that command area

                    ReservoirsThatAreCurrentlyReservoirs = np.where(self.var.waterBodyTypCTemp == 2,
                                                                    self.var.reservoirStorageM3C, 0)
                    reservoirStorageM3 = globals.inZero.copy()
                    # np.put(reservoirStorageM3, self.var.decompress_LR, self.var.reservoirStorageM3C)
                    np.put(reservoirStorageM3, self.var.decompress_LR, ReservoirsThatAreCurrentlyReservoirs)
                    
                    resStorageTotal_alloc = np.where(self.var.reservoir_command_areas > 0,
                                                     npareamaximum(reservoirStorageM3,
                                                                   self.var.reservoir_command_areas), 0)  # [M3]

                    # In the map resStorageTotal_allocC, the maximum storage from each allocation segment is held in all reservoir cells within that allocation segment.
                    # We now correct to remove the reservoirs that are not this maximum-storage-reservoir for the command area.
                    resStorageTotal_allocC = np.compress(self.var.compress_LR, resStorageTotal_alloc)
                    resStorageTotal_allocC = np.multiply(resStorageTotal_allocC == self.var.reservoirStorageM3C,
                                                         resStorageTotal_allocC)

                    # The rules for the maximum amount of water to be released for irrigation are found for the chosen maximum-storage reservoir in each command area
                    day_of_year = globals.dateVar['currDate'].timetuple().tm_yday

                    if 'Reservoir_releases' in binding:
                        resStorage_maxFracForIrrigation = readnetcdf2('Reservoir_releases', day_of_year,
                                                                      useDaily='DOY', value='Fraction of Volume')
                    else:
                        resStorage_maxFracForIrrigation = 0.05 + globals.inZero.copy()

                    # resStorage_maxFracForIrrigationC holds the fractional rules found for each reservoir, so we must null those that are not the maximum-storage reservoirs
                    resStorage_maxFracForIrrigationC = np.compress(self.var.compress_LR,
                                                                   resStorage_maxFracForIrrigation)
                    resStorage_maxFracForIrrigationC = np.multiply(
                        resStorageTotal_allocC == self.var.reservoirStorageM3C, resStorage_maxFracForIrrigationC)
                    np.put(resStorage_maxFracForIrrigation, self.var.decompress_LR, resStorage_maxFracForIrrigationC)

                    resStorage_maxFracForIrrigation_CA = np.where(self.var.reservoir_command_areas > 0,
                                                                  npareamaximum(resStorage_maxFracForIrrigation,
                                                                                self.var.reservoir_command_areas), 0)

                    if self.var.modflow and 'Water_conveyance_efficiency' in binding:
                        Water_conveyance_efficiency = loadmap('Water_conveyance_efficiency') + globals.inZero
                    else:
                        Water_conveyance_efficiency = 1.0 + globals.inZero

                    act_bigLakeResAbst_alloc = np.minimum(resStorage_maxFracForIrrigation_CA * resStorageTotal_alloc,
                                                          demand_Segment / Water_conveyance_efficiency)  # [M3]

                    ResAbstractFactor = np.where(resStorageTotal_alloc > 0,
                                                 divideValues(act_bigLakeResAbst_alloc, resStorageTotal_alloc),
                                                 0)  # fraction of water abstracted versus water available for total segment reservoir volumes
                    # Compressed version needs to be corrected as above
                    ResAbstractFactorC = np.compress(self.var.compress_LR, ResAbstractFactor)
                    ResAbstractFactorC = np.multiply(resStorageTotal_allocC == self.var.reservoirStorageM3C,
                                                     ResAbstractFactorC)

                    self.var.lakeStorageC -= self.var.reservoirStorageM3C * ResAbstractFactorC
                    self.var.lakeVolumeM3C -= self.var.reservoirStorageM3C * ResAbstractFactorC
                    self.var.lakeResStorageC -= self.var.reservoirStorageM3C * ResAbstractFactorC
                    self.var.reservoirStorageM3C -= self.var.reservoirStorageM3C * ResAbstractFactorC

                    self.var.abstractedLakeReservoirM3C += self.var.reservoirStorageM3C * ResAbstractFactorC
                    np.put(self.var.abstractedLakeReservoirM3, self.var.decompress_LR,
                           self.var.abstractedLakeReservoirM3C)

                    self.var.lakeResStorage = globals.inZero.copy()
                    np.put(self.var.lakeResStorage, self.var.decompress_LR, self.var.lakeResStorageC)

                    metRemainSegment = np.where(demand_Segment > 0,
                                                divideValues(act_bigLakeResAbst_alloc * Water_conveyance_efficiency,
                                                             demand_Segment), 0)  # by definition <= 1

                    # self.var.leakageC_daily = resStorageTotal_allocC * ResAbstractFactorC * (
                    #            1 - Water_conveyance_efficiency)
                    self.var.leakageC_daily = resStorageTotal_allocC * ResAbstractFactorC * (
                                1 - np.compress(self.var.compress_LR, Water_conveyance_efficiency))

                    self.var.leakage = globals.inZero.copy()
                    np.put(self.var.leakage, self.var.decompress_LR, self.var.leakageC_daily)


                    #self.var.leakageC += self.var.leakageC_daily

                    self.var.leakageCanalsC_M = np.where(self.var.canalsAreaC > 0,
                                                         self.var.leakageC_daily / self.var.canalsAreaC, 0)

                    self.var.leakageCanals_M = globals.inZero.copy()  # Without this, npareamaximum uses the historical maximum
                    np.put(self.var.leakageCanals_M, self.var.decompress_LR, self.var.leakageCanalsC_M)  # good
                    self.var.leakageCanals_M = npareamaximum(self.var.leakageCanals_M,
                                                             self.var.canals)


                    self.var.act_bigLakeResAbst += remainNeed2 * metRemainSegment
                    self.var.act_SurfaceWaterAbstract += remainNeed2 * metRemainSegment

                    self.var.act_ResAbst += remainNeed2 * metRemainSegment

                    ## End of using_reservoir_command_areas

                    if self.var.sectorSourceAbstractionFractions:

                        self.var.Res_Irrigation = np.minimum(
                            remainNeed2 * metRemainSegment,
                            pot_Res_Irrigation)

                # B

            # remaining is taken from groundwater if possible
            if self.var.sectorSourceAbstractionFractions:
                pot_GW_Domestic = np.minimum(
                    self.var.gwAbstractionFraction_Domestic * self.var.domesticDemand,
                    self.var.domesticDemand - self.var.Channel_Domestic - self.var.Lake_Domestic - self.var.Res_Domestic)

                pot_GW_Livestock = np.minimum(
                    self.var.gwAbstractionFraction_Livestock * self.var.livestockDemand,
                    self.var.livestockDemand - self.var.Channel_Livestock - self.var.Lake_Livestock - self.var.Res_Livestock)

                pot_GW_Industry = np.minimum(
                    self.var.gwAbstractionFraction_Industry * self.var.industryDemand,
                    self.var.industryDemand - self.var.Channel_Industry - self.var.Lake_Industry- self.var.Res_Industry)

                pot_GW_Irrigation = np.minimum(
                    self.var.gwAbstractionFraction_Irrigation * self.var.totalIrrDemand,
                    self.var.totalIrrDemand - self.var.Channel_Irrigation - self.var.Lake_Irrigation - self.var.Res_Irrigation)


                self.var.pot_GroundwaterAbstract = pot_GW_Domestic + pot_GW_Livestock + pot_GW_Industry + pot_GW_Irrigation
            else:
                self.var.pot_GroundwaterAbstract = totalDemand - self.var.act_SurfaceWaterAbstract

            self.var.nonFossilGroundwaterAbs = np.maximum(0.,np.minimum(self.var.readAvlStorGroundwater, self.var.pot_GroundwaterAbstract))

            # calculate renewableAvlWater_local (non-fossil groundwater and channel) - environmental flow
            #self.var.renewableAvlWater_local = self.var.readAvlStorGroundwater + self.var.readAvlChannelStorageM

            if self.var.modflow:
                # if available storage is too low, no pumping in this cell (defined in transient module)

                # self.var.allowedPumping = np.maximum(self.var.groundwater_storage_available - (1-self.var.availableGWStorageFraction)*self.var.gwstorage_full, 0)
                # self.var.nonFossilGroundwaterAbs = np.minimum(self.var.allowedPumping, self.var.pot_GroundwaterAbstract)  # gwstorage_cell
                self.var.nonFossilGroundwaterAbs = np.where(self.var.groundwater_storage_available > (
                        1 - self.var.availableGWStorageFraction) * self.var.gwstorage_full, np.minimum(self.var.groundwater_storage_available, self.var.pot_GroundwaterAbstract), 0)
                # self.var.nonSatisfiedGWrequest = self.var.pot_GroundwaterAbstract - self.var.nonFossilGroundwaterAbs

            # if limitAbstraction from groundwater is True
            # fossil gwAbstraction and water demand may be reduced
            # variable to reduce/limit groundwater abstraction (> 0 if limitAbstraction = True)

            if self.var.sectorSourceAbstractionFractions:
                # A
                self.var.GW_Domestic = np.minimum(self.var.nonFossilGroundwaterAbs, pot_GW_Domestic)
                self.var.GW_Livestock = np.minimum(self.var.nonFossilGroundwaterAbs - self.var.GW_Domestic,
                                                   pot_GW_Livestock)
                self.var.GW_Industry = np.minimum(
                    self.var.nonFossilGroundwaterAbs - self.var.GW_Domestic - self.var.GW_Livestock,
                    pot_GW_Industry)
                self.var.GW_Irrigation = np.minimum(
                    self.var.nonFossilGroundwaterAbs - self.var.GW_Domestic - self.var.GW_Livestock - self.var.GW_Industry,
                    pot_GW_Irrigation)

                unmet_Domestic = self.var.domesticDemand - self.var.Channel_Domestic - self.var.Lake_Domestic - self.var.Res_Domestic - self.var.GW_Domestic
                unmet_Livestock = self.var.livestockDemand - self.var.Channel_Livestock - self.var.Lake_Livestock - self.var.Res_Livestock - self.var.GW_Livestock
                unmet_Industry = self.var.industryDemand - self.var.Channel_Industry - self.var.Lake_Industry - self.var.Res_Industry - self.var.GW_Industry
                unmet_Irrigation = self.var.totalIrrDemand - self.var.Channel_Irrigation - self.var.Lake_Irrigation - self.var.Res_Irrigation - self.var.GW_Irrigation

            if checkOption('limitAbstraction'):
                # real surface water abstraction can be lower, because not all demand can be done from surface water
                act_swAbstractionFraction = divideValues(self.var.act_SurfaceWaterAbstract, totalDemand)
                # Fossil groundwater abstraction is not allowed
                # allocation rule here: domestic& industry > irrigation > paddy

                if self.var.sectorSourceAbstractionFractions:

                    self.var.act_nonIrrWithdrawal = self.var.Channel_Domestic + self.var.Channel_Livestock + self.var.Channel_Industry + \
                                                    self.var.Lake_Domestic + self.var.Lake_Livestock + self.var.Lake_Industry +\
                                                    self.var.Res_Domestic + self.var.Res_Livestock + self.var.Res_Industry + \
                                                    self.var.GW_Domestic + self.var.GW_Livestock + self.var.GW_Industry
                    self.var.act_irrWithdrawal = self.var.Channel_Irrigation +  self.var.Lake_Irrigation + self.var.Res_Irrigation + self.var.GW_Irrigation
                    act_irrWithdrawalSW = self.var.Channel_Irrigation +  self.var.Lake_Irrigation + self.var.Res_Irrigation
                    act_irrWithdrawalGW = self.var.GW_Irrigation
                    self.var.act_irrNonpaddyWithdrawal = np.minimum(self.var.act_irrWithdrawal, self.var.fracVegCover[3] * self.var.irrDemand[3])
                    self.var.act_irrPaddyWithdrawal = self.var.act_irrWithdrawal - self.var.act_irrNonpaddyWithdrawal

                    act_gw = np.copy(self.var.nonFossilGroundwaterAbs)



                elif self.var.includeIndusDomesDemand:  # all demands are taken into account
                    # non-irrgated water demand: adjusted (and maybe increased) by gwabstration factor
                    # if non-irrgated water demand is higher than actual growndwater abstraction (what is needed and what is stored in gw)
                    act_nonIrrWithdrawalGW = self.var.nonIrrDemand * (1 - act_swAbstractionFraction)
                    act_nonIrrWithdrawalGW = np.where(act_nonIrrWithdrawalGW > self.var.nonFossilGroundwaterAbs, self.var.nonFossilGroundwaterAbs, act_nonIrrWithdrawalGW)
                    act_nonIrrWithdrawalSW = act_swAbstractionFraction * self.var.nonIrrDemand
                    self.var.act_nonIrrWithdrawal = act_nonIrrWithdrawalSW + act_nonIrrWithdrawalGW

                    # irrigated water demand:
                    act_irrWithdrawalGW = self.var.totalIrrDemand * (1 - act_swAbstractionFraction)
                    act_irrWithdrawalGW = np.minimum(self.var.nonFossilGroundwaterAbs - act_nonIrrWithdrawalGW, act_irrWithdrawalGW)
                    act_irrWithdrawalSW = act_swAbstractionFraction * self.var.totalIrrDemand
                    self.var.act_irrWithdrawal = act_irrWithdrawalSW + act_irrWithdrawalGW
                    # (nonpaddy)
                    act_irrnonpaddyGW = self.var.fracVegCover[3] * (1 - act_swAbstractionFraction) * self.var.irrDemand[3]
                    act_irrnonpaddyGW = np.minimum(self.var.nonFossilGroundwaterAbs - act_nonIrrWithdrawalGW, act_irrnonpaddyGW)
                    act_irrnonpaddySW = self.var.fracVegCover[3] * act_swAbstractionFraction * self.var.irrDemand[3]
                    self.var.act_irrNonpaddyWithdrawal = act_irrnonpaddySW + act_irrnonpaddyGW
                    # (paddy)
                    act_irrpaddyGW = self.var.fracVegCover[2] * (1 - act_swAbstractionFraction) * self.var.irrDemand[2]
                    act_irrpaddyGW = np.minimum(self.var.nonFossilGroundwaterAbs - act_nonIrrWithdrawalGW - act_irrnonpaddyGW, act_irrpaddyGW)
                    act_irrpaddySW = self.var.fracVegCover[2] * act_swAbstractionFraction * self.var.irrDemand[2]
                    self.var.act_irrPaddyWithdrawal = act_irrpaddySW + act_irrpaddyGW

                    act_gw = act_nonIrrWithdrawalGW + act_irrWithdrawalGW
                    # This should be equal to self.var.nonFossilGroundwaterAbs?


                else:  # only irrigation is considered

                    self.var.act_nonIrrWithdrawal = globals.inZero.copy()

                    # irrigated water demand:
                    act_irrWithdrawalGW = self.var.totalIrrDemand * (1 - act_swAbstractionFraction)
                    act_irrWithdrawalGW = np.minimum(self.var.nonFossilGroundwaterAbs, act_irrWithdrawalGW)
                    act_irrWithdrawalSW = act_swAbstractionFraction * self.var.totalIrrDemand
                    self.var.act_irrWithdrawal = act_irrWithdrawalSW + act_irrWithdrawalGW
                    # (nonpaddy)
                    act_irrnonpaddyGW = self.var.fracVegCover[3] * (1 - act_swAbstractionFraction) * self.var.irrDemand[3]
                    act_irrnonpaddyGW = np.minimum(self.var.nonFossilGroundwaterAbs, act_irrnonpaddyGW)
                    act_irrnonpaddySW = self.var.fracVegCover[3] * act_swAbstractionFraction * self.var.irrDemand[3]
                    self.var.act_irrNonpaddyWithdrawal = act_irrnonpaddySW + act_irrnonpaddyGW
                    # (paddy)
                    act_irrpaddyGW = self.var.fracVegCover[2] * (1 - act_swAbstractionFraction) * self.var.irrDemand[2]
                    act_irrpaddyGW = np.minimum(
                        self.var.nonFossilGroundwaterAbs - act_irrnonpaddyGW, act_irrpaddyGW)
                    act_irrpaddySW = self.var.fracVegCover[2] * act_swAbstractionFraction * self.var.irrDemand[2]
                    self.var.act_irrPaddyWithdrawal = act_irrpaddySW + act_irrpaddyGW

                    act_gw = np.copy(act_irrWithdrawalGW)


                # todo: is act_irrWithdrawal needed to be replaced? Check later!!
                # consumption - irrigation (without loss) = demand  * efficiency   (back to non fraction value)

                ## back to non fraction values
                # self.var.act_irrWithdrawal[2] = divideValues(self.var.act_irrPaddyWithdrawal, self.var.fracVegCover[2])
                #self.var.act_irrWithdrawal[3] = divideValues(self.var.act_irrNonpaddyWithdrawal, self.var.fracVegCover[3])
                ## consumption - irrigation (without loss) = demand  * efficiency

                # calculate act_ water demand, because irr demand has still demand from previous day included
                # if the demand from previous day is not fulfilled it is taken to the next day and so on
                # if we do not correct we double account each day the demand from previous days
                self.var.act_irrPaddyDemand = np.maximum(0, self.var.irrPaddyDemand - self.var.unmetDemandPaddy)
                self.var.act_irrNonpaddyDemand = np.maximum(0, self.var.irrNonpaddyDemand - self.var.unmetDemandNonpaddy)

                # unmet is either pot_GroundwaterAbstract - self.var.nonFossilGroundwaterAbs or demand - withdrawal
                if self.var.includeIndusDomesDemand:  # all demands are taken into account
                    self.var.unmetDemand = (self.var.totalIrrDemand - self.var.act_irrWithdrawal) + (self.var.nonIrrDemand - self.var.act_nonIrrWithdrawal)
                else:  # only irrigation is considered
                    self.var.unmetDemand = (self.var.totalIrrDemand - self.var.act_irrWithdrawal) - self.var.act_nonIrrWithdrawal
                self.var.unmetDemandPaddy = self.var.irrPaddyDemand - self.var.act_irrPaddyDemand
                self.var.unmetDemandNonpaddy = self.var.irrNonpaddyDemand - self.var.act_irrNonpaddyDemand


            else:
                # This is the case when using ModFlow coupling (limitation imposed previously)
                if self.var.modflow:
                    # This is the case when using ModFlow coupling (limitation imposed previously)
                    # part of the groundwater demand unsatisfied
                    self.var.unmetDemand = self.var.pot_GroundwaterAbstract - self.var.nonFossilGroundwaterAbs

                    if self.var.includeIndusDomesDemand:  # all demands are taken into account
                        self.var.act_nonIrrWithdrawal = np.copy(self.var.nonIrrDemand)
                    self.var.act_irrWithdrawal = np.copy(self.var.totalIrrDemand)

                    act_gw = np.copy(self.var.nonFossilGroundwaterAbs)

                else:
                    # Fossil groundwater abstractions are allowed (act = pot)

                    # using allocation from abstraction zone
                    # this might be a regualr grid e.g. 2x2 for 0.5 deg
                    left_sf = self.var.readAvlChannelStorageM # already removed - self.var.act_channelAbst
                    # sum demand, surface water - local used, groundwater - local use, not satisfied for allocation zone

                    if self.var.sectorSourceAbstractionFractions:
                        unmetChannel_Domestic = pot_Channel_Domestic - self.var.Channel_Domestic
                        unmetChannel_Livestock = pot_Channel_Livestock - self.var.Channel_Livestock
                        unmetChannel_Industry = pot_Channel_Industry - self.var.Channel_Industry
                        unmetChannel_Irrigation = pot_Channel_Irrigation - self.var.Channel_Irrigation

                        pot_Channel_Domestic = np.minimum(unmetChannel_Domestic, unmet_Domestic)
                        pot_Channel_Livestock = np.minimum(unmetChannel_Livestock, unmet_Livestock)
                        pot_Channel_Industry = np.minimum(unmetChannel_Industry, unmet_Industry)
                        pot_Channel_Irrigation = np.minimum(unmetChannel_Irrigation, unmet_Irrigation)

                        unmet_Channel = pot_Channel_Domestic + pot_Channel_Livestock + pot_Channel_Industry + pot_Channel_Irrigation

                        zoneDemand = npareatotal(unmet_Channel, self.var.allocation_zone)

                    else:
                        zoneDemand = npareatotal(self.var.unmetDemand, self.var.allocation_zone)


                    zone_sf_avail = npareatotal(left_sf, self.var.allocation_zone)

                    # zone abstraction is minimum of availability and demand
                    zone_sf_abstraction = np.minimum(zoneDemand,zone_sf_avail)
                    # water taken from surface zone and allocated to cell demand
                    cell_sf_abstraction = np.maximum(0.,divideValues(left_sf,zone_sf_avail) * zone_sf_abstraction)
                    cell_sf_allocation = np.maximum(0.,divideValues(self.var.unmetDemand, zoneDemand) * zone_sf_abstraction)

                    # sum up with other abstraction
                    self.var.act_SurfaceWaterAbstract = self.var.act_SurfaceWaterAbstract + cell_sf_abstraction
                    self.var.act_channelAbst = self.var.act_channelAbst +  cell_sf_abstraction

                    if self.var.sectorSourceAbstractionFractions:
                        self.var.Channel_Domestic_fromZone = np.minimum(cell_sf_abstraction, pot_Channel_Domestic)
                        self.var.Channel_Domestic += self.var.Channel_Domestic_fromZone

                        self.var.Channel_Livestock_fromZone = np.minimum(cell_sf_abstraction - self.var.Channel_Domestic_fromZone,
                                                                         pot_Channel_Livestock)
                        self.var.Channel_Livestock += self.var.Channel_Livestock_fromZone

                        self.var.Channel_Industry_fromZone = np.minimum(
                            cell_sf_abstraction - self.var.Channel_Domestic_fromZone - self.var.Channel_Livestock_fromZone,
                            pot_Channel_Industry)
                        self.var.Channel_Industry += self.var.Channel_Industry_fromZone

                        self.var.Channel_Irrigation_fromZone = np.minimum(
                            cell_sf_abstraction - self.var.Channel_Domestic_fromZone - self.var.Channel_Livestock_fromZone - self.var.Channel_Industry_fromZone,
                            pot_Channel_Irrigation)
                        self.var.Channel_Irrigation += self.var.Channel_Irrigation_fromZone


                    # new potential groundwater abstraction
                    self.var.pot_GroundwaterAbstract = np.maximum(0.,self.var.pot_GroundwaterAbstract - cell_sf_allocation)

                    left_gw_demand = np.maximum(0.,self.var.pot_GroundwaterAbstract - self.var.nonFossilGroundwaterAbs)
                    left_gw_avail = self.var.readAvlStorGroundwater - self.var.nonFossilGroundwaterAbs
                    zone_gw_avail = npareatotal(left_gw_avail, self.var.allocation_zone)

                    # for groundwater substract demand which is fulfilled by surface zone, calc abstraction and what is left.
                    #zone_gw_demand = npareatotal(left_gw_demand, self.var.allocation_zone)
                    zone_gw_demand = zoneDemand -  zone_sf_abstraction
                    zone_gw_abstraction = np.minimum(zone_gw_demand,zone_gw_avail)
                    #zone_unmetdemand = np.maximum(0., zone_gw_demand - zone_gw_abstraction)

                    # water taken from groundwater zone and allocated to cell demand
                    cell_gw_abstraction = np.maximum(0.,divideValues(left_gw_avail,zone_gw_avail) * zone_gw_abstraction)
                    cell_gw_allocation = np.maximum(0.,divideValues(left_gw_demand,zone_gw_demand) * zone_gw_abstraction)

                    self.var.unmetDemand = np.maximum(0.,left_gw_demand - cell_gw_allocation)
                    self.var.nonFossilGroundwaterAbs = self.var.nonFossilGroundwaterAbs + cell_gw_abstraction

                    #UNDER CONSTRUCTION
                    if self.var.sectorSourceAbstractionFractions:
                        self.var.GW_Domestic_fromZone = np.minimum(self.var.nonFossilGroundwaterAbs, pot_GW_Domestic)
                        self.var.GW_Domestic += self.var.GW_Domestic_fromZone.copy()

                        self.var.GW_Livestock_fromZone = np.minimum(self.var.nonFossilGroundwaterAbs - self.var.GW_Domestic_fromZone,
                                                                         pot_GW_Livestock)
                        self.var.GW_Livestock += self.var.GW_Livestock_fromZone.copy()

                        self.var.GW_Industry_fromZone = np.minimum(
                            self.var.nonFossilGroundwaterAbs - self.var.GW_Domestic_fromZone - self.var.GW_Livestock_fromZone,
                            pot_GW_Industry)
                        self.var.GW_Industry += self.var.GW_Industry_fromZone.copy()

                        self.var.GW_Irrigation_fromZone = np.minimum(
                            self.var.nonFossilGroundwaterAbs - self.var.GW_Domestic_fromZone - self.var.GW_Livestock_fromZone - self.var.GW_Industry_fromZone,
                            pot_GW_Irrigation)
                        self.var.GW_Irrigation += self.var.GW_Irrigation_fromZone.copy()

                    #self.var.unmetDemand = self.var.pot_GroundwaterAbstract - self.var.nonFossilGroundwaterAbs
                    ## end of zonal abstraction

                    # unmet demand is again checked for water from channels and abstraction from surface is increased
                    #channelAbs2 = np.minimum(self.var.readAvlChannelStorageM - self.var.act_channelAbst, self.var.unmetDemand)
                    #self.var.act_SurfaceWaterAbstract = self.var.act_SurfaceWaterAbstract + channelAbs2
                    #self.var.act_channelAbst = self.var.act_channelAbst +  channelAbs2
                    #self.var.unmetDemand = self.var.unmetDemand - channelAbs2
                    #self.var.pot_GroundwaterAbstract = self.var.pot_GroundwaterAbstract - channelAbs2

                    if self.var.includeIndusDomesDemand:  # all demands are taken into account
                        self.var.act_nonIrrWithdrawal = np.copy(self.var.nonIrrDemand)
                    self.var.act_irrWithdrawal = np.copy(self.var.totalIrrDemand)

                    act_gw = np.copy(self.var.pot_GroundwaterAbstract)
                    # LUCA: TAKE CARE OF THE ORDER OF FRACLANDCOVER

                self.var.act_irrNonpaddyWithdrawal = self.var.fracVegCover[3] * self.var.irrDemand[3]
                self.var.act_irrPaddyWithdrawal = self.var.fracVegCover[2] * self.var.irrDemand[2]

                self.var.act_irrNonpaddyDemand = self.var.act_irrNonpaddyWithdrawal.copy()
                self.var.act_irrPaddyDemand = self.var.act_irrPaddyWithdrawal.copy()


            ## End of limit extraction if, then

            self.var.act_irrConsumption[2] = divideValues(self.var.act_irrPaddyWithdrawal, self.var.fracVegCover[2]) * self.var.efficiencyPaddy
            self.var.act_irrConsumption[3] = divideValues(self.var.act_irrNonpaddyWithdrawal, self.var.fracVegCover[3]) * self.var.efficiencyNonpaddy

            if self.var.modflow:
                if self.var.GW_pumping:  # pumping demand is sent to ModFlow (used in transient module)
                    self.var.modfPumpingM += act_gw  # modfPumpingM is initialized every "modflow_timestep" in "groundwater_modflow/transient.py"
                    self.var.Pumping_daily = np.copy(act_gw)
                    self.var.PumpingM3_daily = act_gw * self.var.cellArea

            if self.var.sectorSourceAbstractionFractions:

                self.var.act_indWithdrawal = self.var.Channel_Industry + self.var.Lake_Industry + self.var.Res_Industry + self.var.GW_Industry
                self.var.act_domWithdrawal = self.var.Channel_Domestic + self.var.Lake_Domestic + self.var.Res_Domestic + self.var.GW_Domestic
                self.var.act_livWithdrawal = self.var.Channel_Livestock + self.var.Lake_Livestock + self.var.Res_Livestock + self.var.GW_Livestock
                self.var.act_indConsumption = self.var.ind_efficiency * self.var.act_indWithdrawal
                self.var.act_domConsumption = self.var.dom_efficiency * self.var.act_domWithdrawal
                self.var.act_livConsumption = self.var.liv_efficiency * self.var.act_livWithdrawal
                self.var.act_nonIrrConsumption = self.var.act_domConsumption + self.var.act_indConsumption + self.var.act_livConsumption

            elif self.var.includeIndusDomesDemand:  # all demands are taken into account

                self.var.act_indWithdrawal = frac_industry * self.var.act_nonIrrWithdrawal
                self.var.act_domWithdrawal = frac_domestic * self.var.act_nonIrrWithdrawal
                self.var.act_livWithdrawal = frac_livestock * self.var.act_nonIrrWithdrawal
                self.var.act_indConsumption = self.var.ind_efficiency * self.var.act_indWithdrawal
                self.var.act_domConsumption = self.var.dom_efficiency * self.var.act_domWithdrawal
                self.var.act_livConsumption = self.var.liv_efficiency * self.var.act_livWithdrawal
                self.var.act_nonIrrConsumption = self.var.act_domConsumption + self.var.act_indConsumption + self.var.act_livConsumption

            else:  # only irrigation is considered
                self.var.act_nonIrrConsumption = globals.inZero.copy()

            self.var.act_totalIrrConsumption = self.var.fracVegCover[2] * self.var.act_irrConsumption[2] + self.var.fracVegCover[3] * self.var.act_irrConsumption[3]
            self.var.act_paddyConsumption = self.var.fracVegCover[2] * self.var.act_irrConsumption[2]
            self.var.act_nonpaddyConsumption = self.var.fracVegCover[3] * self.var.act_irrConsumption[3]

            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.var.totalWaterDemand = self.var.fracVegCover[2] * self.var.irrDemand[2] + self.var.fracVegCover[3] * self.var.irrDemand[3] + self.var.nonIrrDemand
                self.var.act_totalWaterWithdrawal = self.var.act_nonIrrWithdrawal + self.var.act_irrWithdrawal
                self.var.act_totalWaterConsumption = self.var.act_nonIrrConsumption + self.var.act_totalIrrConsumption
            else:  # only irrigation is considered
                self.var.totalWaterDemand = self.var.fracVegCover[2] * self.var.irrDemand[2] + self.var.fracVegCover[3] * self.var.irrDemand[3]
                self.var.act_totalWaterWithdrawal = np.copy(self.var.act_irrWithdrawal)
                self.var.act_totalWaterConsumption = np.copy(self.var.act_totalIrrConsumption)

            # --- calculate return flow
            #Sum up loss - difference between withdrawn and consumed - split into return flow and evaporation
            sumIrrLoss = self.var.act_irrWithdrawal - self.var.act_totalIrrConsumption

            self.var.returnflowIrr =  self.var.returnfractionIrr * sumIrrLoss
            self.var.addtoevapotrans = (1- self.var.returnfractionIrr) * sumIrrLoss

            if self.var.sectorSourceAbstractionFractions:
                self.var.returnflowNonIrr = self.var.act_nonIrrWithdrawal - self.var.act_nonIrrConsumption
            elif self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.var.returnflowNonIrr  = self.var.nonIrrReturnFlowFraction * self.var.act_nonIrrWithdrawal

            # limit return flow to not put all fossil groundwater back into the system, because
            # it can lead to higher river discharge than without water demand, as water is taken from fossil groundwater (out of system)
            unmet_div_ww = 1. - np.minimum(1, divideValues(self.var.unmetDemand, self.var.act_totalWaterWithdrawal))

            if checkOption('limitAbstraction'):
                unmet_div_ww = 1

            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.var.unmet_lost = ( self.var.returnflowIrr + self.var.returnflowNonIrr +  self.var.addtoevapotrans) * (1-unmet_div_ww)
            else:  # only irrigation is considered
                self.var.unmet_lost = (self.var.returnflowIrr + self.var.addtoevapotrans) * (1 - unmet_div_ww)


            #self.var.waterDemandLost = self.var.act_totalWaterConsumption + self.var.addtoevapotrans
            self.var.unmet_lostirr = ( self.var.returnflowIrr  +  self.var.addtoevapotrans) * (1-unmet_div_ww)
            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.var.unmet_lostNonirr = self.var.returnflowNonIrr * (1-unmet_div_ww)

            self.var.returnflowIrr = self.var.returnflowIrr * unmet_div_ww
            self.var.addtoevapotrans = self.var.addtoevapotrans * unmet_div_ww
            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.var.returnflowNonIrr =  self.var.returnflowNonIrr * unmet_div_ww

            # returnflow to river and to evapotranspiration
            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.var.returnFlow = self.var.returnflowIrr + self.var.returnflowNonIrr
            else:  # only irrigation is considered
                self.var.returnFlow = self.var.returnflowIrr
            self.var.waterabstraction = self.var.nonFossilGroundwaterAbs + self.var.unmetDemand + self.var.act_SurfaceWaterAbstract

            if 'adminSegments' in binding:

                self.var.act_irrWithdrawalSW_month += npareatotal(act_irrWithdrawalSW * self.var.cellArea,
                                                                  self.var.adminSegments)

                if 'irrigation_agent_SW_request_month_m3' in binding and self.var.activate_irrigation_agents:

                    self.var.swAbstractionFraction_Channel_Irrigation = np.where(
                        self.var.act_irrWithdrawalSW_month > self.var.irrWithdrawalSW_max, 0,
                        self.var.swAbstractionFraction_Channel_Irrigation)

                    self.var.swAbstractionFraction_Lake_Irrigation = np.where(
                        self.var.act_irrWithdrawalSW_month > self.var.irrWithdrawalSW_max, 0,
                        self.var.swAbstractionFraction_Lake_Irrigation)

                    self.var.swAbstractionFraction_Res_Irrigation = np.where(
                        self.var.act_irrWithdrawalSW_month > self.var.irrWithdrawalSW_max, 0,
                        self.var.swAbstractionFraction_Res_Irrigation)

                    self.var.ratio_irrWithdrawalSW_month = self.var.act_irrWithdrawalSW_month / self.var.irrWithdrawalSW_max


            if 'adminSegments' in binding:
                self.var.act_irrWithdrawalGW_month += npareatotal(act_irrWithdrawalGW * self.var.cellArea,
                                                                  self.var.adminSegments)
                if 'irrigation_agent_GW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                    self.var.gwAbstractionFraction_Irrigation = np.where(
                        self.var.act_irrWithdrawalGW_month > self.var.irrWithdrawalGW_max, 0,
                        self.var.gwAbstractionFraction_Irrigation)

                    self.var.ratio_irrWithdrawalGW_month = self.var.act_irrWithdrawalGW_month / self.var.irrWithdrawalGW_max

            if self.var.relax_irrigation_agents:
                if dateVar['currDate'].day == 10:
                    if 'irrigation_agent_SW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                        self.var.relaxSWagent += np.where(self.var.ratio_irrWithdrawalSW_month > 0.95, 1, 0)
                    if 'irrigation_agent_GW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                        self.var.relaxGWagent += np.where(self.var.ratio_irrWithdrawalGW_month > 0.95, 1, 0)

                # This will decrease values that have increased, but not on agents that were never too large
                if dateVar['currDate'].day == 28:
                    if 'irrigation_agent_SW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                        self.var.relaxSWagent -= np.where(self.var.relaxSWagent > 0,
                                                          np.where(self.var.ratio_irrWithdrawalSW_month > 0.98, 0, 1), 0)
                    if 'irrigation_agent_GW_request_month_m3' in binding and self.var.activate_irrigation_agents:
                        self.var.relaxGWagent -= np.where(self.var.relaxGWagent > 0,
                                                          np.where(self.var.ratio_irrWithdrawalGW_month > 0.98, 0, 1), 0)


            #---------------------------------------------
            # testing
            
            
            if self.var.includeIndusDomesDemand:  # all demands are taken into account
                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_irrWithdrawal],  # In
                    [self.var.act_totalIrrConsumption,self.var.unmet_lostirr,self.var.addtoevapotrans,self.var.returnflowIrr],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand5a", False)


                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_nonIrrWithdrawal],  # In
                    [self.var.act_nonIrrConsumption , self.var.returnflowNonIrr, self.var.unmet_lostNonirr],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand5b", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.ind_efficiency * frac_industry * self.var.act_nonIrrWithdrawal],  # In
                    [self.var.act_indConsumption],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand5c", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [ self.var.act_indWithdrawal],  # In
                    [self.var.act_indConsumption/ self.var.ind_efficiency],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand5d", False)


            # ----------------------------------------------------------------
            if checkOption('calcWaterBalance'):
                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_irrWithdrawal],  # In
                    [self.var.act_totalIrrConsumption, self.var.returnflowIrr,self.var.unmet_lostirr,self.var.addtoevapotrans],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterlossdemand1", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.nonIrrDemand, self.var.totalIrrDemand],  # In
                    [self.var.nonFossilGroundwaterAbs, self.var.unmetDemand, self.var.act_SurfaceWaterAbstract],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand1", False)
                if checkOption('includeWaterBodies'):
                    self.model.waterbalance_module.waterBalanceCheck(
                        [self.var.act_SurfaceWaterAbstract],  # In
                        [ self.var.act_bigLakeResAbst,self.var.act_smallLakeResAbst, self.var.act_channelAbst],  # Out
                        [globals.inZero],
                        [globals.inZero],
                        "Waterdemand1b", False)


                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.nonFossilGroundwaterAbs, self.var.unmetDemand, self.var.act_SurfaceWaterAbstract],  # In
                    [self.var.act_totalWaterWithdrawal],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand2", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_totalWaterWithdrawal],  # In
                    [self.var.act_irrPaddyWithdrawal, self.var.act_irrNonpaddyWithdrawal, self.var.act_nonIrrWithdrawal],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand3", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_totalWaterWithdrawal],  # In
                    [self.var.act_totalIrrConsumption, self.var.act_nonIrrConsumption, self.var.addtoevapotrans, self.var.returnflowIrr, self.var.returnflowNonIrr, self.var.unmet_lost],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand4", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_totalWaterWithdrawal],  # In
                    [self.var.act_totalIrrConsumption, self.var.act_nonIrrConsumption, self.var.addtoevapotrans, self.var.returnFlow, self.var.unmet_lost ],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand5", False)

                self.model.waterbalance_module.waterBalanceCheck(
                    [self.var.act_totalWaterWithdrawal],  # In
                    [self.var.waterabstraction],  # Out
                    [globals.inZero],
                    [globals.inZero],
                    "Waterdemand level1", False)
