# author: TK, wrapped by AJF
# modified on 11-17-2020 #NOTE -- Modify the date when changes are made
# description: an optimization code to decide on land and water use by agent.
# This is a standalone file with the optimization wrapped in one function.
# This funciton is called by model component > institutions > farm_decisions.py
# author: TK, wrapped by AJF
# modified on 11-17-2020 -- eliminate redundancy #NOTE -- Modify the date when changes are made

# --------------------------
# description: an optimization code to decide on land and water use by agent.
# This is a standalone file with the optimization wrapped in one function.
# This funciton is called by model component > institutions > farm_decisions.py
# --------------------------

#########################################################
#       _           ____
#      / \   __ _  |  _ \ _   _  ___  _ __ ___   ___
#     / _ \ / _` | | |_) | | | |/ _ \| '_ ` _ \ / _ \
#    / ___ \ (_| | |  __/| |_| | (_) | | | | | | (_) |
#   /_/   \_\__, | |_|    \__, |\___/|_| |_| |_|\___/
#           |___/         |___/
#########################################################

import pyomo.environ as pyo
# import
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
from helper_objects.helper_functions import tic


# =====================================================
# Optimization wrapped as a function
# =====================================================

def ag_optimization(input_df):
    print()
    print('Starting Ag Optimization Run seasonal ++++')
    tic()

    # Creation of a Concrete Model
    model = ConcreteModel()

    model.j = Set(initialize=['Rice1','Rice2','SugarAdsali','SugarPreseasonal','SugarSuru1','SugarSuru2','Wheat1','Wheat2',
                              'FruitVegK1','FruitVegK2','FruitVegR1','FruitVegR2','SpicesK1','SpicesK2','SpicesR1','SpicesR2',
                              'Cotton1','Cotton2','Soybean1','Soybean2','Chickpea1','Chickpea2','Groundnut1','Groundnut2',
                              'SorghumK1','SorghumK2','SorghumR1','SorghumR2','MaizeK1','MaizeK2','MaizeR1','MaizeR2',
                              'GeneralK1','GeneralK2','GeneralR1','GeneralR2','Solar'], doc='Crops with solar')

    model.jj = Set(initialize=['Rice1','Rice2','SugarAdsali','SugarPreseasonal','SugarSuru1','SugarSuru2','Wheat1','Wheat2',
                              'FruitVegK1','FruitVegK2','FruitVegR1','FruitVegR2','SpicesK1','SpicesK2','SpicesR1','SpicesR2',
                              'Cotton1','Cotton2','Soybean1','Soybean2','Chickpea1','Chickpea2','Groundnut1','Groundnut2',
                              'SorghumK1','SorghumK2','SorghumR1','SorghumR2','MaizeK1','MaizeK2','MaizeR1','MaizeR2',
                              'GeneralK1','GeneralK2','GeneralR1','GeneralR2'], doc='Crops')

    model.s = Set(initialize=['khy1','rby1','khy2','rby2'], doc='Season')
    agent_list = input_df['gw_depth'].Agent.tolist()
    model.d = Set(initialize=[str(x) for x in agent_list], doc='Agent')

    # model.d = Set(initialize=['10100','10200','10214','10226','10240','10300','10400','10410','10426','10500','10600','10609','10614',
    #                           '10626','10650','20700','20718','20729','20740','20800','30900','30923','31000','31023','31046','41100',
    #                           '41200','41300','41400','41403','41408','41422','41436','41441','41500','41516','41539','41626',
    #                           '51600','51610','51624','51625','51627','51700','51719','51724','51727','51800','51817','51821','51832',
    #                           '51849','51900','51915','51933','52000','52004','52017','52021','52035','52100','52101','52104','52121',
    #                           '52135','52145','52200','52221','52249','52300','52301','52304','52309','52319','52400','52401','52420',
    #                           '52448','52500','52520','52531','52545','52600','52615','52617','52632','52635','52649','52700','52709',
    #                           '52714','52724','52727','52800','52815','62900','62907','62930','62944','63000','63023','63038','63044',
    #                           '63100','63123','63144','63200','63244','63349','73300','73312','73333','73400','73443','73444','73447',
    #                           '73500','73512','73533','73549','83600','83700','83703','83706','83716','83734','83800','83811','83826',
    #                           '83828','83841','83900','83902','83903','83906','83942','83946','84000','84033','84046','84049','84100',
    #                           '84123','84133','84137','84146','84200','84202','84205','84206','84216','84239','84246','84300','84302',
    #                           '84346','84349','84400','84407','84423','84430','84433','84437','84444','84449','84500','84505','84513',
    #                           '84516','84539','84600','84605','84613','84646'], doc='Agent')

    # =====================================================
    ## Define parameters
    # =====================================================
    print('Setting Parameters ----')

    print('Reading land availability in 1000 ha (land_avai)')

    # land_avai------------------------------------------------------------ Availabe land for agriculture
    df_land_avai = input_df['land_avai'] # pd.read_excel(filepath, sheet_name = "land_avai")
    dict_land_avai = dict(
        zip(df_land_avai['Agent'].astype(str), df_land_avai['Land_avai']))
    model.land_avai_d_p = Param(model.d, initialize=dict_land_avai, doc='Land availability by agent (1000 ha)')

    print('Reading land requirement per season')

    # land_req_p(j,s)---------------------------------------------------Observed land by crop and agent (1000 ha)
    df_land_req = input_df['land_req']
    dict_land_req = {}
    for i in range(0, len(model.j)):
        for j in range(1, df_land_req.shape[1]):
            dict_land_req[(str(df_land_req['Crop'][i])), df_land_req.columns[j]] = df_land_req[df_land_req.columns[j]][
                i]
    model.land_req_j_s_p = Param(model.j, model.s, initialize=dict_land_req,
                                 doc='crop land requirements per season')

    print(
        'Reading groundwater depth in m (gw_depth) [d,m]')  # Note this one seems inversed it's d, m to run in the ag_module_test

    # gw_depth------------------------------------------------------------ Groundwater Depth
    df_gw_depth = input_df['gw_depth'] # pd.read_excel(filepath, sheet_name="gw_depth")
    dict_gw_depth = {}
    for i in range(0, len(model.d)):
        for j in range(1, df_gw_depth.shape[1]):
            dict_gw_depth[(str(df_gw_depth['Agent'][i]), df_gw_depth.columns[j])] = df_gw_depth[df_gw_depth.columns[j]][i]
    model.gw_table_depth_s_d_p = Param(model.d, model.s, initialize=dict_gw_depth,
                                       doc='Initial groundwater table depth by agent (m)')

    print('Reading observed land (obs_land)')

    # obs_land_j_d_p(j,d)---------------------------------------------------Observed land by crop and agent (1000 ha)
    df_obs_land = input_df['obs_land'] # pd.read_excel(filepath, sheet_name="obs_land")
    dict_obs_land = {}
    for j in range(1, df_obs_land.shape[1]):
        for i in range(0, len(model.d)):
            dict_obs_land[(df_obs_land.columns[j], str(df_obs_land['Agent'][i]))] = df_obs_land[df_obs_land.columns[j]][i]
    model.obs_land_j_d_p = Param(model.j, model.d, initialize=dict_obs_land,
                                 doc='Observed land by crop and agent (1000 ha)')

    print('Reading surface water availability (sw_avai)')

    # sw_avai_s_d_p(d,s)---------------------------------------------------Surface water availability by month and agent (km3 per month)
    df_sw_avai = input_df['sw_avai'] # pd.read_excel(filepath, sheet_name="sw_avai")
    dict_sw_avai = {}
    for i in range(0, len(model.d)):
        for j in range(1, df_sw_avai.shape[1]):
            dict_sw_avai[(str(df_sw_avai['Agent'][i]), df_sw_avai.columns[j])] = df_sw_avai[df_sw_avai.columns[j]][i]
    model.sw_avai_s_d_p = Param(model.d, model.s, initialize=dict_sw_avai,
                                doc='Surface water availability by month and agent (km3 per month)')

    print('Reading groundwater availability (gw_avai)')

    # gw_avai_s_d_p(d,s)---------------------------------------------------Groundwater availability by month and agent (km3 per month)
    df_gw_avai = input_df['gw_avai'] # pd.read_excel(filepath,sheet_name="gw_avai")
    dict_gw_avai = {}
    for i in range(0, len(model.d)):
        for j in range(1, df_gw_avai.shape[1]):
            dict_gw_avai[(str(df_gw_avai['Agent'][i]), df_gw_avai.columns[j])] = df_gw_avai[df_gw_avai.columns[j]][i]
    model.gw_avai_s_d_p = Param(model.d, model.s, initialize=dict_gw_avai,
                                doc='Groundwater availability by month and agent (km3 per month)')

    print('Reading net revenue (net_revenue)')

    # net_revenue_ha_j_p(j)--------------------------------------- net revenue not including water costs (billion RS per 1000 ha)
    df_net_revenue = input_df['net_revenue'] # pd.read_excel(filepath, sheet_name="net_revenue")
    dict_net_revenue = {}
    for i in range(len(model.j)):
        for j in range(len(model.d)):
            dict_net_revenue[(df_net_revenue['Crop'][i], str(df_sw_avai['Agent'].unique()[j]))] = \
                df_net_revenue['net_revenue'][i]
    model.net_revenue_ha_j_p = Param(model.j, model.d, initialize= dict_net_revenue,
                                     doc='net revenue not including water costs (billion RS per 1000 ha)')
    # model.net_revenue_ha_j_v_d_p.display()

    print('Reading surfacewater costs (sw_cost)')

    # sw_cost_j_d_p(j,d)--------------------------------------------------Surface water cost (billion RS per km3)
    model.sw_cost_p = Param(initialize=0, doc='Surface water cost (billion USD per km3)')
    # model.sw_cost_j_d_p.display()

    print('Reading groundwater cost (gw_cost)')

    # gw_cost_j_d_p(j,d)-------------------------------------------------groundwater water cost (billion RS per m per km3)
    model.gw_cost_p = Param(initialize=0.16, doc='groundwater water cost (billion USD per m per km3)')
    # model.gw_cost_j_d_p.display()

    print('Reading beta')

    # beta_j_d_p(j,d)---------------------------------------------------PMP beta parameter
    df_gamma = input_df['pmp_gamma'] # pd.read_excel(filepath, sheet_name="pmp_beta")
    dict_gamma = {}
    for i in range(len(model.j)):
        for j in range(len(model.d)):
            dict_gamma[(df_net_revenue['Crop'].unique()[i], str(df_gamma['Agent'].unique()[j]))] = 0
    for k in range(df_gamma.shape[0]):
        dict_gamma[(df_gamma['Crop'][k], str(df_gamma['Agent'][k]))] = df_gamma['Gamma'][k]
    model.gamma = Param(model.j, model.d, initialize=dict_gamma, mutable=True, doc='PMP gamma parameter')

    # beta_j_d_p(j,d)---------------------------------------------------PMP beta parameter
    df_alpha = input_df['pmp_alpha']  # pd.read_excel(filepath, sheet_name="pmp_beta")
    dict_alpha = {}
    for i in range(len(model.j)):
        for j in range(len(model.d)):
            dict_alpha[(df_net_revenue['Crop'].unique()[i], str(df_alpha['Agent'].unique()[j]))] = 0
    for k in range(df_alpha.shape[0]):
        dict_alpha[(df_alpha['Crop'][k], str(df_alpha['Agent'][k]))] = df_alpha['Alpha'][k]
    model.alpha = Param(model.j, model.d, initialize=dict_alpha, mutable=True, doc='PMP alpha parameter')

    # AJF comment -- this reading can sometimes take a while; unclear why
    # grs_ir_wat_req_j_s_d_p(s,j,d)-------------------------------------Monthly gross irrigation water requirements by crop and irrigation technique (km3 per month per 1000 ha)
    df_grs_ir_wat_req = input_df['crop_wat_app'] # pd.read_excel(filepath, sheet_name="crop_wat_app")
    dict_grs_ir_wat_req = {}
    for k in range(len(model.s)):
        for i in range(len(model.j)):
            for j in range(len(model.d)):
                dict_grs_ir_wat_req[(['khy1','rby1','khy2','rby2'][k],(df_net_revenue['Crop'].unique()[i], str(df_gw_depth['Agent'].unique()[j])))] = 0
    for i in range(0, len(model.jj) * len(model.d)):
        for j in range(2, df_grs_ir_wat_req.shape[1]):
            dict_grs_ir_wat_req[(
                df_grs_ir_wat_req.columns[j], df_grs_ir_wat_req['Crop'][i],
                str(df_grs_ir_wat_req['Agent'][i]))] = df_grs_ir_wat_req[df_grs_ir_wat_req.columns[j]][i] / 1000000
    model.grs_ir_wat_req_j_s_d_p = Param(model.s, model.j, model.d, initialize=dict_grs_ir_wat_req,
                                  doc='Monthly gross irrigation water requirements by crop (km3 per month per 1000 ha)')
    # model.grs_ir_wat_req_j_s_d_p.display()

    # Variables
    # tot_ir_wat_use_opt_j_s_d_v       (s,j,d)        Irrigation water use per irrigation tech and month and agent (km3)
    model.tot_ir_wat_use_opt_j_s_d_v = Var(model.s, model.j, model.d, bounds=(None, None),
                                             doc='Irrigation water use and month and agent (km3)')

    # revenue_opt_j_d_v                  (  j,d)                     Crop revenue per crop and agent (billion RS)
    model.revenue_opt_j_d_v = Var(model.j, model.d, bounds=(None, None),
                                  doc='Crop revenue per crop and agent (billion RS)')

    # prod_cost_opt_j_d_v                (  j,d)      Production costs excluding water costs per crop and agent (billion RS)
    model.prod_cost_opt_j_d_v = Var(model.j, model.d, bounds=(None, None),
                                    doc='Production costs excluding water costs per crop and agent (billion RS)')

    # sw_cost_opt_j_d_v                  (d)                SW use cost per crop and agent (billion RS)
    model.sw_cost_opt_j_d_v = Var(model.d, bounds=(None, None),
                                  doc='SW use cost per crop and agent (billion RS)')

    # gw_cost_opt_j_d_v                  ( d)         SW use cost per crop and agent (billion RS)
    model.gw_cost_opt_j_d_v = Var(model.d, bounds=(None, None),
                                  doc='GW use cost per crop and agent (billion RS)')

    # income_opt_j_d_v                   (j,  d)                       Farm income by crop and irrigation agent (billion RS)
    model.income_opt_j_d_v = Var(model.j, model.d, bounds=(None, None),
                                 doc='Farm income by crop and irrigation agent (billion RS)')

    # income_opt_d_v                     (    d)                     Total farm income in each time step (billion RS)
    model.income_opt_d_v = Var(model.d, bounds=(None, None), doc='Total farm income in each time step (billion RS)')

    # Positive Variables

    # sw_use_opt_j_s_d_v                 (s,j,d)                                      Surface water use per month and agent (km3)
    model.sw_use_opt_j_s_d_v = Var(model.s, model.j, model.d, bounds=(0.0, None),
                                 doc='Surface water use per month and agent (km3)')

    # gw_use_opt_j_s_d_v                 (s,j,d)                                      Groundwater use per month and agent (km3)
    model.gw_use_opt_j_s_d_v = Var(model.s, model.j, model.d, bounds=(0.0, None),
                                 doc='Groundwater use per month and agent (km3)')

    # land_opt_j_d_v                   (j,d)                                      Optimal land allocation to crops and irrigation tech and agent (1000 ha)
    model.land_opt_j_d_v = Var(model.j, model.d, bounds=(0.0, None),
                               doc='Optimal land allocation to crops and irrigation tech and agent (1000 ha)  ')


    # =====================================================
    ## Define constraints ##
    # =====================================================
    print('Defining Constraints ---')


    # land_opt_j_d_e                   (j,d)                             Exclude crops not included in the observed crop mix
    def land_opt_j_d_e(model, jj, d):
        if model.obs_land_j_d_p[jj, d] > 0:
            return Constraint.Skip
        else:
            return model.land_opt_j_d_v[jj, d] == 0


    model.land_opt_j_d_e = Constraint(model.jj, model.d, rule=land_opt_j_d_e, doc='Exclude crops not included in the observed crop mix ')

    #
    # land_opt_solar_d_e             ('Solar',d)                               Solar cannot be irrigated
    # land_opt_solar_d_e             ('Solar',d)..     land_opt_j_v_d_v.fx('Solar',d) = 0 ;
    def land_opt_solar_d_e(model, j, d):
        return model.land_opt_j_d_v['Solar', d] == 0

    model.land_opt_solar_d_e = Constraint(model.j, model.d, rule=land_opt_solar_d_e,
                                          doc='Solar cannot be irrigated')


    # tot_ir_wat_use_opt_j_s_d_e     (s,j,d)                  Irrigation water use per irrigation tech and month and agent
    # tot_ir_wat_use_opt_s_j_d_v  (s,j,dd)  =e= grs_ir_wat_app_j_d_s_p(s,j,dd) * land_opt_j_d_v(j,dd) ;
    def tot_ir_wat_use_opt_j_s_d_e(model, s, j, d):
        return model.grs_ir_wat_req_j_s_d_p[s, j, d] * model.land_opt_j_d_v[j, d] == \
               model.tot_ir_wat_use_opt_j_s_d_v[s, j, d]


    model.tot_ir_wat_use_opt_j_s_d_e = Constraint(model.s, model.j, model.d,
                                                    rule=tot_ir_wat_use_opt_j_s_d_e,
                                                    doc='Irrigation water use per irrigation tech and month and agent')


    # ir_wat_avai_opt_j_s_d_e            (s,j,d)                  Irrigation water availability constraint
    # tot_ir_wat_use_opt_s_j_d_v  (s,j,dd)  =e= sw_use_opt_s_j_d_v(s,j,dd) + gw_use_opt_s_j_d_v(s,j,dd) ;
    def ir_wat_avai_opt_j_s_d_e(model, s, j, d):
        return model.sw_use_opt_j_s_d_v[s, j, d] + model.gw_use_opt_j_s_d_v[s, j, d] == \
               model.tot_ir_wat_use_opt_j_s_d_v[s, j, d]


    model.ir_wat_avai_opt_j_s_d_e = Constraint(model.s, model.j, model.d, rule=ir_wat_avai_opt_j_s_d_e,
                                               doc='Irrigation water availability constraint')


    # sw_avai_opt_s_d_e                  (s,  d)                                      Surface water availability constraint
    # sum(j, sw_use_opt_s_j_d_v   (s,j,dd)) =l= 1.0 * sw_avai_s_d_p(dd,s) ;
    def sw_avai_opt_s_d_e(model, s, d):
        return sum(model.sw_use_opt_j_s_d_v[s, j, d] for j in model.j) <= model.sw_avai_s_d_p[d,s]


    model.sw_avai_opt_s_d_e = Constraint(model.s, model.d, rule=sw_avai_opt_s_d_e,
                                         doc='Surface water availability constraint')

    # # gw_avai_opt_s_d_e                  (s,  d)                                      Groundwater availability constraint
    # # sum(j, gw_use_opt_s_j_d_v   (s,j,dd)) =l= 1.0 * gw_avai_s_d_p(dd,s) ;
    # def gw_avai_opt_s_d_e(model, s, d):
    #     return sum(model.gw_use_opt_j_s_d_v[s, j, d] for j in model.j) <= model.gw_avai_s_d_p[d,s]
    #
    #
    # model.gw_avai_opt_s_d_e = Constraint(model.s, model.d, rule=gw_avai_opt_s_d_e,
    #                                      doc='Groundwater availability constraint')


    # land_constraint_opt_d_e            (    d)                                      Land availability
    # sum(j, land_opt_j_d_v(j, dd) * land_req_p(j,s)) = l = 1.0 * land_avai_d_p(dd);
    def land_constraint_opt_d_s_e(model, d, s):
        return sum(model.land_opt_j_d_v[j, d] * model.land_req_j_s_p[j, s] for j in model.j) <= model.land_avai_d_p[d]

    model.land_constraint_opt_d_s_e = Constraint(model.d, model.s, rule=land_constraint_opt_d_s_e,
                                                 doc='Land availability')


    # revenue_opt_j_d_e                  ( j,d)                                      Crop revenue per crop and agent
    # revenue_opt_j_d_v           (j,dd)    =e= net_revenue_ha_j_p(j) * land_opt_j_d_v(j,dd) ;
    def revenue_opt_j_d_e(model, j, d):
        return model.net_revenue_ha_j_p[j, d] * model.land_opt_j_d_v[j, d] == model.revenue_opt_j_d_v[j, d]


    model.revenue_opt_j_d_e = Constraint(model.j, model.d, rule=revenue_opt_j_d_e,
                                         doc='Crop revenue per crop and agent')


    # sw_cost_opt_j_d_e                  (  j,d)                                      SW use cost per crop and agent
    # sw_cost_opt_j_d_v           (  dd)    =e= sum((s,j), sw_cost_p * sw_use_opt_s_j_d_v(s,j,dd)) ;
    def sw_cost_opt_d_e(model, d):
        return sum(sum(model.sw_cost_p * model.sw_use_opt_j_s_d_v[s, j, d] for s in model.s) for j in model.j) == \
                model.sw_cost_opt_j_d_v[d]


    model.sw_cost_opt_d_e = Constraint(model.d, rule=sw_cost_opt_d_e, doc='SW use cost per crop and agent')


    # gw_cost_opt_j_d_e                  (  j,d)                                      GW use cost per crop and agent
    # gw_cost_opt_j_d_v(dd) = e = sum((s, j), gw_cost_p * gw_table_depth_d_s_p(dd, s) * gw_use_opt_s_j_d_v(s, j, dd));
    def gw_cost_opt_d_e(model, d):
        return sum(model.gw_table_depth_s_d_p[d, s] * sum(model.gw_cost_p * model.gw_use_opt_j_s_d_v[s, j, d]
                   for j in model.j) for s in model.s) == model.gw_cost_opt_j_d_v[d]


    model.gw_cost_opt_d_e = Constraint(model.d, rule=gw_cost_opt_d_e, doc='GW use cost per crop and agent')


    # income_opt_j_d_e                   (j,  d)                                     Net income by crop and irrigation agent
    # income_opt_j_d_v            (j,dd)    =e= revenue_opt_j_d_v(j,dd) - [0.5 * beta_p(j,dd) * land_opt_j_d_v(j,dd) * land_opt_j_d_v(j,dd)] ;
    def income_opt_j_d_e(model, j, d):
        return model.revenue_opt_j_d_v[j, d] - model.alpha[j,d] * model.land_opt_j_d_v[j,d] -\
               0.5 * model.gamma[j,d] * model.land_opt_j_d_v[j,d] * \
                   model.land_opt_j_d_v[j,d] == model.income_opt_j_d_v[j, d]



    model.income_opt_j_d_e = Constraint(model.j, model.d, rule=income_opt_j_d_e,
                                        doc='Net income by crop and irrigation agent')


    # income_opt_d_e                     (    d)                                      Net income in ach time step
    # income_opt_d_v           (  dd)    =e= sum(j, income_opt_j_d_v(j,dd)) - sw_cost_opt_j_d_v(dd) - gw_cost_opt_j_d_v(dd)
    def income_opt_d_e(model, d):
        return sum(model.income_opt_j_d_v[j, d] for j in model.j) - model.sw_cost_opt_j_d_v[d] - model.gw_cost_opt_j_d_v[d] \
               == model.income_opt_d_v[d]


    model.income_opt_d_e = Constraint(model.d, rule=income_opt_d_e, doc='Net income in ach time step')

    # =====================================================
    ## Define Objective and solve ##
    # =====================================================
    print('Defining Objective ----')


    def objective_rule(model):
        return sum(model.income_opt_d_v[d] for d in model.d)


    model.objective = Objective(rule=objective_rule, sense=maximize, doc='Total income over all time steps ')

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    ## Creating and running the solver:
    opt = SolverFactory("ipopt", solver_io='nl')
    opt.options['max_iter']= 1000000  # number of iterations you wish
    results = opt.solve(model,keepfiles=False, tee=False)
    print(results.solver.termination_condition)
    results.write()


    return model, results
