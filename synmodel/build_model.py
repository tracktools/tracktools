"""         
        Script : Prepare MODFLOW6 Steady State DISV refined model
"""

#------------------------------------------------------------------------------------------------------
# ----  PART 1 : SETTINGS
#------------------------------------------------------------------------------------------------------

# ---- Local settings
import os, sys
sig_dir = 'sig'
exe_dir = os.path.join('..','exe')
gridgen_exe = os.path.join(exe_dir, 'gridgen.exe')
mf6_exe = os.path.join(exe_dir, 'mf6')
mp7_exe = os.path.join(exe_dir, 'mp7')

# ---- Import packages
import numpy as np
import pandas as pd
import flopy
from flopy.utils.gridgen import Gridgen
import geopandas as gpd
from shapely.geometry import Point
from shapely import speedups
speedups.disable()
import matplotlib.pyplot as plt
import gstools as gs
from rasterio.transform import from_origin

#------------------------------------------------------------------------------------------------------
# ----  PART 2 : PREPROCESSING
#------------------------------------------------------------------------------------------------------

# ---- Set spatial model discretization
xll, yll = 0, 0         # Origin point coordinates (lower left corner)
delr = delc = 50        # Cell resolution (m)
nrow, ncol = 100, 75    # Number of rows and columns 
nlay = 1                # Number of layer
top = 40                # Cell top (m)
botm = 10               # cell bottom (m)

# ---- Build simple structured modflow model
base_name = 'base'
# -- Modflow simulation package
b_sim = flopy.mf6.MFSimulation(sim_name= 'bsim', exe_name = mf6_exe, version='mf6')
# -- Modflow groundwater flow model 
base = flopy.mf6.ModflowGwf(b_sim, modelname= base_name)
# -- Modflow spatial discretisation 
b_dis = flopy.mf6.ModflowGwfdis(base, length_units='METERS',
                                      xorigin=xll,
                                      yorigin=yll,
                                      nlay=nlay,
                                      nrow=nrow,
                                      ncol=ncol,
                                      delr=delr,
                                      delc=delc,
                                      top=top,
                                      botm=botm,
                                      idomain=1)    # All cell actives

'''
# ---- Plot base model discretisation
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
mm = flopy.plot.PlotMapView(model=base)
mm.plot_grid(linewidth=0.2)
plt.show()
'''

# ---- Create Gridgen object
g = Gridgen(b_dis, exe_name = gridgen_exe)

# ---- Fetch all shapefiles in a dictionary
shpfiles = [f for f in os.listdir(sig_dir) if f.endswith('.shp')]
shp_dic = {f.split('.')[0] : os.path.join(sig_dir,f) for f in shpfiles}

# ---- Refine basic grid (focus on bound conditions)
refine_dic = {'river' : ['line', shp_dic['river']],
              'drain' : ['line', shp_dic['drain']],
              'wells' : ['point', shp_dic['wells']]}
for ref_id, data in refine_dic.items():
        ftype, path = data
        g.add_refinement_features(path.replace('.shp',''), ftype, 2, range(nlay))
        

# ---- Build new grid from gridgen object
g.build()


# ---- Fetch DISV properties
gridprops = g.get_gridprops_disv()

# ---- Perform intersection for all boundaries with gridgen mesh

# -- Set list of boundaries to intersect with grid
inter_list = ['drain', 'chd', 'river','wells']
# -- Set geometries
geoms_dic =  {  shp_dic['drain']: 'line',
                shp_dic['chd']: 'line',
                shp_dic['river']: 'line',
                shp_dic['wells']: 'point' }

# -- Create main dictionary : icpl_dic (icpl = i-cell per layer)
''' This dictionary will contain all the cell (nodenumber) intersected 
    by each required boundary (river, drain, wells,...) '''

icpl_dic = {}

for inter_id in inter_list:
    # ---- Fetch shp path without '.shp'
    shp_id = shp_dic[inter_id].replace('.shp','')
    # ---- Fetch shp type
    feat_type = geoms_dic[shp_dic[inter_id]]
    # ---- Perform intersection and convert into dataframe
    df = pd.DataFrame(g.intersect(shp_id, feat_type, nlay-1))
    # ---- Drop possible duplicated node during intersection
    df.drop_duplicates(subset='nodenumber', keep='first', inplace=True, ignore_index = True)
    # ---- Decode the 'ID' column into utf-8 
    df['FID'] = [ x.decode('utf-8') for x in df['FID']]
    # ---- Fetch ID column  categories
    feat_ids = pd.Categorical(df['FID']).categories
    # ---- Add to main dictionary
    icpl_dic[inter_id] = { feat_id : df.loc[df['FID'] == feat_id,'nodenumber'].tolist() 
                           for feat_id in feat_ids }

# -- Create heterogenous hydraulic conditivity field using Unconditional
#     Sequence Gaussian Simulation with an exponential variogram with 500m range

# -- Build variogram model
vgm = gs.Exponential(dim=2, var=0.1, len_scale=500, Nugget=0, anis=1.)
srf = gs.SRF(vgm, mean=0, seed=50)

#------------------------------------------------------------------------------------------------------
# ----  PART 3 : BUILD AND RUN MODFLOW6 MODEL
#------------------------------------------------------------------------------------------------------

# ---- Set model name
name = 'syn_model'

# ---- Modflow simulation package
sim = flopy.mf6.MFSimulation(sim_name= name, exe_name = mf6_exe, version='mf6')

# ---- Time discretization package
''' Set steady-state model with nper = 1 '''
perioddata = [ ( 1, 1, 1) ]
tdis = flopy.mf6.ModflowTdis(sim, time_units = 'seconds', nper=1, perioddata = perioddata)

# ---- Modflow groundwater flow model 
gwf = flopy.mf6.ModflowGwf(sim, modelname= name)

# ---- Iterative Model Solution (IMS) package
ims = flopy.mf6.ModflowIms(sim)
sim.register_ims_package(ims, [gwf.name])

# ---- Spatial discretization
disv = flopy.mf6.ModflowGwfdisv(gwf, length_units = 'METERS',
                                     xorigin      = xll,
                                     yorigin      = yll, 
                                     nlay         = gridprops['nlay'],
                                     ncpl         = gridprops['ncpl'],
                                     nvert        = gridprops['nvert'],
                                     vertices     = gridprops['vertices'],
                                     cell2d       = gridprops['cell2d'],
                                     top          = gridprops['top'],
                                     botm         = gridprops['botm'])

# ---- Set initial conditions
ic = flopy.mf6.ModflowGwfic(gwf, strt=30)

# ---- NPF package (Node Property Flow)
xc, yc, zc = map(np.array, gwf.modelgrid.xyzcellcenters)
logk = srf.unstructured([xc, yc]) + 4.2
k = np.power(10,-logk)

npf = flopy.mf6.ModflowGwfnpf(gwf,   icelltype  = 0,           
                                     k          = k,
                                     save_flows = True,
                                     save_specific_discharge = True)  

# ---- STO package
S = 0.008   # Storage coeficient (not required in steady-state mode)
sto = flopy.mf6.ModflowGwfsto(gwf,   steady_state = {0 : True}, # Define steady model
                                     ss           = S, 
                                     save_flows   = True,
                                     iconvert     = 0)          # 0, non-convertible storage

# ---- CHD package

# Build ghb stress period data: [cellid, shead, ehead, aux, boundname], used format cellid = (lay,node)
head = 20   # Imposed head
chd_spd = []
for chd_id, nodes in icpl_dic['chd'].items():
    for node in nodes:
        cell_chd_data = [(0, node), head, chd_id]                 
        chd_spd.append(cell_chd_data)  

# Build GHB package
chd = flopy.mf6.ModflowGwfchd(gwf,   stress_period_data = chd_spd,          
                                     maxbound           = len(chd_spd),
                                     boundnames         = True,
                                     save_flows         = True)

# ---- Drain package

# Set drain stage and conductance (constant along all the drain cell)
n_drn = 26      # m
c_drn = 5e-3    # m²/s

# Build drain stress period data: [cellid, elev, cond, aux, boundname], used format cellid = (lay,node)
drn_spd = []
for drn_id, nodes in icpl_dic['drain'].items():
    for node in nodes:
        cell_drain_data = [(0,node), n_drn, c_drn, drn_id]
        drn_spd.append(cell_drain_data)

# Build DRN package
drn = flopy.mf6.ModflowGwfdrn(gwf,   stress_period_data = drn_spd,
                                     maxbound           = len(drn_spd),
                                     save_flows         = True,        
                                     boundnames         = True)

# ---- RIVER package

# Set river conductance and hydraulic slope 
c_riv = 1e-3   # m²/s
# hslope = 0.002 # 2 °%

# Load river shapefile as GeoDataFrame
riv_gdf = gpd.read_file(shp_dic['river'])
# Set reaches as index
riv_gdf.set_index('FID', inplace = True)

# Build riv stress period data: [cellid, stage, cond, rbot, aux, boundname], used format cellid = (lay,node)
riv_spd = []
for reach_id, nodes in icpl_dic['river'].items():
        # Get river stages upstream and downstream
        h_up, h_down = riv_gdf.loc[reach_id,['h_up', 'h_down']].astype(np.float)
        riv_line = riv_gdf.loc[reach_id,'geometry']
        # Get river stage for each cell as stage = xi * h_up + (1 - xi) * h_down
        # where xi is the normalized curvilinear distance (between 0and 1) of 
        # the projection of the cell center on the reach
        for node in nodes:
                xc, yc = g.get_center(node)
                xi = (riv_line.length - riv_line.project(Point(xc,yc))) / riv_line.length
                stage = xi * h_up + (1 - xi) * h_down
                zbot = stage - 2   # zbot is set to 2meters below the river stage
                cell_riv_data = [(0, node), stage, c_riv, zbot, reach_id]
                riv_spd.append(cell_riv_data)



# Build RIV package
riv = flopy.mf6.ModflowGwfriv(gwf,   stress_period_data = riv_spd,
                                     maxbound           = len(riv_spd),
                                     save_flows         = True,       
                                     boundnames         = True)

# ---- WEL Package

q_well_dic = {k:v for k,v in zip(icpl_dic['wells'].keys(), [110,80])}  # pumping rates in m3/h

# Build wel stress period data: [cellid, q, aux, boundname], used format cellid = (lay,node)
well_spd = []
for well_id, nodes in icpl_dic['wells'].items():
        q = - q_well_dic[well_id] / 3600   # transform m3/h to m3/s
        node = nodes[0]
        cell_well_data = [(0, node), q, well_id]                  
        well_spd.append(cell_well_data)

wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, stress_period_data = well_spd,
                                                    maxbound           = len(well_spd),
                                                    save_flows         = True,       
                                                    boundnames         = True)


# -- RCH Package
rpy = 300  # mm/y
rech = rpy / ((365*24*3600) * (1000)) 
rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge = rech, save_flows = False)



# -- OC Package
oc_rec_list =[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
printrecord = [('HEAD', 'LAST')]

oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf,  saverecord        = oc_rec_list, 
                                                  head_filerecord   = [name + '.hds'],
                                                  budget_filerecord = [name + '.cbc'],
                                                  printrecord       = printrecord,
                                                  pname             = 'oc' )

# ---- Write model simulation files
sim.write_simulation()

# ---- Run model simulation
success, buff = sim.run_simulation()



'''
#------------------------------------------------------------------------------------------------------
# ----  PART 4 : PLOT SIMULATED HEAD
#------------------------------------------------------------------------------------------------------

# ---- Set required layer
ilay = 0

# ---- Fetch heads data
hds = sim.simulation_data.mfdata[gwf.name,'HDS','HEAD'][-1,ilay,-1,:]

# ---- Set plot figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect = 'equal')
ax.tick_params(axis='both', which='major', labelsize=9)

# ---- Build MapPlotView object
pmv = flopy.plot.PlotMapView(model=gwf, ax=ax)

# ---- Plot head data
heads = pmv.plot_array(hds, masked_values=[1.e+30], cmap ='viridis', alpha=0.7)
cb = plt.colorbar(heads, shrink = 0.5)

# ---- Plot contours
hmin, hmax = hds.min(), hds.max()
levels = np.arange(hmin, np.ceil(hmax), 1)
contours = pmv.contour_array(hds, masked_values=[1.e+30], 
                                  levels=levels, linewidths = 0.4,
                                  linestyles = 'dashed', colors = 'black')
contours.clabel(fmt = '%1.1f', inline=True, colors = 'black', inline_spacing = 20, fontsize = 6)

# ---- Set boundaries colors
bc_colors_dic = { 'RIV': 'cyan', 'DRN': 'red', 'CHD': 'navy', 'WEL': 'coral'}
for bc in bc_colors_dic.keys():
        quadmesh = pmv.plot_bc(bc, color = bc_colors_dic[bc])
        
# ---- Set title
ax.set_title('Model Layer {}; hmin={:6.2f}, hmax={:6.2f}'.format(ilay + 1, hmin, hmax), fontsize = 12)

# ---- Plot grid
pmv.plot_grid(lw = 0.1)

plt.show()
'''
