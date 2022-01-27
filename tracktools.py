import os
import numpy as np
import warnings


try : 
    import pandas as pd
except ImportError :
    print('Could not load pandas...')


# TODO : fix distance function for vertices  
# TODO : check structured grid support 
# TODO : check generation from geometries (Pierre)
# TODO : update zone vulnerability 

class ParticleGenerator():
    """ 
    Particle tracking pre-processor. 
    Generates particles around features 
    for a given set of features (geometries or shapefile items)
    and a groundwater flow model (ModflowGwf instance)
    Methods
    -----------
    gen_points() # missing relative distance 
    get_particlegroup()
    
    Missing:
    to_shapefile()
    Attributes
    -----------
    """
    def __init__(self, ml):
        """
        Parameters
        -----------
        ml   (flopy.mf6.modflow.mfgwf.ModflowGwf) : mf6 ground water flow model
        
        Examples
        -----------
        >>> pb = ParticleBuilder('WELLS', ml, g)
        """
        try : 
            from flopy.mf6.mfmodel import MFModel
            from flopy.utils.gridintersect import GridIntersect
        except ImportError :
            print('Could not load FloPy modules')
            return

        try : 
            import geopandas as gpd
        except ImportError :
            print('Could not load geopandas...')


        assert isinstance(ml,MFModel)
        
        self.ml = ml

        # instantiate a FloPy GridIntersect object
        self.gi = GridIntersect(self.ml.modelgrid)
        
        # model grid vertices
        self.vxs, self.vys, self.vzs = self.ml.modelgrid.xyzvertices

        # initialize particle data as list, will be converted to
        # geopandas df after first particle generation 
        self.particledata = None


    def _gen_points_in_polygon(self, n_point, polygon, tol = 0.1):
        """
        Description
        -----------
        Generate n regular spaced points within a shapely Polygon geometry
        
        Parameters
        -----------
        - n_point (int) : number of points required
        - polygon (shapely.geometry.polygon.Polygon) : Polygon geometry
        - tol (float) : spacing tolerance (Default is 0.1)
        
        Returns
        -----------
        - points (list) : generated point geometries
        
        Examples
        -----------
        >>> geom_pts = gen_n_point_in_polygon(200, polygon)
        >>> points_gs = gpd.GeoSeries(geom_pts)
        >>> points_gs.plot()
        """
        try : 
            from shapely.geometry import Point
        except ImportError : 
            print('Could not load shapely module')
            return
        
        try : 
            import geopandas as gpd
        except ImportError :
            print('Could not load geopandas...')

        # Get the bounds of the polygon
        minx, miny, maxx, maxy = polygon.bounds    
        # ---- Initialize spacing and point counter
        spacing = polygon.area / n_point
        point_counter = 0
        # Start while loop to find the better spacing according to tolérance increment
        while point_counter <= n_point:
            # --- Generate grid point coordinates
            x = np.arange(np.floor(minx), int(np.ceil(maxx)), spacing)
            y = np.arange(np.floor(miny), int(np.ceil(maxy)), spacing)
            xx, yy = np.meshgrid(x,y)
            # ----
            pts = [Point(X,Y) for X,Y in zip(xx.ravel(),yy.ravel())]
            # ---- Keep only points in polygons
            points = [pt for pt in pts if pt.within(polygon)]
            # ---- Verify number of point generated
            point_counter = len(points)
            spacing -= tol
        # ---- Return
        return points


    def _get_local_xy(self, gx, gy, cell_vertices):
        """
        Description
        -----------
        Returns local (x, y) coordinates from global (gx, gy) coordinates
        
        Parameters
        -----------
        - gx (float) : Point global x-coordinates 
        - gy (float) : Point global y-coordinates
        - vertices (list) : Vertices of model cell
                            Format : [(vx0, vy0), (vx1, vy1), (vx2, vy2), (vx3, vy3),(vx0, vy0)]
                            Usefull to improve eficiency if used iterativly without gridgen object 
        Returns
        -----------
        [lx, ly] = (list) : local coordinates
        
        Examples
        -----------
        >>> pb = ParticleBuilder('WELL', ml, g, 'well.shp', 'NAME')
        >>> well_lx, well_ly = pb.get_local_xy(well.node, well.x, well.y)
        """
        # ---- Fetch coordinates of origin point
        vx0, vy0 = cell_vertices[0]
        vx1, vy1 = cell_vertices[2]
        # ---- Get resolution
        delc = vx1 - vx0
        delr = vy0 - vy1
        # ---- Calculate local corrdinates
        lx = (gx - vx0)/delc
        ly = (gy - vy1)/delr
        # ---- Return list of xy local coordinates
        return([lx, ly])




    def gen_points(self, features, dist=None, n = 100, id_field = 'fid', 
                   fids = None, gen_type='around', export = None):
        """
        Description
        -----------
        Generate points around/within provided features
        
        Parameters
        -----------
        features: str or dict of geometries 
            features can be either a shapefile or
            dictionary with feature IDs as keys and geometries as items.
        dist: float 
            seeding distance of generated points, in geometry length unit
            If None, infer safe distance from geometry according to intersected
            cell resolution : 
                dist = inter_cell_resolution * sqrt(2) (for rectangular grid)
        n: int
            number of points
            Default is 100.
        fid: str or list of str
            feature ids to consider
            If None, all features are considered
        export: str
            path to export generated points as shapefile
            If None, nothing is exported
            Default is None.
                         
        Examples
        -----------
        >>> pg = ParticleGenerator(ml)
        >>> pg.gen_points(dist = 5, n = 500, fids = 'WELL1')
        """
        try : 
            import geopandas as gpd
        except ImportError :
            print('Could not load geopandas...')

        # load feature geometries 
        if isinstance(features, str) :
            gdf = gpd.read_file(features)
            gdf['fid'] = gdf[id_field]
        elif isinstance(features, dict):
            gdf = gpd.GeoDataFrame(
                    {'fid' : geometry_dic.keys()},
                    geometry = list(geometry_dic.values())
                    )
        else :
            print('Check type of features.\
                    Should be str with shp filename or dict of geometries.')

        gdf.set_index('fid', drop=False, inplace = True)
        
        # subset by feature ids 
        if fids is not None :
            if not isinstance(fids, list): fids = [fids]
            gdf = gdf.loc[gdf[id_field].isin(fids)]

        # building distance dictionary
        if dist is None:
            dist_dic = self._infer_dist_dic(gdf)
        else:
            dist_dic = {fid : dist for fid in gdf[id_field]}

        # initialize output list of gdf(s) containing
        # generated points for each feature 
        point_gdf_list = []
        
        # ---- iterate over features
        for fid in gdf.index:

            # feature geometry
            geom = gdf.loc[fid,'geometry']

            # buffer around feature geometry
            buff = geom.buffer(dist_dic[fid])

            # generate points 
            if gen_type == 'around':
                # keep external boundary as LineString object
                line = buff.boundary
                # find the distances from origin between starting points along the line
                distances = np.linspace(0, line.length,n)
                # starting point atr each distance
                point_list = [line.interpolate(distance) for distance in distances]
            elif gen_type == 'within':
                point_list= self._gen_points_in_polygon(n, buff, tol)[:n]
            
            # iterate over points and compute local coordinates
            # in model grid cell coordinate system
            nodes, lxs, lys = [],[],[]
            for point in point_list:
                node, sp_vertices, points_geom  = self.gi.intersect(point)[0]
                vertices = [(vx,vy) for vx, vy in zip(self.vxs[node],self.vys[node])]
                vertices.append(vertices[0])
                lx, ly = self._get_local_xy(points_geom.x, points_geom.y, vertices)
                nodes.append(node)
                lxs.append(lx)
                lys.append(ly)
            # build gdf for current feature
            point_gdf = gpd.GeoDataFrame({'gid':fid,'node':nodes,
                'lx':lxs, 'ly':lys},
                geometry = point_list)
            point_gdf['pid'] = np.arange(len(point_gdf))
            point_gdf_list.append(point_gdf)

        # concatenate starting points of all feature
        out_gdf = pd.concat(point_gdf_list, ignore_index=True)

        # Add particle data
        if self.particledata is None :
            self.particledata = out_gdf

        else :
            # Manage existing feature (avoid duplicates)
            for fid in out_gdf['gid'].unique():
                if fid in self.particledata['gid']:
                    # -- Raise warning message
                    warn_msg = f'Warning : feature `{fid}` already exist. ' \
                               'It will be overwrited.'
                    warnings.warn(warn_msg, Warning)
                    # -- Remove existing set of observation
                    self.remove_particledata(fid)

            # Update particle data
            self.particledata = pd.concat([self.particledata, out_gdf],
                                          ignore_index=True)

        # export points to shapefile if required
        if export is not None:
            out_gdf.to_file(export)


    def remove_particledata(self, fids=None, verbose=False):
        """
        Description
        -----------
        Remove particle data from ids.
        
        Parameters
        -----------
        fids : str, list of str
            features names to removed from particledata.
            If None, all particledata are removed.
            Default is None.

        verbose : bool
            print removed features names.
        
        Examples
        -----------
        >>> pg = ParticleGenerator(ml = ml)
        >>> pg.gen_points('wells.shp', dist=d, n_part = n_part)
        >>> pg.removed_particledata(fids = ['well1', 'well2'], verbose =True)
        """
        if fids is None:
            self.particledata = None
            if verbose:
                print('All features had been removed.')
        else:
            _fids = [fids] if isinstance(fids, str) else fids
            self.particledata = self.particledata.query('gid not in @_fids')
            if verbose:
                print('\n'.join([f'Feature `{gid}` had been removed.' for gid in _fids]))


    def get_particlegroups(self, pgids=None, pgid_file=None):
        """
        Description
        -----------
        Get list of FloPy ParticleGroup from particledata
        Optionally writes group id file for post-processing.

        Parameters
        -----------
        pgids : list of str
            particle group id or list of group ids
        pgid_file : str
            path to create a simple text file with particle 
            groups names and and numerix ids.

        Returns
        -----------
        List of flopy.modpath.mp7particlegroup.ParticleGroup
        
        Examples
        -----------
        >>> pg = ParticleGenerator(ml = ml)
        >>> pg.gen_points('well.shp', dist=d, n_part = n_part)
        >>> particlegroups = pg.get_particlegroups()
   
        """
        try : 
            from flopy.modpath import ParticleData, ParticleGroup
        except ImportError :
            print('Could not load flopy.modpath module')

        # when pgids is None, all particle groups considered
        if pgids is None:
            # all groups considered 
            pgids = list(set(self.particledata.gid))
        
        # when a single pgid is provided
        if not isinstance(pgids, list):
            pgids = [pgids]
        
        # initialize list of particle groups
        pgrp_list = []

        # iterate of particle group ids 
        for pgid in pgids:
            # subset by group id(s)
            subset = self.particledata[self.particledata['gid']==pgid]
            if len(subset)==0:
                print(f'No particle for group id {pgid}')
                continue
            # set up ParticleData
            cols = ['node','lx','ly','pid']
            locs, lxs, lys, pids = [subset[col].tolist() for col in cols]
            pdata = ParticleData(partlocs = locs, structured=False,
                                 timeoffset = 0., drape=0,
                                 localx=lxs, localy=lys, particleids = pids)
            # set up ParticleGroup
            pg = ParticleGroup(particlegroupname= pgid, particledata=pdata)
            pgrp_list.append(pg)
        
        # write particle group id csv file 
        if pgid_file is not None :
            pd.DataFrame(
            {'name':[pgrp.particlegroupname 
                for pgrp in pgrp_list]}
            ).to_csv(pgid_file,header=False)

        return pgrp_list
    







class TrackingAnalyzer():
    """ 
        Particle tracking post-processor 
    -----------
    Arguments
    -----------
    - ml
    - mpsim
    -----------
    Methods
    -----------
    - compute_mixing_ratio()
    """
    
    def __init__(self, endpoint_file=None, pathline_file=None, 
            cbc_file = None, grb_file = None,
             mpsim=None, ml=None, precision = 'double'):
        """
        Description
        -----------
            (NOTE: only available for steady-state simulation on
             DISV spatial discretization with backward particle tracking)
        
        Parameters
        -----------
        - ml   (flopy.mf6.modflow.mfgwf.ModflowGwf) : mf6 ground water flow model
        - mpsim (flopy.modpath.mp7sim.Modpath7Sim) : mp7 simulation
        - rivfile (str) : path to a modflowriv package to
                          read externally if required.
                          Default is None
        - precision (str) : precision of the floating point data of the cbc file
                          Can be 'simple' or 'double'
                          (Default is 'double')
        Examples
        -----------
        >>> ta = TrackingAnalyzer(ml, mpsim)
        """

        try :
            from flopy.utils import EndpointFile, PathlineFile, CellBudgetFile
            from flopy.mf6.utils import MfGrdFile
            
        except ImportError :
            print('Could not load flopy modules.')
            return
        
        # initialise flowmodel to None
        self.ml, self.mpsim = ml, mpsim

        # initialize river name dictionary  
        self.pgrpname_dic, self.rivname_dic = None, None

        if  ml is not None and mpsim is not None :
            # Assuming that the model is steady-stade
            assert ml.nper == 1, 'The flow model must be in steady-state conditions'
            # Assuming that the model spatial dscretisation is DISV like
            assert ml.get_grid_type().name == 'DISV', 'The spatial discretisation must be DISV like'
            # ---- Assuming that the groudwater flow model contain RIV package
            assert ml.get_package('RIV'), 'The flow model does not contains river condition'
            # ---- Assuming that each river cell contains a boundanme (reach name)
            msg = 'boundnames must be specified for each river cells in RIV stress period data'
            boundnames = ml.get_package('RIV').stress_period_data.get_data(0)['boundname']
            assert all([x is not None for x in boundnames]), msg
            grb_file = os.path.join(ml.model_ws,
                           '{}.{}.grb'.format(ml.name,
                                              ml.get_grid_type().name.lower()))
            cbc_name = ''.join(ml.get_package('OC').budget_filerecord.get_data().budgetfile)
            cbc_file = os.path.join(ml.model_ws, cbc_name)
            endpoint_file = os.path.join(ml.model_ws, mpsim.endpointfilename)
            pathline_file = os.path.join(ml.model_ws, mpsim.pathlinefilename)
            # ---- Assuming that the particle tracking is backward oriented
            assert mpsim.trackingdirection == 2, 'The Particle tracking has to be backward'
            # get particle group data 
            self.pgrpname_dic = {i:pgrp.particlegroupname for i, pgrp in enumerate(mpsim.particlegroups)}

        # fetch endpoint file
        assert endpoint_file is not None, 'No endpoint file, provide mpsim or endpoint_file'
        self.edp = EndpointFile(endpoint_file)
        # fetch pathline file
        assert pathline_file is not None, 'No pathline file, provide mpsim or pathline_file'
        self.pth = PathlineFile(pathline_file)
        # fetch cbc file 
        assert cbc_file is not None, 'No cbc file, provide ml or cbc_file'
        self.cbc = CellBudgetFile(cbc_file, precision = precision)
        # fetch grb file 
        assert grb_file is not None, 'No grd file, provide ml or grd_file'
        bgf = MfGrdFile(grb_file) 
                
        
        # river leakage from cbc
        self.riv_leak_df = pd.DataFrame(self.cbc.get_data(text='RIV')[0])
        self.riv_leak_df['node'] = self.riv_leak_df['node'] - 1 # back to 0-based
        self.riv_leak_df.set_index('node', drop=False, inplace=True)
        
        # river cells 
        self.riv_cells = self.riv_leak_df.index.values

        # inter-cell flows (FLOW-JA-FACE)
        self.flowja = self.cbc.get_data(text='FLOW-JA-FACE')[0][0, 0, :]
        
        # IA information from binary grid file
        self.ia = bgf._datadict['IA'] - 1


    def get_part_velocity(self):
        '''
        -----------
        Description
        -----------
        Get particle velocity from Pathline file
        '''
        # load pathline data as a list of nd-array
        # (one array per particle trajectory)
        alldata = self.pth.get_alldata()

        # sort pathline data by rising pid and time 
        spdata = [ pdata[np.lexsort((pdata['particleid'], pdata['time']))] for pdata in alldata]

        # get index of time = 1.0
        idx = [ int(np.argwhere(pdata['time']==1.)) for  pdata in spdata]

        # get particle ids, dt, dx, dy, v
        pid = np.array([pdata['particleid'][i]  for i, pdata in zip(idx,spdata)])
        dt = np.array([pdata['time'][i+1] -1. for i, pdata in zip(idx,spdata)])
        dx = np.array([ pdata['x'][i+1] - pdata['x'][i]  for i, pdata in zip(idx,spdata)])
        dy = np.array([ pdata['y'][i+1] - pdata['y'][i]  for i, pdata in zip(idx,spdata)])
        v = np.sqrt(dx**2+dy**2)/dt

        v_df = pd.DataFrame({'pid':pid,'v':v})
        v_df.set_index('pid', inplace=True)

        return(v_df)


    def get_cell_inflows(self, node):
        '''
        Description
        -----------
        Sum of inter-cell inflows for given node 
        '''

        inflows = []
        for ipos in range(self.ia[node]+1, self.ia[node+1]):
            flow = self.flowja[ipos]
            if flow > 0 : inflows.append(flow)
        return(np.array(inflows).sum())

 

    def load_pgrp_names(self, pgrp_file=None):
        '''
        Description
        -----------
        Fetch and store particle group names from:
            - modpath7 simulation object (internal, recommended)
            - `pgrp_file` if provided (external, recommended)


        Parameters
        -----------
        - pgrp_file (str): path to a external text file (.txt, .csv, ..)
                           without headers, comma separated.
                          Format:   
                                0, pg_name0
                                1, pg_name1
                                3, pg_name2
                                ...
                          Default is None.


        Returns
        -----------
        - pgrpname_dic (dict) : store particle group names with their numerical id
                                Format :
                                    {pgid_0: pg_name0, pgid_1: pg_name1, ..}

        Examples
        -----------
        >>> ta = TrackingAnalyzer(ml, mpsim)
        >>> reach_dic = ta.load_pgrp_names()

        '''
        # -- from internal modpath simulation 
        if self.mpsim is not None:
            self.pgrpname_dic = {i:pg.particlegroupname
                                    for i, pg in enumerate(mpsim.particlegroups)}
        
        # -- from external particle group file
        elif pgrp_file is not None:
            pg_df = pd.read_csv(pgrp_file, header=None, names=['id','name'])
            self.pgrpname_dic = {gid:gnme for gid,gnme in zip(pg_df.id, pg_df.name)}
        
        # -- generate default
        else:
            gids = np.unique(self.edp.get_alldata()['particlegroup'])
            self.pgrpname_dic = {gid: f'PG{gid}' for gid in gids}





    def load_rivname_dic(self, riv_file=None, mfriv_file=None):
        '''

        Description
        -----------
        Fetch and store boundary condition name (ex. river reach) 
        (boundname) with related river node as list from:
            - modflow river object (internal, recommended)
            - `riv_file` if provided (external, recommended)
            - `mfriv_file` if provided (external, not recommended)

        When bc_names are provided and bc ids are given as auxiliary 
        variable in the cbc, mixing ratios can be provided with name as labels 

        Parameters
        -----------
        - riv_file (str): path to a external text file (.txt, .csv, ..)
                          without headers, comma separated.
                          Format:   
                                234, river_name1
                                235, river_name1
                                456, river_name2
                                ...
                          Default is None.

        - mfriv_file (str, deprecated): path to a external modflow6 river
                                        package file (.riv)
                                        Default is None.

        Returns
        -----------
        - reac_dic (dict) : store river reaches with river nodes
                            Format :
                                {reach_name0: [node_0, node_1, ..],
                                 reach_name1: [node_0, node_1, ..]}

        Examples
        -----------
        >>> ta= TrackingAnalyzer(ml, mpsim)
        >>> reach_dic = ta.get_reach_dic()

        '''
        # ---- Check input access to river data
        err_msg = 'ERROR : Could not access river data.'
        assert any(obj is not None for obj in [self.ml, riv_file, mfriv_file]), err_msg

        # ---- river DataFrame from external river package
        if self.ml is not None:
            riv_df = pd.DataFrame(self.ml.riv.stress_period_data.get_data(0))
            riv_df['node'] = riv_df['cellid'].apply(lambda cid: cid[1])

        # ---- Read data from external text file
        else:
            cols = ['node', 'boundname']

            if riv_file is not None:
                riv_df = pd.read_csv(riv_file, header=None, names=cols)

            elif mfriv_file is not None:

                with open(mfriv_file,'r') as f:
                    c = f.read()

                inter_lookup = ['BEGIN period  1\n',
                                'END period  1\n' ]

                if all(s in c for s in inter_lookup):
                    start, end = inter_lookup
                    lines = c[c.index(start)+len(start) : c.index(end)].splitlines()
                else:
                    lines = c.splitlines()

                # -- River DataFrame
                data = [np.array(l.strip().split()) [[1,-1]] for l in lines]
                riv_df = pd.DataFrame(data, columns = cols)
                riv_df = riv_df.astype({c:dt for c,dt in zip(cols,[int,str])})

        # ---- Convert to dictioanry
        self.rivname_dic = riv_df.groupby('boundname').apply(
                                    lambda r: r.node.tolist()
                                        ).to_dict()





    def compute_mixing_ratio(self, on='river', orient='source'):
        """
        -----------
        Description
        -----------
        Compute the mixing ratio between river water and ground water at a 
        given water production unit

        -----------
        Parameters
        -----------
        - on (str/dict) : information to compute mixing ratios.
                          Can be a:
                          - keyword :
                                - 'river': compute mixing ratios for all river cells
                                - 'reach': compute mixing rations by river reaches

                          - aggregation dictionary:
                                Format: {'reach_group1' : ['reach1', 'reach2',..],
                                         'reach_group2' : 'reach4'} )
                           Default is 'river'.
        - orient (str): orientation of the resulting mixing ratio DataFrame.
                        Can be :
                            - 'source': the water sources as columns
                            Format :
                                           src        river    others
                                        grpnme                    
                                           r20     0.129836  0.870164
                                           r21     0.130381  0.869619

                            - 'particle': the particle groups as columns
                            Format :
                                     grpnme          r20      r21
                                        src                    
                                      river     0.129836  0.130381
                                     others     0.870164  0.869619

                            Default is 'source'.

        Returns
        -----------
        - mr_df  (DataFrame) : computed mixing ratios

        Examples
        -----------
        >>> ta= TrackingAnalyzer(ml, mpsim)
        >>> ta.load_rivname_dic(riv_file='rivnames.csv')
        >>> mr_df = ta.compute_mixing_ratio(on='reach')

        """
        # fetch endpoint data
        edp_df = pd.DataFrame(self.edp.get_alldata())
        edp_df = edp_df.astype({'node': int})

        # add particle group name
        if self.pgrpname_dic is None:
            warn_msg = 'Particle group names not loaded yet. ' \
                       'Consider using .load_pgrp_names(), ' \
                       'default names will be used instead.'
            warnings.warn(warn_msg, Warning)
            self.load_pgrp_names()

        edp_df['grpnme'] = [self.pgrpname_dic[gid] 
                                 for gid in edp_df.particlegroup.values]

        # identify endpoints in river cells
        edp_df['endriv'] = edp_df.node.apply(
                lambda n: n in self.riv_cells)

        # add river leakage 
        edp_df['riv_leak'] = 0.
        edp_df.loc[edp_df.endriv,'riv_leak'] = edp_df.loc[
                edp_df.endriv,
                'node'].apply(
                        lambda n: self.riv_leak_df.loc[n,'q'].sum())

        # compute cell inflows from cbc
        edp_df.loc[edp_df.endriv,'rivcell_q'] = edp_df.loc[edp_df.endriv,
                'node'].apply(self.get_cell_inflows)


        # add particle velocity and merge value into edp_df
        v_df = self.get_part_velocity()
        edp_df.loc[edp_df.particleid,'v'] = v_df.loc[edp_df.particleid,'v']

        # compute particle mixing ratio
        edp_df['alpha'] = edp_df['riv_leak']/ (edp_df['riv_leak']+edp_df['rivcell_q'])
        edp_df.loc[edp_df.alpha.isnull(),'alpha']=0.

        # Grouped weighted average of mixing ratios
        # When they are provided, results are labeled with particle group names 
        # and bc id names, rather than integer ids.

        if on == 'river':
            # ---- Set basic river names
            edp_df['src'] = edp_df.endriv.replace({True: 'river', False: 'others'})
            
        else:
            if self.rivname_dic is None:
                try:
                    self.load_rivname_dic()
                except:
                    raise Exception('River names not loaded yet. ' \
                                    'Consider using .load_rivname_dic().')
            
            # -- Set river names by reaches
            for rivname, rivnodes in self.rivname_dic.items(): 
                edp_df.loc[edp_df.node.isin(rivnodes), 'src'] = rivname
            edp_df.fillna('others',inplace=True)
            
            # -- Aggregating reaches
            agg = {} if on == 'reach' else on
            for k,v in agg.items(): edp_df['src'].replace(v,k, inplace=True)

        # -- Compute composite mixing ratios
        mr = edp_df.groupby(
                ['grpnme', 'src'], dropna=False).apply(
                                    lambda d: np.average(d.alpha, weights=d.v))

        # -- Return mixing ratios as DataFrame
        mr_df = mr.unstack(level=0, fill_value=0)
        mr_df.loc['others'] = 1 - mr_df.sum()


        if orient == 'particle':
            return mr_df
        elif orient == 'source':
            return mr_df.T










class SSZV():
    """ 
        Class to define zonal vulnerability object from a backward 
        steady-state particle tracking 
    -----------
    Arguments
    -----------
    - ml
    - mpsim
    - id_field
    - shp_name
    - geometry_dic
    -----------
    Methods
    -----------
    - get_fids()
    - numbool2percent()
    - get_zonal_vulnerability()
    - get_methods()
    - get_pvelocity()
    - get_numbool()
    - plot_venn()
    """
    
    def __init__(self, ml, mpsim, id_field = 'FID', shp_name = None, geometry_dic = None):
      """
      -----------
      Description
      -----------
      Constructor of SSZV instance
      (NOTE: only available for steady-state simulation on
             DISV spatial discretization with backward particle tracking)
      -----------
      Parameters
      -----------
      - self  (vulnerability.SSZV)
      - ml   (flopy.mf6.modflow.mfml.ModflowGwf) : mf6 ground water flow model
      - mpsim (flopy.modpath.mp7sim.Modpath7Sim) : mp7 simulation 
      - id_field (str) : shapefile field that contains geometry ids
                         (Default is 'FID')
      - shp_name (str) : name of a required shapefile that contains polygon geometries
                         (Default is None)
      - geometry_dic (dict) : geometry information into a dictionary
                              (format: { 'geom1' : shapely.geometry.Polygon})
                              (Default is None)
      -----------
      Returns
      -----------
      - SSZV instance
      -----------
      Examples
      -----------
      >>> sszv = SSZV(ml, mpsim, id_field = 'ID', shp_name = 'valnerabilities.shp')
      """
      # ---- Bunch of assertion to raise
      # Assuming that the model is steady-stade
      assert ml.nper == 1, 'The flow model must be steady-state like'
      # Assuming that the model spatial dscretisation is DISV like
      assert ml.get_grid_type().name == 'DISV', 'The spatial discretisation must be DISV like'
      # Assuming that the particle tracking is backward oriented
      assert mpsim.trackingdirection == 2, 'The Particle tracking is not set as backward'
      # Assuming that at least one geospatial object (zone) is implemented
      msg = 'Geospatial object (zone polygon) must be implemented (geometry or shp_name argument)'
      assert any(x is not None for x in [shp_name, geometry_dic]), msg

      # ---- Define flow and transport models
      self.ml = ml
      self.mpsim = mpsim

      # ---- Fetch zones from shapefile or geometry
      self.id_field = id_field

      if geometry_dic is None:
        self.vul_zone = gpd.read_file(shp_name)
      else:
        gdf = gpd.GeoDataFrame({id_field : geometry_dic.keys()})
        self.vul_zone = gdf.set_geometry(list(geometry_dic.values()))
         

      # ---- Build Pathlines GeoDataFrame
      # 1) Get endpoint data
      edp  = EndpointFile(os.path.join(ml.model_ws, mpsim.endpointfilename))
      edp_df = pd.DataFrame(edp.get_alldata()).set_index('particleid')
      self.edp = edp

      # 2) Get pathline data
      pth = PathlineFile(os.path.join(ml.model_ws, mpsim.pathlinefilename))
      pth_df = pd.concat([pd.DataFrame(rec) for rec in pth.get_alldata()])
      self.pth = pth

      # 3) Convert points to lines
      pth_points = gpd.GeoDataFrame(pth_df, geometry = gpd.points_from_xy(pth_df.x, pth_df.y))
      pth_lines = pth_points.groupby(['particleid'])['geometry'].apply(lambda p: LineString(p.tolist()))

      self.pth_gdf = pth_lines.to_frame().merge(edp_df, left_index = True, right_index = True)

      # ---- Build equivalence between particle group name / particle group numeric id
      self.part_group_ids_num = {pg.particlegroupname: pg_id for pg_id, pg in enumerate(mpsim.particlegroups)}




    def numbool2percent(self, df, columns, groupby_col = None):
        """
        -----------
        Description
        -----------
        Transform numerical boolean columns of a DataFrame in percent
        -----------
        Parameters
        -----------
        - df (pandas.core.frame.DataFrame) : original DataFrame that contains boolean columns
        - columns (list) : required boolean columns
        - groupby_col (str) : column name to group data before computing percent if required
                              (Default is None)
        -----------
        Returns
        -----------
        res (pandas.core.frame.DataFrame) : resulting percent DataFrame
        -----------
        Examples
        -----------
        >>> data = {col_id: np.random.randint(2, size=200) for col_id in list('abcdefg')}
        >>> df = pd.DataFrame(data)
        >>> percent_df = numbool2percent(df = df, columns = df.columns) * 100
        """
        if groupby_col is None:
            # ---- Compute pourcent on all DataFrame
            res = pd.DataFrame(df[columns].apply(sum, axis = 0) / len(df)).T
        else:
            # ---- Group by groupby_col with count and sum statistics
            stat_df = df.groupby(groupby_col)[columns].agg(['sum', 'count'])
            # ---- Switch from MultIndex to SimpleIndex
            stat_df.columns = ['_'.join(col).strip() for col in stat_df.columns.values]
            # ---- Compute percent as sum/count
            for col in columns: stat_df[col] = (stat_df[col + '_sum'] / stat_df[col + '_count'])
            # ---- Drop useless columns
            res = stat_df.drop(stat_df.columns[stat_df.columns.str.endswith(('_sum', '_count'))], axis = 1)
        # ---- Return 
        return(res)



    def get_fids(self):
        """
        -----------
        Description
        -----------
        Get feature names
        -----------
        Parameters
        -----------
        - self  (vulnerability.SSZV)
        -----------
        Returns
        -----------
        - fids (list) : feature names
        -----------
        Examples
        -----------
        >>> sszv = SSZV(ml, 'vulnerable_zones.shp', id_field = 'NAME')
        >>> zone_names = sszv.get_fids()
        """
        return self.vul_zone[self.id_field].to_list()



    def get_methods(self):
        """
        -----------
        Description
        -----------
        Get particle pathline intersection methods
        -----------
        Parameters
        -----------
        - self  (vulnerability.SSZV)
        -----------
        Returns
        -----------
        - methods (list) : list of methods to consider intersection between particle 
                           pathlines and vulnerability zones
        -----------
        Examples
        -----------
        >>> sszv = SSZV(ml, 'vulnerable_zones.shp', id_field = 'NAME')
        >>> methods = sszv.get_methods()
        """
        return ['all', 'first', 'last']



    def get_pvelocity(self, part_id):
        """
        -----------
        Description
        -----------
        Compute particle velocity
        -----------
        Parameters
        -----------
        - self  (vulnerability.SSZV)
        - part_id (int) : id of a given particle
        -----------
        Returns
        -----------
        - vi (float) : particle arriving velocity 
        -----------
        Examples
        -----------
        >>> sszv = SSZV(ml, 'vulnerable_zones.shp', id_field = 'NAME')
        >>> vi = sszv.get_pvelocity(part_id = 1)
        """
        # ---- Get particle data
        rec = self.pth.get_data(part_id)
        # ---- Extract time > 1 sec
        t0_data, t1_data = rec[rec.time > 1][:2]
        # ---- Get dt (time between 2 positional points)
        dt = t1_data.time - t0_data.time
        # ---- Get distance between points
        #dist = Point(t0_data.x, t0_data.y).distance(Point(t1_data.x,t1_data.y))
        dist = np.sqrt((t0_data.x - t1_data.x)**2 + (t0_data.y - t1_data.y)**2)
        vi = dist/dt
        # Return
        return vi


    def get_numbool(self, method = 'all'):
        """
        -----------
        Description
        -----------
        Compute numeric boolean mask from particle pathline intersection
        -----------
        Parameters
        -----------
        - self  (vulnerability.SSZV)
        - method (str) : intersection whith vulnerability zone to keep 
                         to compute zonal vulnerability
                         (Default is 'all')
                         Can be
                            - 'all' : consider all zones of vulnerability intersected
                            - 'first' : consider only the first zone of vulnerability intersected
                            - 'last' : consider only the last zone of vulnerability intersected 
        -----------
        Returns
        -----------
        - pth_gdf (pandas.core.frame.GeoDataFrame) : intersection numerix boolean DataFrame
                                                     extended with pathline data
        - zlabels (set) : set of all zones or zone groups intersected
        -----------
        Examples
        -----------
        >>> sszv = SSZV(ml, 'vulnerable_zones.shp', id_field = 'NAME')
        >>> zlabels, numbool = sszv.get_numbool()
        """
        # ---- Create a copy of pathline GeoDataFrame
        pth_gdf = self.pth_gdf.copy()
        if not self.vul_zone.crs is None:
            pth_gdf.set_crs(epsg = self.vul_zone.crs.to_epsg(), inplace = True)

        # ---- Intersect pathline with vulnerability zone
        inter_gdf = gpd.sjoin(pth_gdf, self.vul_zone, how = 'left', op = 'intersects')
        # ---- Add GroundWater origin
        inter_gdf[self.id_field].fillna('GW', inplace =True)

        # ---- Get list of vulnerability zones intersected
        groupby = inter_gdf[self.id_field].groupby('particleid')
        inter_vul_ids = groupby.apply(lambda x: ' & '.join([str(f) for f in x]))

        # ---- Get all groups of intersected zones
        zlabels = set(inter_vul_ids)

        # ---- Build numeric boolean mask accoding to the choosen method
        for zlabel in zlabels:
            if method == 'all':
                pth_gdf[zlabel] = [1 if zlabel in inter_vuls else 0 for inter_vuls in inter_vul_ids]
            elif method == 'first':
                pth_gdf[zlabel] = [1 if zlabel == inter_vuls.split(' & ')[0] else 0 for inter_vuls in inter_vul_ids]
            elif method == 'last':
                pth_gdf[zlabel] = [1 if zlabel == inter_vuls.split(' & ')[-1] else 0 for inter_vuls in inter_vul_ids]
            else:
                err_msg = f" - INVALID METHOD - \n Method must be 'all', 'first' or 'last'. Given : '{method}'"
                raise ValueError(err_msg)

        # ---- return numeric boolean mask (as DataFrame)
        
        return zlabels, pth_gdf


    
    def compute_zonal_vulnerability(self, method = 'all', pond_velocity = True):
        """
        -----------
        Description
        -----------
        Compute zonal vulnerability from MODPATH7 backward particle tracking
        -----------
        Parameters
        -----------
        - method (str) : intersection whith vulnerability zone to keep 
                         to compute zonal vulnerability
                         (Default is 'all')
                         Can be
                            - 'all' : consider all zones of vulnerability intersected
                            - 'first' : consider only the first zone of vulnerability intersected
                            - 'last' : consider only the last zone of vulnerability intersected
        - pond_velocity (bool) : ponderate particle by arrival velocity at vulnerable zone
                                 (Default is True)
        -----------
        Returns
        -----------
        - zv_df (pandas.core.frame.DataFrame) : % of water that comes from each vulnerable zone
        -----------
        Examples
        -----------
        >>> sszv = SSZV(ml, 'vulnerable_zones.shp', id_field = 'NAME')
        >>> zv_df = sszv.compute_zonal_vulnerability(filled = True)
        """
        # ---- Get numeric boolean mask of particle pathline intersection with vulnerability zones
        zlabels, pth_gdf = self.get_numbool(method = method)

        # ---- Add velocity ponderation
        if pond_velocity:
            # ---- Get particles velocity
            pth_gdf['vi'] = [self.get_pvelocity(part_id) for part_id in self.pth_gdf.index]
            # ---- Compute contribution by performing numeric boolean mask * velocity
            contri_df = pth_gdf[zlabels].multiply(pth_gdf['vi'], axis = 0)
            # ---- Change columns names
            contri_df.columns = [f'{vul_id}_contri' for vul_id in contri_df.columns]
            # ---- Join numeric boolean DataFrame with contribution & group by particlegroup
            gb_df = pth_gdf.join(contri_df, how='outer').groupby('particlegroup').sum()
            # -- Compute zonal vulnerability as sum(contribution) / sum(vi)
            zv = {vul_id : gb_df[f'{vul_id}_contri'].div(gb_df.vi).values for vul_id in zlabels}
            # ---- Convert to DataFrame
            zv_df = pd.DataFrame(zv)
        else:
            # ---- Convert GeoDataFrame in regular DataFrame
            pth_df = pd.DataFrame(pth_gdf.drop('geometry', axis = 1))
            # ---- Convert numeric boolean intersection into % of zonal origin particles
            zv_df = self.numbool2percent(df = pth_df, columns = list(zlabels), groupby_col = 'particlegroup')
            # ---- Complete mixing ratio with other naturals apportation
            zv_df['GW'] = 1 - zv_df.sum(axis = 1)

        # ---- Convert particle group ids into particle group names
        zv_df.index = self.part_group_ids_num.keys()
        # ---- Set meaningful headers
        zv_df.columns = [f'from {col}' for col in zv_df.columns]
        # ---- Convert NaN to 0
        zv_df.fillna(0,inplace =True)

        # ---- Return
        if method == 'all':
            return zv_df
        else:
            return zv_df[[col for col in zv_df.columns if not ' & ' in col]]



    def plot_venn(self, pg, ax=None, colors = None, export = None, textsize = 10, **kwargs):
        """
        -----------
        Description
        -----------
        Plot Venn diagram from particle intersection with vulnerability zones
        -----------
        Parameters
        -----------
        - self  (vulnerability.SSZV)
        - pg (int/str) : particle group name (str) or particle group id (int) to plot
        - ax (AxesSubplot) : existing subplot to using while ploting
        - colors (list) : colors of venn circles
                          (Default is None)
        - export (str, optional) : filename of output plot if required
                                   (Default is None)
        - textsize (int/float) : size of the writtent text of the venn plot
                                 Default is 10
        - **kwargs : keyword arguments for venn circle such as ls, lw, alpha, color, ..
        -----------
        Returns
        -----------
        - ax (matplotlib.axes._subplots.AxesSubplot) : axe of Venn diagram
        - v (matplotlib_venn._common.VennDiagram) : venn2 or venn3 object
        - c (list) : list of matplotlib.patches.Circle that define Venn diagram
        -----------
        Examples
        -----------
        >>> self = sszv(ml, 'vulnerable_zones.shp', id_field = 'NAME')
        >>> ax, v, c = sszv.plot_venn('WELL1', colors = ['red', 'green', 'blue'], circle_ls = ':')
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles

        # ---- Make sure that plotting Venn Diagram is possible (Only 2 and 3 sets available)
        msg = f'Venn plot are available only for 2 or 3 vulnerability zones \n Given {len(self.vul_zone)}'
        assert len(self.vul_zone) in [2, 3] , msg

        # ---- Create a copy of pathline GeoDataFrame
        pth_gdf = self.pth_gdf.copy()
        if not self.vul_zone.crs is None:
            pth_gdf.set_crs(epsg = self.vul_zone.crs.to_epsg(), inplace = True)

        # ---- Intersect pathline with vulnerability zone
        inter_gdf = gpd.sjoin(pth_gdf, self.vul_zone, how = 'left', op = 'intersects')

        # ---- Convert intersected zone serie into DataFrame
        if isinstance(pg, str):
            pg_num_id = self.part_group_ids_num[pg]
        else:
            pg_num_id = pg

        df = inter_gdf.loc[pth_gdf.particlegroup == pg_num_id, self.id_field].to_frame()

        # ---- Extract number total of particle
        npart = len(set(df.index))

        # ---- Copy index in a new column
        df['pid'] = df.index
        # ---- Group by vulnerability zone to get venn sets
        venn_dic = df.groupby(self.id_field)['pid'].apply(set).to_dict()
        # ---- Plot Venn diagram

        # -- Prepare plot
        plt.rc('font', family='serif', size=textsize)
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)

        # -- Set colors to rgba
        if not colors is None:
            if len(colors) == len(self.vul_zone):
                ccolors =  [mpl.colors.to_rgba(c) for c in colors]
            else:
                raise Exception(f'Invalid number of colors: expected {len(self.vul_zone)}')
        else:
            ccolors = plt.cm.Accent(np.arange(len(self.vul_zone)))

        # -- Plot Venn for 3 sets
        if len(self.vul_zone) == 2:
            v = venn2(subsets = venn_dic.values(),
                      set_labels = venn_dic.keys(),
                      set_colors = ccolors, ax= ax)
            # -- Modify labels colors and opacity
            for lid, color in zip(['A', 'B', 'C'], ccolors):
                v.get_label_by_id(lid).set_color(color)
            # --- Modify circle line
            c = venn2_circles(venn_dic.values(), **kwargs)

        # -- Plot Venn for 3 sets
        elif len(self.vul_zone) == 3:
            v = venn3(subsets = venn_dic.values(),
                      set_labels = venn_dic.keys(),
                      set_colors = ccolors, ax= ax)
            # -- Modify labels colors and opacity
            for lid, color in zip(['A', 'B', 'C'], ccolors):
                v.get_label_by_id(lid).set_color(color)
            # --- Modify circle line
            c = venn3_circles(venn_dic.values(), **kwargs)

        # ---- Add number total of particle and particle group name
        pg_name = list(self.part_group_ids_num.keys())[list(self.part_group_ids_num.values()).index(pg_num_id)]
        text = f'Particle group n° {pg_num_id} : {pg_name} \n Number total of particle : {npart}'
        ax.text(0.8, 0.9, text, fontsize = textsize, ha = 'center', va ='center', transform = ax.transAxes)

        # ---- Export plot if required
        if export is not None:
            fig.savefig(export, dpi = 400)

        # ---- Return
        return ax, v, c



    def __str__(self):

        print('\n')
        # ---- Print head
        header = ' Steady-State Zonal Vulnerability Class '
        # ---- Collect SSRV main informations
        inf =  ['Number of vulnerability zones', 'Names of vulnerability zones',
                'Particle Group names', 'Particle Group ids']
        res = [len(self.vul_zone), ', '.join(sorted(self.get_fids())),
               ', '.join(list(self.part_group_ids_num.keys())),
                ', '.join([str(i) for i in self.part_group_ids_num.values()])]
        # ---- Build DataFrame
        df = pd.DataFrame({' ': inf, header : res})
        # ---- Print table of information
        print(df.to_markdown(index = False, tablefmt="simple"))
        return '\n'
