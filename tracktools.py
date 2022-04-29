

import os
import numpy as np
import warnings

try : 
    import pandas as pd
except ImportError :
    print('Could not load pandas...')



base_particledata = pd.DataFrame(
                        columns=['fid', 'node', 'lx', 
                                 'ly', 'pid', 'geometry'])

# TODO 
# -> set velocity weighing as optional
# -> regular grid support 
# -> zone vulnerability 

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

        # initialize particle data 
        self.particledata = base_particledata


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


    def _import_shp(self,shpname):
        """
        Description
        -----------
        Import shapefile as DataFrame with flopy utilities.
        
        Parameters
        -----------
        - shpname (str) : path to the shapefile
        Returns
        -----------
        df (DataFrame) : attribut table of the shapefile
                         with a additional 'geometry' column
                         with related shapely geometries. 
        
        Examples
        -----------
        >>> well_df = _import_shp('wells.shp') 
        """
        # -- Import flopy spatial utilities
        try :
            from flopy.export.shapefile_utils import shp2recarray
            from flopy.utils.geospatial_utils import GeoSpatialUtil as gsu
        except ImportError :
            print('Could not import flopy geospatial utils ...')

        # -- Convert DataFrame with shapely geometries
        df = pd.DataFrame.from_records(shp2recarray(shpname))
        df['geometry'] = [gsu(g).shapely for g in df.geometry]

        # -- Return DataFrame
        return df




    def add_particledata(self, particledata):
        """
        """
        # warn/remove duplicates
        fids =  particledata['fid'].unique()
        self.particledata.drop(self.particledata.query('fid in @fids').index, inplace=True)
        # concatenate clean particle data with new one
        self.particledata = pd.concat(
                                [self.particledata, particledata],
                                     ignore_index=True)
        # update particle ids
        self.particledata['pid'] = np.arange(len(self.particledata))


                

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
        # Remove all particle data
        if fids is None:
            self.particledata = base_particledata
            if verbose:
                print('All features had been removed.')
        # Remove selected particle data
        else:
            _fids = [fids] if isinstance(fids, str) else fids
            self.particledata.drop(self.particledata.query('fid in @_fids').index, inplace=True)
            if verbose:
                print('\n'.join([f'Feature `{fid}` had been removed.' for fid in _fids]))
        # update particle ids
        self.particledata['pid'] = np.arange(len(self.particledata))




    def gen_points(self, obj, n = 100, id_field= 'fid', fids=None,  gen_type='around', export = None):
            """
            Description
            -----------
            Generate n points around model cells, or groups of adjacent model cells.
            Models cell(s) can also be collected by spatial intersection when obj 
            is a dict of geometry or a shapefile.
            Dictionary keys or feature ids of obj will be used as particle group
            names.
            Parameters
            -----------
            obj: str or dict of nodes or dict of geometries
                Can be :
                    - dictionary of model node lists
                        Example: {'w1': [343], 'd1': [521, 522, 523]}
                    - dictionary of shapely geometry, flopy.utils.geometry objects, 
                                    list of vertices, geojson geometry objects, 
                                    shapely.geometry objects
                        Example: - {'w1': [x_w1, y_w1]} 
                                 - {'d1': shapely.LineString([p1, p2, p3])}
                    - path to shapefile
                        Example: 'gis/wells.shp'
            n: int
                number of points
                Default is 100.
            id_field: str
                column name in the shapefile attribute table to fetch geometries names.
                Default is 'fid'.
                Only effective when obj is a shapefile
            fids: str or list of str
                feature ids to consider
                If None, all features are considered
                Only effective when obj is a shapefile
            export: str
                path to export generated points as shapefile
                If None, nothing is exported
                Default is None.
            Examples
            -----------
            >>> pg = ParticleGenerator(ml)
            >>> pg.gen_points('mywells.shp', n = 500, fids = 'WELL1')
            """

            try :
                from flopy.utils.geospatial_utils import GeoSpatialUtil as gsu
                from shapely.geometry import Point, LineString, Polygon
                from shapely.ops import unary_union
            except ImportError :
                print('Could not import shapely.geometry utils ...')

            # manage input type
            ispath = isinstance(obj, str)
            isdict = isinstance(obj, dict)

            # fetch GeoDataFrame from shapefile or dictionary of geometry
            if ispath:
                df = self._import_shp(obj)
                df['fid'] = df[id_field]
                if fids is not None :
                    fids = fids if isinstance(fids, list) else [fids]
                    df = df.loc[df['fid'].isin(fids)]
            else:
                hasgeom = any(isinstance(list(obj.values())[0],t)
                          for t in [Point, LineString, Polygon])
                if np.logical_and(isdict, hasgeom):
                    df = pd.DataFrame({ 'fid' : obj.keys(),
                                         'geometry' : [gsu(g).shapely for g in obj.values()]})
                    if fids is not None:
                        fids = fids if isinstance(fids, list) else [fids]
                        df = df.loc[df['fid'].isin(fids)]
                else:
                    df = None

            # get node dictionary
            if df is None:
                node_dic = obj
            else:
                # intersect grid
                df['node'] = df.geometry.apply(lambda g:
                                                    [node[0] for node in self.gi.intersects(g)])
                node_dic = {k:v for k,v in zip(df.fid, df.node)}


            # iterate over feature ids
            point_df_list = []
            for fid, nodes in node_dic.items():
                if isinstance(nodes, int):
                    nodes = [nodes]
                # merge cells and collect cell vertices as Point
                cells_envelope = unary_union([
                                            Polygon(np.column_stack(
                                                        [self.vxs[node], self.vys[node]])
                                                    ) 
                                                        for node in nodes ])
                # generate points
                if gen_type == 'around':
                    # Convert to cell contour line
                    line = cells_envelope.boundary
                    # find the distances from origin between starting points along the line
                    distances = np.linspace(0, line.length,n)
                    # starting point at each distance
                    point_list = [line.interpolate(distance) for distance in distances]
                elif gen_type == 'within':
                    point_list= self._gen_points_in_polygon(n, cells_envelope)[:n]
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
                # build points DataFrame for current feature
                point_df =  pd.DataFrame({  'fid':fid,
                                            'node':nodes,
                                            'lx':lxs,
                                            'ly':lys,
                                            'pid':  np.arange(len(lxs)),
                                            'geometry': point_list})
                point_df_list.append(point_df)

            # concatenate point groups 
            particledata = pd.concat(point_df_list, ignore_index=True)
            self.add_particledata(particledata)

            # export points to shapefile if required
            if export is not None:
                try :
                    from shapely.geometry import GeometryCollection
                    from flopy.export.shapefile_utils import recarray2shp
                except ImportError :
                    print('Could not import flopy.export.shapefile_utils utils ...')

                rec = particledata.drop('geometry', axis=1).to_records(index=False)
                geoms = GeometryCollection(particledata['geometry'].tolist())
                recarray2shp(recarray=rec, geoms=geoms, shpname=export)



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
            pgids = list(set(self.particledata.fid))
        
        # when a single pgid is provided
        if not isinstance(pgids, list):
            pgids = [pgids]
        
        # initialize list of particle groups
        pgrp_list = []

        # iterate of particle group ids 
        for pgid in pgids:
            # subset by group id(s)
            subset = self.particledata[self.particledata['fid']==pgid]
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

        '''
        # get index of time = 1.0
        idx = [ int(np.argwhere(pdata['time']==1.)) for  pdata in spdata]

        # get particle ids, dt, dx, dy, v
        pid = np.array([pdata['particleid'][i]  for i, pdata in zip(idx,spdata)])
        dt = np.array([pdata['time'][i+1] -1. for i, pdata in zip(idx,spdata)])
        dx = np.array([ pdata['x'][i+1] - pdata['x'][i]  for i, pdata in zip(idx,spdata)])
        dy = np.array([ pdata['y'][i+1] - pdata['y'][i]  for i, pdata in zip(idx,spdata)])

        '''

        pid = np.array([pdata['particleid'][0]  for pdata in spdata])
        dt = np.array([pdata['time'][2] -pdata['time'][1] for pdata in spdata])
        dx = np.array([ pdata['x'][2] - pdata['x'][1]  for pdata in spdata])
        dy = np.array([ pdata['y'][2] - pdata['y'][1]  for pdata in spdata])
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
        else :
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
            
            # back to 0-based 
            riv_df['node']=riv_df.node -1

        # ---- Convert to dictioanry
        self.rivname_dic = riv_df.groupby('boundname').apply(
                                    lambda r: r.node.tolist()
                                        ).to_dict()




    def compute_mixing_ratio(self, on='river', edp_cell_budget = True, v_weight = True):
        """
        -----------
        Description
        -----------
        Compute the mixing ratio between river and ground water at a sink 

        -----------
        Parameters
        -----------
        - on (str/dict) : sets identification method for weak sources  
                          Can be :
                          - keyword :
                                - 'river': compute mixing ratios for all river cells
                                - 'reach': compute mixing rations by river reaches

                          - aggregation dictionary:
                                Format: {'reach_group1' : ['reach1', 'reach2',..],
                                         'reach_group2' : 'reach4'} )
                           Default is 'river'.
        
        - edp_cell_budget (bool): Account for mixing in weak source endpoints
                                  Default is True.

        - v_weight (bool) : Weight particle with velocities 
                            Default is True.

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
            warn_msg = 'Particle group names not loaded, ' \
                       'default names will be used.'\
                       'Consider using load_pgrp_names(). ' 
            warnings.warn(warn_msg, Warning)
            self.load_pgrp_names()

        edp_df['grpnme'] = [self.pgrpname_dic[gid] 
                                 for gid in edp_df.particlegroup.values]

        # identify endpoints in river cells
        edp_df['endriv'] = edp_df.node.apply(lambda n: n in self.riv_cells)

        if edp_cell_budget:

            # add river leakage 
            edp_df['riv_leak'] = 0.
            edp_df.loc[edp_df.endriv,'riv_leak'] = edp_df.loc[
                    edp_df.endriv,
                    'node'].apply(
                            lambda n: self.riv_leak_df.loc[n,'q'].sum())

            # compute cell inflows from cbc
            edp_df.loc[edp_df.endriv,'rivcell_q'] = edp_df.loc[edp_df.endriv,
                    'node'].apply(self.get_cell_inflows)

            
            # compute particle mixing ratio
            edp_df['alpha'] = edp_df['riv_leak']/ (edp_df['riv_leak']+edp_df['rivcell_q'])
            edp_df.loc[edp_df.alpha.isnull(),'alpha']=0.

        else:
            # Set alpha to 0 or 1
            edp_df['alpha'] = edp_df['endriv'].astype(int) # convert True/False to binary 1/0


        if v_weight:
            # add particle velocity and merge value into edp_df
            v_df = self.get_part_velocity()
            edp_df.loc[edp_df.particleid,'v'] = v_df.loc[edp_df.particleid,'v']

        else:
            edp_df['v'] = 1 # Unitary velocity


        # Grouped weighted average of mixing ratios
        # When they are provided, results are labeled with particle group names 
        # and bc id names, rather than integer ids.

        if on == 'river':
            # set basic river names
            edp_df['src'] = edp_df.endriv.replace({True : 'river', False : 'others'})
            
        else :
            if self.rivname_dic is None:
                try:
                    self.load_rivname_dic()
                except:
                    raise Exception('River names not loaded. ' \
                                    'Consider using .load_rivname_dic().')
            
            # set river names by reaches
            for rivname, rivnodes in self.rivname_dic.items(): 
                edp_df.loc[edp_df.node.isin(rivnodes), 'src'] = rivname.upper()
            edp_df['src'].fillna('others', inplace=True)
            
            # aggregate by reaches
            agg = {} if on == 'reach' else on
            for k,v in agg.items(): edp_df['src'].replace(v,k, inplace=True)


        # compute group mixing ratios from weighted average of particle mixing ratios
        contrib = edp_df.groupby(
                ['grpnme', 'src'], dropna=False).apply(
                                    lambda d: np.sum(d.alpha*d.v))

        mr = contrib/edp_df.groupby(
                ['grpnme'], dropna=False).apply(
                                    lambda d: np.sum(d.v))

        # return mixing ratios as DataFrame
        mr_df = mr.unstack(level=0, fill_value=0)
        mr_df.loc['others'] = 1 - mr_df.sum()

        return mr_df.T

