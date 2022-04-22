
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

