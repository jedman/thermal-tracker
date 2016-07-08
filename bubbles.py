import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
# TO DO
## add a way to identify "good" tracked thermals from bad ones
## one idea is a quick change in volume -- or velocity?
class Bubble():
    def __init__(self, datafile):
        '''contains radial averaged data and thermal contour froma bubble simulation'''
        dataset = nc.Dataset(datafile, 'r')
        self.coords = {}
        for variable in ('x', 'y', 'z', 'time'):
            self.coords[variable] = dataset[variable][:]
        self.coords['time'] = self.coords['time']*86400 # convert to seconds
        # get the radial grid centered on the Bubble
        self.radius_grid()

        # bin points around the origin by radius for radial dimension
        self.get_rbins() # sets coord['rbins'] and preps for azimuthal average

        # get azimuthal averages
        self.az_av = {}
        self.az_av['w'] = self.azimuthal_average(dataset['w'][:])
        self.az_av['rho'] = self.azimuthal_average(dataset['rho'][:])
        self.az_av['tracer'] = self.azimuthal_average(dataset['bubble'][:])
        self.az_av['buoyancy'] = self.azimuthal_average(self.buoyancy(dataset['rho'][:]))

        # time-mean profiles
        self.profiles = {}
        self.profiles['rho'] = np.mean(dataset['rho'][:], axis = (0,2,3))

        # close the netCDF file
        dataset.close()

        self.az_av['psi'] = self.streamfunction() # radially-averaged streamfunction




    def thermal_init(self):

        self.thermaldata = {} # initialize thermal dictionary
        self.thermal_volume()
        self.thermal_height()
        self.thermal_radius()
        self.thermaldata['w_top'] = self.get_thermal_w()
        self.thermaldata['buoyancy'] = self.get_thermal_average('buoyancy')
        self.thermalindex = np.arange(0,np.shape(self.coords['time'])[0]) # index contains all time steps

        ## add a way to identify "good" tracked thermals from bad ones
        ## one idea is a quick change in volume -- or velocity?

    def thermal_trim(self, crit, **kwargs):
        '''select for thermals satisfying crit, where crit is a function taking a bubble object and a timestep,
        and returning a boolean
        kwargs are additional keyword arguments to function crit'''
        index = [i for i, t in enumerate(self.thermaldata['volume']) if crit(self, i, **kwargs)]
        for variable in self.thermaldata:
            self.thermaldata[variable] = self.thermaldata[variable][index]
        self.thermalindex = index
        return

    def radius_grid(self):
        '''initialize xgrid, ygrid, and rgrid'''
        dx = self.coords['x'][1] - self.coords['x'][0]
        dy = self.coords['y'][1] - self.coords['y'][0]

        coordinates = list()
        for coord, delta in zip((self.coords['x'], self.coords['y']), (dx, dy)):
            mid_ind = int(coord.shape[0]/2)
            if coord.shape[0] % 2 == 0:
                mid = coord[mid_ind]  - delta/2
            else:
                mid = coord[mid_ind]

            coordinates.append(coord - mid)


        self.xgrid, self.ygrid = np.meshgrid(coordinates[0], coordinates[1])
        self.rgrid = np.sqrt(self.xgrid**2 + self.ygrid**2)
        return

    def get_rbins(self, rmax = 6000., rounding = True):
        '''find coord['rbins'], the binned radial coordinate,
         and the grid points per bin, rbin_counts'''
        index = np.argsort(self.rgrid.flat) # index to sort by radius
        self.sorted_rgrid = self.rgrid.flat[index]
        self.sorted_rgrid = self.sorted_rgrid[self.sorted_rgrid <= rmax] # only keep if r less than rmax
        if (rounding):
            self.sorted_rgrid = np.around(self.sorted_rgrid, decimals = -1) #  round to 10m bins

        dr = np.diff(self.sorted_rgrid) # nonzero if r changes
        self.r_step_locs = dr.nonzero()[0] # indices of changes in r
        self.rbin_counts = np.diff(self.r_step_locs) # difference is number in each r bin
        self.coords['rbins'] = self.sorted_rgrid[self.r_step_locs][0:-1]
        return

    def azimuthal_average(self, field):
        '''return azimuth average of field, using rbins'''

        averaged_field = np.zeros((self.coords['time'].shape[0], self.coords['z'].shape[0], self.coords['rbins'].shape[0]))
        index = np.argsort(self.rgrid.flat)
        for i, _ in enumerate(self.coords['time']):
            for k, _ in enumerate(self.coords['z']):
                flat_field = field[i, k, : ,:].flat[index] # sorted data by radius
                summed = np.cumsum(flat_field) # integrate in r
                bin_total = summed[self.r_step_locs[1:]] - summed[self.r_step_locs[:-1]] # calculate total field for each r bin
                averaged_field[i, k, :] = bin_total/self.rbin_counts # average over number of counts in each r bin

        return averaged_field

    def streamfunction(self):
        '''takes azimuthally-averaged w, rho field and convert to streamfunction (az_av['psi'])'''
        psi = np.zeros((self.coords['time'].shape[0], self.coords['z'].shape[0], self.coords['rbins'].shape[0]))
        dr = np.diff(self.coords['rbins'])
        dr = np.append(dr, dr[-1]) # make same size

        w_top = self.get_thermal_w() # estimated velocity of bubble rise

        for i, _ in enumerate(self.coords['time']):
            for k, _ in enumerate(self.coords['z']):
                psi[i, k, :] = np.cumsum(self.coords['rbins'][:]*(self.az_av['w'][i,k,:]
                                                           - w_top[i])*self.az_av['rho'][i,k,:]*2.*np.pi*dr)
        return psi

    def buoyancy(self, rho):
        '''calculate buoyancy for entire 4D grid covered by rho'''
        av_rho = np.mean(rho, axis = (2,3))
        buoyancy_4d = np.zeros(rho.shape)
        for k, _ in enumerate(self.coords['time']):
            for i, _ in enumerate(self.coords['x']):
                for j, _ in enumerate(self.coords['y']):
                    buoyancy_4d[k, :, i, j] = (-rho[k, :, i, j] + av_rho[k, :])*9.81/(av_rho[k,:])

        return buoyancy_4d

    # thermal utilities -- perhaps I should make the thermal a separate class?

    def get_thermal_average(self, az_av_field):
        '''average field within thermal mask'''
        dr = np.diff(self.coords['rbins'])
        dr = np.append(dr, dr[-1]) # make same size
        dz = self.coords['z'][1] - self.coords['z'][0] # assumes uniform z grid
        average_purity = np.sum( 2.*np.pi * np.sum( self.coords['rbins'] * (self.get_thermal_mask() * self.az_av[az_av_field] * dr),
         axis = 2) * dz, axis = 1)/(self.thermaldata['volume'])
        return average_purity

    def thermal_volume(self):
        '''compute volume of the thermal
        NOTE: could add some quality controls here... or check for multiple contours'''
        self.thermaldata['volume'] = np.zeros(self.coords['time'].shape)
        for step, _ in enumerate(self.coords['time']):
            self.thermaldata['volume'][step] = self.get_contour_volume(self.thermal_contour( step ))
        return

    def thermal_height(self):
        '''compute height of thermal from stream fucntion'''
        self.thermaldata['height'] = np.zeros(self.coords['time'].shape)
        for step, _ in enumerate(self.coords['time']):
            self.thermaldata['height'][step] = self.get_contour_height(self.thermal_contour( step ))
        return

    def thermal_radius(self):
        '''use thermal volume to find thermal radius'''
        self.thermaldata['radius']=(0.75*self.thermaldata['volume'])**(1./3.)
        return

    def froude(self, thetaprof):
        # ingredients: N, thermal rise rate, thermal radius, thermal height
        # calculating N
        ggr = 9.81
        c_p = 719. + 287.04
        Nsq = ggr / theta_prof * ddzp(theta_prof, self.coords['z'])

        # pull Nsq at the thermal height
        N_loc = np.zeros(self.coords['time'][self.thermalindex].shape)
        for i, _ in enumerate(self.coords['time'][self.thermalindex]):
            N_loc[i] = Nsq[np.where(self.thermaldata['height'][i] == self.coords['z'])]

        self.thermaldata['froude'] = self.thermaldata['w_top']/(N_loc* self.thermaldata['radius'])

        return


    def thermal_contour(self, step):
        '''return the zero contour from a slice of stream function'''
        CS = plt.contour( self.coords['rbins'],self.coords['z'], self.az_av['psi'][step,:,:],
                        (  -1e1, 0), colors='k')
        try:
            return CS.collections[1] # return 0 contour
        except IndexError:
            print 'oh no'
            return

    def get_thermal_w(self):
        '''vertical velocity of bubble, estimated from thermal theory'''
        return np.max(self.az_av['w'], axis =(1,2))/2.

    def get_thermal_mask(self):
        '''return interior of zero-contour of radially-averaged stream function'''
        masked_region = np.where(self.az_av['psi'] >= 0, 1.,0.)
        return masked_region

    def get_contour_volume(self, contour):
        '''returns contour volume'''
        rsq = []
        y = []

        try:
            for i, _ in contour.get_paths()[0].iter_segments():
                rsq.append(i[0]**2)
                y.append(i[1])

            dy = np.diff(y)
            dy = np.append(dy, dy[-1])
            thermal_volume = np.pi*np.sum(rsq*dy)

            return thermal_volume
        except IndexError:
            return 0. # no thermal no volume

    def get_contour_height(self, contour):
        '''return height of max contour radius'''
        r = []
        y = []

        try:
            for i, _ in contour.get_paths()[0].iter_segments():
                r.append(i[0])
                y.append(i[1])
            return(y[np.argmax(r)])
        except IndexError:
            return 0.


    ### Plotting methods
    def plot_1d(self, coord, fields, **kwargs):
        '''a 1D plot'''
        print 'not implemented!'
        return

    def plot_2d(self, coords,):
        '''take coords: (x, y); field; and plot them using pcolormesh'''
        print 'not implemented!'

        return
    def panelplot(self, coords, fields):
        '''make panel plot using plot_1d or plot_2d methods'''
        print 'not implemented!'
        return

def ddzp(prof1, z, sdo=False):
    ''' calculate ddz of some scalar profile on the scalar levels,\
       interpolating to the surface if sdo =True'''
    dz = np.zeros(len(z))
    dz[0] = 0.5*(z[0]+z[1])
    for i in range(1,len(z)-1):
       dz[i] = 0.5*(z[i+1]-z[i-1])
    dz[-1]= dz[-2] # fudge for the top level -- don't know dzi[-1]
    vflux = np.zeros(len(z)+1)
    for k in xrange(1,len(z)):
        vflux[k] = 0.5*(prof1[k]+prof1[k-1]) # value of prof1 at the interface k
        if(sdo): # for EvRTdv and Eldl, the surface flux is nonzero, so use the value at the interface
            vflux[0] = prof1[0]-(prof1[1]-prof1[0])/(z[1]-z[0])*z[0]
    ddz_something = (vflux[1:] - vflux[0:-1])/dz[0:]
    return ddz_something
