import argparse as ap
import healpy as hp
import numpy as np
#from astropy 

def sph2cart(a,b,c):
   return 0
def cart2sph(a,b,c):
   return 0
def cart2sph_ar(a,b,c):
   return 0

class FootpointGrid:
    ''' Grid of field line foopoints. 
        Parameters: 
            grid_type: "spherical", "quasiuniform", "healpix"
            
            
    
    '''
    def __init__(self, grid_type='', **kwargs):
        if not(grid_type in ['spherical', 'healpix', 'quasiuniform']):
            print("Grid grid_type must be 'spherical', 'quasiuniform', or 'healpix'")
            return
        self.grid_type = grid_type
        if self.grid_type=='spherical':
            self.nlon = nlon
            self.nlat = nlat
            self.nline = self.nlon*(self.nlat-2)+2
            
            ph_gr = np.linspace(0, 2*np.pi, self.nlon, endpoint=False) 
            th_gr = np.linspace(0, np.pi, self.nlat)
            
            ph_grid, th_grid = np.meshgrid(ph_gr, th_gr)
            ph_grid = np.ravel(ph_grid)
            th_grid = np.ravel(th_grid)
            duplicate = np.where((ph_grid>0) & ((th_grid==0) | (th_grid==np.pi)))
            self.th_or = np.delete(th_grid, duplicate)
            self.ph_or = np.delete(ph_grid, duplicate)
            self.x_or, self.y_or, self.z_or = sph2cart(1, self.th_or, self.ph_or)
            
            if rot is not None:
             vec_rot = rot.apply(np.transpose((self.x_or,self.y_or,self.z_or)))
             self.x = vec_rot[:,0]
             self.y = vec_rot[:,1]
             self.z = vec_rot[:,2]
             
             _, self.th, self.ph = cart2sph_ar(self.x, self.y, self.z)
             
            else:
             self.th = self.th_or
             self.ph = self.ph_or
             self.x = self.x_or
             self.y = self.y_or
             self.z = self.z_or
         
        if self.grid_type=='quasiuniform':  
            self.n = n
            self.nline = n
            self.ph = np.zeros(self.n)
            k = np.arange(1,self.n+1,1)
            h = -1 + 2.*(k-1)/(self.n-1)
            self.th = np.arccos(h)
            self.ph[0] = 0
            self.ph[-1] = 0
            for i in np.arange(1, self.n-1, 1):
             self.ph[i] = np.mod((self.ph[i-1] + 3.6/self.n**0.5/(1-h[i]**2)**0.5), 2*np.pi)
             
        if self.grid_type=='healpix':
            self.nside = nside
            self.nline = hp.nside2npix(nside)
            print( " Approx res at NSIDE {} is {:.2} deg ({} points)". format(nside, hp.nside2resol(nside, arcmin=True)/60,self.nline ))
            self.th, self.ph = hp.pix2ang(nside=self.nside, ipix=np.arange(self.nline))
            
        if rot is None:    
         self.x, self.y, self.z = sph2cart(1, self.th, self.ph)    


if __name__ == "__main__":
   #parser = ap.ArgumentParser(description='Initialize fiel line footpoint grid.')
   #parser.add_argument('grid_type', metavar='N', type=str,
   #                    options=['spherical_coord', 'healpix', 'quasiuniform'],
   #                    help='Should be spherical_coord, healpix, or quasiuniform')


    # Create the main parser
    parser = ap.ArgumentParser(description='Initialize the grid of field line footpoints')

    # Create subparsers for different values of argument #1
    subparsers = parser.add_subparsers(dest='grid_type', help='Tesselation type')

    # Subparser for spherical
    parser_spherical = subparsers.add_parser('spherical', help='Uniform division in longitude/latitude. \
                                             Number of footpoints: nlat*(nlong-2)+2')
    parser_spherical.add_argument('nlat', help='Number of latitude divisions')
    parser_spherical.add_argument('nlong', help='Number of longitude divisions')

    # Subparser for random
    parser_random = subparsers.add_parser('random', help='Random uniform distribution of points on sphere')
    parser_random.add_argument('parameters', nargs='*', help='Grid parameters: number of footpoints nline=<int>')
    #parser_random.add_argument('--nline', help='Number of footpoints')

    # Subparser for healpix
    parser_healpix = subparsers.add_parser('healpix', help='Healpix tesselation')
    parser_healpix.add_argument('nside', help='Number of pixels per side. Number of footpoints: 12*nside^2.')

    print("Hi! I'm here!")

    # Parse the arguments
    args = parser.parse_args()
    items_dict = {}

    if args.grid_type == 'random':    
        for item in args.parameters:
            if '=' in item:
                key, value = item.split('=', 1)
                items_dict[key] = value
            else:
                print(f"Invalid item format: {item}. Expected format: key=value")

    grid = FootpointGrid(grid_kind=args.grid_type, **items_dict)

    print("Some stuff")
    """
    # Depending on the chosen command, handle the arguments accordingly
    if args.command == 'spherical':
        print(f'Option 1 chosen with argument #1: {args.nlat} and subseq argument #2: {args.nlong}')
    elif args.command == 'random':
        print(f'Option 2 chosen with argument #1: {args.grid_type} and required argument #2: {args.nside}')

    """
    #grid = FootpointGrid(args.grid_grid_type, *grid_parameters)

# TODO: add visualization of points on 2D plane.

