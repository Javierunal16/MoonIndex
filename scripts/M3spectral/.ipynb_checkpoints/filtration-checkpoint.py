import numpy as np
from specutils import Spectrum1D
from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
import astropy.units as u

#Gauss filtration
def gauss_filter (cube_filter1,wavelen):
    tcube=np.transpose(cube_filter1.data,(2,1,0))  #Transposing data to ingest in Spec1d2
    
    spec1d2=Spectrum1D(spectral_axis=wavelen[0::]*u.AA, flux=tcube*u.dimensionless_unscaled)  #Transforming data in spec1d2 file type 
    spec1d2_gsmooth=gaussian_smooth(spec1d2, stddev=2) #Filtering with a 1d gaussian function
    
    tcube_smooth=spec1d2_gsmooth.flux.value  #Slicing the filtered cube to avoid the affected data at longer wavelengths an attaching the original
    tslice1=tcube_smooth[:,:,0:79]
    tslice2=tcube[:,:,79:83]
    spec1d2_filtered=np.dstack((tslice1,tslice2))
    
    M3_gaussfilter=cube_filter1.copy()  #Saving the filtered data in a new cube, copied from the original to maintain the projection
    M3_gaussfilter.data=np.transpose(spec1d2_filtered,(2,1,0))
    return(M3_gaussfilter)