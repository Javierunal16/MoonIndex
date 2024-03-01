import numpy as np
from specutils import Spectrum1D
from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
import astropy.units as u
import cv2

def fourier_filter(original_cube,percentage_width,percentage_height):
    '''Performs the fourier filtation of the cube in the spatial domain. 
    
    Inputs:
    original_cube = the prepared cube, 
    percentage_width = width of the filter in percentange (60 recommended),
    percentage_height = height of the filter in percentage (2 recommended).
    
    Outputs:
    The cube after the fourier filter.'''

    #Check input percentage values
    if percentage_width > 100 or percentage_height > 100:
        raise ValueError("Invalid percentage")
    #Creating a new cube to put the final data
    fourier_cube=original_cube.copy()  
    #Getting the dimensions to do the mask
    rows,cols=original_cube[0,:,:].shape  
    mask= np.ones((rows, cols, 2), np.uint8)
    y,z=original_cube[0,:,:].shape
    #Calculating the input size of the box
    filter_width=int((z*((100-percentage_width)/100))/2) 
    filter_height=int((y*(percentage_height/100))/2)
    #Horizontal size of the fitler
    cv2.rectangle(mask, (0,((y//2)-filter_height)), (((z//2)-filter_width),((y//2)+filter_height)), 0, -1)  
    #Vertical size of the filter
    cv2.rectangle(mask, (((z//2)+filter_width),((y//2)-filter_height)), (z,((y//2)+filter_height)), 0, -1)  
    #Creating a list to store the values
    stack_fourier=[]  

    for band in range(original_cube.data.shape[0]):
        #Fourier transform 
        input_img=original_cube[band,:,:].data
        fouraster=cv2.dft(input_img, flags=cv2.DFT_COMPLEX_OUTPUT)  
        fouraster_shift=np.fft.fftshift(fouraster)
        #Mask used to fitler the noise
        mfouraster=fouraster_shift*mask  
        #Inverse fourirer to recover the image
        m_ishift= np.fft.ifftshift(mfouraster)  
        fourier_raster=cv2.idft(m_ishift)/(y*z)
        fourier_raster= cv2.magnitude(fourier_raster[:, :, 0], fourier_raster[:, :, 1])
        fourier_raster2=fourier_raster.astype(np.float32)
        stack_fourier.append(fourier_raster2)
        
    fourier_cube.data=np.array(stack_fourier)
    
    #Creating a mask to avoid the non-data regions
    mask_cube=original_cube.copy()
    mask_cube.data[mask_cube.data != 0]=1
    fourier_cube_final=fourier_cube*mask_cube
    return fourier_cube_final


def gauss_filter (cube_filtered,wavelengths):
    '''Performs the Gaussian filter in the spectral domain. 
    
    Inputs:
    cube_filtered = fourier-filtered cube,
    wavelengths = the wavelengths file.
    
    Outputs:
    The filtered cube ready for the continuum removal.'''

    #Transposing data to ingest in Spec1d2
    tcube=np.transpose(cube_filtered.data,(2,1,0))  
    #Transforming data to spec1d2 file type 
    spec1d2=Spectrum1D(spectral_axis=wavelengths[0::]*u.AA, flux=tcube*u.dimensionless_unscaled)  
    #Filtering with a 1d gaussian function, using a 4 pixel kernel
    spec1d2_gsmooth=gaussian_smooth(spec1d2, stddev=1) 
    #Slicing the filtered cube to avoid the affected data at longer wavelengths an re-attaching the original
    tcube_smooth=spec1d2_gsmooth.flux.value  
    tslice1=tcube_smooth[:,:,0:79]
    tslice2=tcube[:,:,79:83]
    spec1d2_filtered=np.dstack((tslice1,tslice2))
    #Saving the filtered data in a new cube, copied from the original to maintain the projection
    M3_gaussfilter=cube_filtered.copy()  
    M3_gaussfilter.data=np.transpose(spec1d2_filtered,(2,1,0))
    
    #Creating a mask to avoid no-data
    mask_cube=cube_filtered.copy()
    mask_cube.data[mask_cube.data != 0]=1
    gauss_cube_final=M3_gaussfilter*mask_cube
    return(gauss_cube_final)