import numpy as np
from specutils import Spectrum1D
from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
import astropy.units as u
import cv2


#Gauss filtration
def gauss_filter (cube_filter1,wavelen):
    tcube=np.transpose(cube_filter1.data,(2,1,0))  #Transposing data to ingest in Spec1d2
    
    spec1d2=Spectrum1D(spectral_axis=wavelen[0::]*u.AA, flux=tcube*u.dimensionless_unscaled)  #Transforming data in spec1d2 file type 
    spec1d2_gsmooth=gaussian_smooth(spec1d2, stddev=1) #Filtering with a 1d gaussian function, using 4 pixel kernel
    
    tcube_smooth=spec1d2_gsmooth.flux.value  #Slicing the filtered cube to avoid the affected data at longer wavelengths an attaching the original
    tslice1=tcube_smooth[:,:,0:79]
    tslice2=tcube[:,:,79:83]
    spec1d2_filtered=np.dstack((tslice1,tslice2))
    
    M3_gaussfilter=cube_filter1.copy()  #Saving the filtered data in a new cube, copied from the original to maintain the projection
    M3_gaussfilter.data=np.transpose(spec1d2_filtered,(2,1,0))
    
    #Creating mask to avoid no-data
    mask_cube=cube_filter1.copy()
    mask_cube.data[mask_cube.data != 0]=1
    gauss_cube_final=M3_gaussfilter*mask_cube
    
    return(gauss_cube_final)


#Fourier filtration
def fourier_filter(original_cube,percentage_width,percentage_high):
    fourier_cube=original_cube.copy()  #Creatign a new cube to put the final data
    rows,cols=original_cube[0,:,:].shape  #Getting dimension to do the mask
    mask= np.ones((rows, cols, 2), np.uint8)
    y2,z2=original_cube[0,:,:].shape
    
    filter_width=int((z2*((100-percentage_width)/100))/2) #Calclating the inut size of te box
    filter_high=int((y2*(percentage_high/100))/2)

    cv2.rectangle(mask, (0,((y2//2)-filter_high)), (((z2//2)-filter_width),((y2//2)+filter_high)), 0, -1)  #Horizontals size of the fitler
    cv2.rectangle(mask, (((z2//2)+filter_width),((y2//2)-filter_high)), (z2,((y2//2)+filter_high)), 0, -1)  #Vertical size of the filter
 
    stack_fourier=[]  #Creating a list to store the values

    for band in range(original_cube.data.shape[0]):
    
        input_img=original_cube[band,:,:].data
        fouraster=cv2.dft(input_img, flags=cv2.DFT_COMPLEX_OUTPUT)  #Fourier transform 
        fouraster_shift=np.fft.fftshift(fouraster)

        mfouraster=fouraster_shift*mask  #Mask used to fitler the noise
        m_ishift= np.fft.ifftshift(mfouraster)  #Inverse fourirer to recover the image
        fourier_raster=cv2.idft(m_ishift)/(y2*z2)
        fourier_raster= cv2.magnitude(fourier_raster[:, :, 0], fourier_raster[:, :, 1])
        fourier_raster2=fourier_raster.astype(np.float32)
        stack_fourier.append(fourier_raster2)
        
    fourier_cube.data=np.array(stack_fourier)
    
    #Creating a mask to avoid the non-data regions
    mask_cube=original_cube.copy()
    mask_cube.data[mask_cube.data != 0]=1
    fourier_cube_final=fourier_cube*mask_cube
    
    return fourier_cube_final