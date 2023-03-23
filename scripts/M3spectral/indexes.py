import numpy as np

#R540, reflectance at 540 nm
def R540 (fourier_cube):
    cube_R540=fourier_cube[0,:,:]  #The first band corresponds to that wavelength
    return cube_R540
