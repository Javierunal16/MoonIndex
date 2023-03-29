import numpy as np
import M3spectral.preparation

#R540, reflectance at 540 nm
def R540 (fourier_cube):
    cube_R540=fourier_cube[0,:,:]  #The first band corresponds to that wavelength
    return cube_R540

#BDI, band depth at the 1000 nm absorption peak, it is obtained by dividing the reflectance by the value of the continnum at that location, always positive
def BDI (fourier_cube, hull_cube, wavelengths):
    cube_BDI=hull_cube[0,:,:].copy()  #Copying the cube to save the results
    stack_BDI=[]
    y,z=hull_cube[0,:,:].shape
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            imput_BDI=fourier_cube.data[:,a,b]
            imput_hull=hull_cube.data[:,a,b]
        
            rfl_1000p=np.where(imput_hull[0:29] == min(imput_hull[0:29]))[0]  #Finding the value of the reflectance
            rfl_1000=imput_BDI[rfl_1000p]
         
            fits_1000BDI=M3spectral.preparation.continnum_1000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Fitting at 1000 nm
            fitrfl_1000=np.polyval(fits_1000BDI,wavelengths[rfl_1000p])  #Obtainign the value of the continumm with the fit
            Depth1000=(1-(rfl_1000/fitrfl_1000))  #DOing the division
        
            stack_BDI.append(Depth1000)
            
    stack_BDIa=np.array(stack_BDI)
    cube_BDI.data=stack_BDIa.reshape(y,z)
    return cube_BDI

#BDII, band depth at the 2000 nm absorption peak, it is obtained by dividing the reflectance by the value of the continnum at that location, always positive
def BDII (fourier_cube, hull_cube, wavelengths):
    cube_BDII=hull_cube[0,:,:].copy()  #Copying the cube to save the results
    stack_BDII=[]
    y,z=hull_cube[0,:,:].shape
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            imput_BDI=fourier_cube.data[:,a,b]
            imput_hull=hull_cube.data[:,a,b]
        
            rfl_2000p=np.where(imput_hull[35:75] == min(imput_hull[35:75]))[0]+35 #Finding the value of the reflectance
            rfl_2000=imput_BDI[rfl_2000p]
         
            fits_2000BDI=M3spectral.preparation.continnum_2000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Fitting at 2000 nm
            fitrfl_2000=np.polyval(fits_2000BDI,wavelengths[rfl_2000p])  #Obtainign the value of the continumm with the fit
            Depth2000=(1-(rfl_2000/fitrfl_2000))  #DOing the division
        
            stack_BDII.append(Depth2000)
            
    stack_BDIIa=np.array(stack_BDII)
    cube_BDII.data=stack_BDIIa.reshape(y,z)
    return cube_BDII

#SS1200, Spectral slope between maximun right shoulder and 540nm
def SSI (fourier_cube, hull_cube, wavelengths):
    SSI=fourier_cube[0,:,:].copy()
    stack_SSI=[]
    y,z=hull_cube[0,:,:].shape
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
        
            imput_SS1200=fourier_cube.data[:,a,b]
            imput_hull=hull_cube.data[:,a,b]
        
            shoulder0p=np.where(imput_hull[0:20] == max(imput_hull[0:20]))[0][-1]  #Obtaining the position of the first shoudler
        
            SS=((imput_SS1200[shoulder0p])-imput_SS1200[0])/(((wavelengths[shoulder0p])-540.84)*imput_SS1200[0])  #Calculating the slope beetwen the R540 and the shoulder
            stack_SSI.append(SS)
        
    stack_SSIa=np.array(stack_SSI)
    SSI.data=stack_SSIa.reshape(y,z)
    return SSI

#Clementine-like RGB. R: R750 nm/R540 nm, G:,R750 nm/R1000 nm, B:R540nm/R750 nm
def clementine (fourier_cube):
    clem=fourier_cube[0:3,:,:].copy()    
    B1=fourier_cube[6,:,:]/fourier_cube[0,:,:]  #Selecting the band to divide in the filtered cube
    B2=fourier_cube[6,:,:]/fourier_cube[19,:,:]
    B3=fourier_cube[0,:,:]/fourier_cube[6,:,:]
    clem.data=np.dstack((B1,B2,B3)).transpose(2,0,1)
    return clem

#RGB1. R: SSI, G: BDI, B: BDII
def RGB1 (fourier_cube,SSI_cube,BDI_cube,BII_cube):
    RGB1=fourier_cube[0:3,:,:].copy()
    RGB1.data=np.dstack((SSI_cube,BDI_cube,BII_cube)).transpose(2,0,1)
    return RGB1


#RGB2. R: SSBI, G: R540 nm, B: BCII
def RGB2 (fourier_cube,SSI_cube, R540_cube, BCII_cube):
    RGB2=fourier_cube[0:3,:,:].copy()
    RGB2.data=np.dstack((SSI_cube,R540_cube,BCII_cube)).transpose(2,0,1)
    return RGB2

#RGB3. R: SSBI, G: R540 nm, B: BCI
def RGB3 (fourier_cube,SSI_cube,R540_cube,BCI_cube):
    RGB3=fourier_cube[0:3,:,:].copy()
    RGB3.data=np.dstack((SSI_cube,R540_cube,BCI_cube)).transpose(2,0,1)
    return RGB3

#Olivine detection index
def olivine (fourier_filter):
    ol=(((fourier_filter[50,:,:]/((0.1*fourier_filter[21,:,:])+(0.1*fourier_filter[29,:,:])+(0.4*fourier_filter[35,:,:])+(0.4*fourier_filter[42,:,:])))-1))
    return ol