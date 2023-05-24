import numpy as np
import M3spectral.preparation


#R540, reflectance at 540 nm
def R540 (fourier_cube):
    cube_R540=fourier_cube[0,:,:]  #The first band corresponds to that wavelength
    return cube_R540

#BDI, band depth at 1000 nm with the convex hull method
def CH_BDI (hull_cube,min1000,wavelengths3):
    cube_CHBDI=hull_cube[0,:,:].copy()  #Copying the cube to save the results
    stack_CHBDI=[]
    y,z=hull_cube[0,:,:].shape
    wavelengths=wavelengths3[0:76]
    
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
            
            imput_CH=hull_cube.data[:,a,b]
            imput_min1000=min1000.data[a,b]
            pre_imput_min1000p=np.where(wavelengths==imput_min1000)[0]
            min1000p=int(pre_imput_min1000p)
            
            CHBDI_1000=1 - imput_CH[min1000p]  #Finding the value of the band depth
    
            stack_CHBDI.append(CHBDI_1000)
            
    stack_CHBDIa=np.array(stack_CHBDI)
    cube_CHBDI.data=stack_CHBDIa.reshape(y,z)
    return cube_CHBDI


#BDII, band depth at 2000 nm with the convex hull method
def CH_BDII (hull_cube, min2000,wavelengths3):
    cube_CHBDII=hull_cube[0,:,:].copy()  #Copying the cube to save the results
    stack_CHBDII=[]
    y,z=hull_cube[0,:,:].shape
    wavelengths=wavelengths3[0:76]
    
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
            
            imput_CH=hull_cube.data[:,a,b]
            imput_min2000=min2000.data[a,b]
            pre_imput_min2000p=np.where(wavelengths==imput_min2000)[0]
            min2000p=int(pre_imput_min2000p)
        
            CHBDI_2000=1 - imput_CH[min2000p]  #Finding the value of the band depth
    
            stack_CHBDII.append(CHBDI_2000)
            
    stack_CHBDIIa=np.array(stack_CHBDII)
    cube_CHBDII.data=stack_CHBDIIa.reshape(y,z)
    return cube_CHBDII


#SS1000, Spectral slope between maximun right shoulder and 540nm
def SSI (fourier_cube,shoulder1, wavelengths):
    SSI=fourier_cube[0,:,:].copy()
    stack_SSI=[]
    y,z=fourier_cube[0,:,:].shape
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
        
            imput_SS1200=fourier_cube.data[:,a,b]
            imput_shoulder1=shoulder1.data[a,b]
            pre_shoulder1=np.where(wavelengths==imput_shoulder1)[0]
            shoulder1p=int(pre_shoulder1)
        
            SS=((imput_SS1200[shoulder1p])-imput_SS1200[0])/(((wavelengths[shoulder1p])-540.84)*imput_SS1200[0])  #Calculating the slope beetwen the R540 and the shoulder
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
def RGB2 (gauss_cube,SSI_cube, R540_cube, BCII_cube):
    RGB2=gauss_cube[0:3,:,:].copy()
    RGB2.data=np.dstack((SSI_cube,R540_cube,BCII_cube)).transpose(2,0,1)
    return RGB2


#RGB3. R: SSBI, G: R540 nm, B: BCI
def RGB3 (gauss_cub,SSI_cube,R540_cube,BCI_cube):
    RGB3=gauss_cub[0:3,:,:].copy()
    RGB3.data=np.dstack((SSI_cube,R540_cube,BCI_cube)).transpose(2,0,1)
    return RGB3


#Olivine detection index
def olivine (fourier_filter):
    ol=(((fourier_filter[50,:,:]/((0.1*fourier_filter[21,:,:])+(0.1*fourier_filter[29,:,:])+(0.4*fourier_filter[35,:,:])+(0.4*fourier_filter[42,:,:])))-1))
    return ol


#RGB4. R:BCI, G: BCII, B:BAI, this index combines the band centers wit the band area at 1000 nm
def RGB4 (fourier_cube,wavelengths,shoulder0,shoulder1,minimum_1000,minimum_2000):
    y,z=fourier_cube[0,:,:].shape
    SR=np.diff(wavelengths)  #Finding the spectral resolution, neccesary to find the area
    SR=np.append(39.92,SR)   #Adding the first value

    #Calculating the band area
    BAI=fourier_cube[0,:,:].copy()
    stack_BAI=[]
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            s0 = shoulder0.data[a,b]  #The shoulders limits the area calculation
            s1 = shoulder1.data[a,b]
        
            start = np.where(wavelengths == s0)[0][0].item()  #Creating the range
            end = np.where(wavelengths == s1)[0][0].item()
        
           
            imput_SR= SR[start:end]
            imput_CCA= fourier_cube.data[:,a,b]
            
            sum3=0
            for c in range(start, end):
                
                sum3 += ((1 - imput_CCA[c-start]) * imput_SR[c-start])  #Calculates the area by te sum of the spectral resolution multiplied by 1 minus the reflectance
                
            stack_BAI.append(sum3)
        
    stack_BAIa=np.array(stack_BAI)
    BAI.data=stack_BAIa.reshape(y,z)

    #Creating the RGB
    CCA=fourier_cube[0:3,:,:].copy()
    CCA.data=np.dstack((minimum_1000,minimum_2000,BAI)).transpose(2,0,1)
    return CCA


#RGB5. R:ASY, G:BCI, B: BCII, this index combines the band asymmetry at 1000 with the center at 2000 and the band area at 1000
def RGB5 (fourier_cube,wavelengths,shoulder0,shoulder1,min1000,min2000):
    SR=np.diff(wavelengths)  #Finding the spectral resolution, neccesary to find the area
    SR=np.append(39.92,SR)   #Adding the first value
    
    #Caculating the asymmetry
    y,z=fourier_cube[0,:,:].shape
    ASY=fourier_cube[0,:,:].copy()

    stack_ASY1=[]
    stack_ASY2=[]
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            s00 = shoulder0.data[a,b]  #The asimmetry is also calculated inside the shoulders
            s11 = shoulder1.data[a,b]
            input_min1000=min1000.data[a,b]

            start1 = np.where(wavelengths == s00)[0][0].item()  #Definnig the range
            end1 = np.where(wavelengths == s11)[0][0].item()
            middle=np.where(wavelengths == input_min1000)[0][0].item()
           
            imput_SR1= SR[start1:middle]
            imput_SR2= SR[middle:end1]
            imput_CCA= fourier_cube.data[:,a,b]
            
            sum4=0
            for c in range(start1, middle):
                
                sum4 += ((1 - imput_CCA[c-start1]) * imput_SR1[c-start1])  #Calculating the area of the first half of the zone
                
            stack_ASY1.append(sum4)
            
            
            sum5=0
            for d in range(middle, end1):
                
                sum5 += ((1 - imput_CCA[d-middle]) * imput_SR2[d-middle])  #Calculating the area of the second half of the zone
                
            stack_ASY2.append(sum5)         
            
    #Asimetry calculation
    sum_ASY=np.add(stack_ASY1,stack_ASY2)  #Calcualting the total area
    stack_ASY3=[]

    for a in range(len(stack_ASY2)):
                
        if stack_ASY1[a] > stack_ASY2[a]:  #If the left side area is bigger, the asmmetry is negative
                    
            stack_ASY3.append (-(((stack_ASY1[a]-stack_ASY2[a])*100)/sum_ASY[a]))  #The asymmetry is the difference beetwen the two areas when dividing the peak in half, it is given in as a pecentage of the total area
                    
        else:  #If the right side area is bigger, the asmmetry is positive
                
            stack_ASY3.append((stack_ASY2[a]-stack_ASY1[a])*100/sum_ASY[a])

    stack_ASY3a=np.array(stack_ASY3)
    ASY.data=stack_ASY3a.reshape(y,z)
    RGB5=fourier_cube[0:3,:,:].copy()
    RGB5.data=np.dstack((ASY,min1000,min2000)).transpose(2,0,1)
    return RGB5


#NIR Color 1, R: BD 1900, IBD 2000, IBD 1000
def CH_NIR (fourier_cube,hull_cube):
    y1000,z1000=hull_cube[0,:,:].shape
    y2000,z2000=hull_cube[0,:,:].shape
    #Band 1. Finds the band depth at 1900 by dividing the reflectance by the continumm value
    NIR1= 1 - (fourier_cube[55,:,:])/(((fourier_cube[70,:,:]-fourier_cube[39,:,:])/(2498-1408)*(1898-1408))+fourier_cube[39,:,:])

    
    #Band 2 The integrated band depth at 2000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 2000 nm region 
    NIR2=hull_cube[0,:,:].copy()
    NIR2_slice=hull_cube[49:70,:,:]  #Defines the section to iterate around 2000 nm
    stack_NIR2=[]
    
    for a in range(NIR2_slice.data.shape[1]):
        for b in range(NIR2_slice.data.shape[2]):
            for c in range(NIR2_slice.data.shape[0]):
                sum1=0
                
                imput_hull=hull_cube.data[:,a,b]
            
                sum1 += (1- imput_hull[c])  #Summatory
                
            stack_NIR2.append(sum1)
        
    stack_NIR2a=np.array(stack_NIR2)
    NIR2.data=stack_NIR2a.reshape(y2000,z2000)
    
     #Band 3 The integrated band depth at 1000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 1000 nm region
    NIR3=hull_cube[0,:,:].copy()
    NIR3_slice=hull_cube[8:34,:,:]  #Defines the section to iterate around 2000 nm
    stack_NIR3=[]
    
    for a in range(NIR3_slice.data.shape[1]):
        for b in range(NIR3_slice.data.shape[2]):
            for c in range(NIR3_slice.data.shape[0]):
                sum2=0
    
                imput_hull4=hull_cube.data[:,a,b]
            
                sum2 += (1-imput_hull4[c])
                
            stack_NIR3.append(sum2)
            
    stack_NIR3a=np.array(stack_NIR3)
    NIR3.data=stack_NIR3a.reshape(y1000,z1000)
    
    #Making the composite
    NIR_total=fourier_cube[0:3,:,:].copy()
    NIR_total.data=np.dstack((NIR1,NIR2,NIR3)).transpose(2,0,1)
    return NIR_total



##INDEXES WITH THE LIENAR FIT REMOVAL METHOD



#BDI, band depth at the 1000 nm absorption peak, it is obtained by dividing the reflectance by the value of the continnum at that location, always positive
def LFBDI (fourier_cube, hull_cube, wavelengths):
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
def LFBDII (fourier_cube, hull_cube, wavelengths):
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


#NIR Color 1, R: BD 1900, IBD 2000, IBD 1000
def NIR (fourier_cube,hull_cube,wavelengths):
    y,z=hull_cube[0,:,:].shape
    #Band 1. Finds the band depth at 1900 by dividing the reflectance by the continumm value
    NIR1=(1 - (fourier_cube[55,:,:]/((fourier_cube[70,:,:]-fourier_cube[39,:,:]/2498-1408)*((1898-1408)+fourier_cube[39,:,:]))))
    
    #Band 2 The integrated band depth at 2000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 2000 nm region 
    NIR2=fourier_cube[0,:,:].copy()
    NIR2_slice=fourier_cube[49:70,:,:]  #Defines the section to iterate around 2000 nm
    stack_NIR2=[]
    
    for a in range(NIR2_slice.data.shape[1]):
        for b in range(NIR2_slice.data.shape[2]):
            for c in range(NIR2_slice.data.shape[0]):
                sum1=0
                imput_nir=fourier_cube.data[c,a,b]
                imput_hull=hull_cube.data[:,a,b]
                
                fits_2000NIR2=M3spectral.preparation.continnum_2000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Continumm function to get the value
                fitnir_2000=np.polyval(fits_2000NIR2, wavelengths[c])
            
                sum1 += (1-(imput_nir/fitnir_2000))  #Summatory
            stack_NIR2.append(sum1)
        
    stack_NIR2a=np.array(stack_NIR2)
    NIR2.data=stack_NIR2a.reshape(y,z)
    
     #Band 3 The integrated band depth at 1000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 1000 nm region
    NIR3=fourier_cube[0,:,:].copy()
    NIR3_slice=fourier_cube[8:34,:,:]  #Defines the section to iterate around 2000 nm
    stack_NIR3=[]
    
    for a in range(NIR3_slice.data.shape[1]):
        for b in range(NIR3_slice.data.shape[2]):
            for c in range(NIR3_slice.data.shape[0]):
                sum2=0
                imput_nir2=fourier_cube.data[c,a,b]
                imput_hull4=hull_cube.data[:,a,b]
            
                fits_2000NIR3=M3spectral.preparation.continnum_1000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Continumm function to get the value
                fitnir_1000=np.polyval(fits_2000NIR3, wavelengths[c])
            
                sum2 += (1-(imput_nir2/fitnir_1000))
            stack_NIR3.append(sum2)
            
    stack_NIR3a=np.array(stack_NIR3)
    NIR3.data=stack_NIR3a.reshape(y,z)
    
    #Making the composite
    NIR_total=fourier_cube[0:3,:,:].copy()
    NIR_total.data=np.dstack((NIR1,NIR2,NIR3)).transpose(2,0,1)
    return NIR_total