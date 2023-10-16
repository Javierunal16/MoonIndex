import numpy as np
import MoonIndex.preparation
import MoonIndex.filtration
import xarray as xa


#Calcualtes all the indexes for the convex hull method
def indexes_total_CH(M3_cube,wavelengths):
    
    #Filtration
    fourier_cube=MoonIndex.filtration.fourier_filter(M3_cube,60,2)
    gauss_cube=MoonIndex.filtration.gauss_filter(fourier_cube,wavelengths)  #Inputs are the original cube and wavelengths

    #Continuum removal
    midpoint_cube=MoonIndex.preparation.midpoint(gauss_cube,wavelengths,6,0.002)  #Inputs are the filtered cube, the wavelengths, and the distance and prominence of the peaks
    hull_cube=MoonIndex.preparation.convexhull_removal(gauss_cube,wavelengths,midpoint_cube)  #Inputs are the filtered cube, wavelengths and the midpoint
    
    #Indexes
    indexes_total=gauss_cube[0:28,:,:].copy()
        
    #Creating the minimums for the convex hull method
    M3_min1000ch, M3_min2000ch=MoonIndex.preparation.find_minimums_ch(hull_cube,midpoint_cube,wavelengths)
    #Obtaining the shoulders for the convex hull method
    M3_shoulder0ch, M3_shoulder1ch, M3_shoulder2ch, M3_shoulder3ch=MoonIndex.preparation.find_shoulders_ch(hull_cube,midpoint_cube,M3_min1000ch,M3_min2000ch,wavelengths)

    #General indexes
    #R540, reflectance at 540 nm (Zambon et al., 2020)
    M3_R540=R540(gauss_cube) 
    #R1580, reflectance at 540 nm (Besse et al., 2011)
    M3_R1580=R1580(gauss_cube) 
    #Spinel detection index (Moriarty III et al. 2022)
    M3_sp=spinel(gauss_cube)  
    #Olivine detection index  (Corley et al., 2018)
    M3_ol=olivine(gauss_cube) 
    #Chromtie detection index This work
    M3_cr=chromite(gauss_cube)
    #Iron detection index (Wu et al., 2012)
    M3_fe=iron(gauss_cube)
    #TiO detection index (Wu et al., 2012)
    M3_ti=titanium(gauss_cube)
    #Clementine-like RGB. R: R750 nm/R540 nm, G:,R750 nm/R1000 nm, B:R540nm/R750 nm
    M3_clem=clementine(gauss_cube)
    #RGB for mineral ratios. R: Pyroxene ratio, G: Spinel ratio, B:Anorthosite ratio (Pieters et al. 2014)
    M3_spanpx=RGB_spanpx(gauss_cube)

    #Convex hull indexes
    #BCI, band center ar 1000 nm
    M3_BCI_CH=band_center(M3_min1000ch)
    #BCII, band center ar 2000 nm
    M3_BCII_CH=band_center(M3_min2000ch)
    #BDI, band depth at 1000 nm with the convex hull method
    M3_BDI_CH=band_depth(hull_cube,M3_min1000ch,wavelengths)
    #BDII, band depth at 2000 nm with the convex hull method
    M3_BDII_CH=band_depth(hull_cube,M3_min2000ch,wavelengths)
    #SS1000, Spectral slope between maximun right shoulder and 540nm
    M3_SSI_CH=SSI(gauss_cube,M3_shoulder1ch,wavelengths) 
    #RGB8. R: band depth (BD) 1900, integrated band depth(IBD) 2000, integrated band depth (IBD) 1000
    M3_RGB8_CH=RGB8(gauss_cube,hull_cube)
    #BA1000
    M3_BAI1000_CH=BA(hull_cube,wavelengths,M3_shoulder0ch,M3_shoulder1ch)
    #ASY1000
    M3_ASY1000_CH=ASY(hull_cube,wavelengths,M3_shoulder0ch, M3_shoulder1ch,M3_min1000ch)
    #BA2000
    M3_BAI2000_CH=BA(hull_cube,wavelengths,M3_shoulder2ch,M3_shoulder3ch)
    #ASY2000
    M3_ASY2000_CH=ASY(hull_cube,wavelengths,M3_shoulder2ch, M3_shoulder3ch,M3_min2000ch)
    #RGB6
    M3_RGB6_CH=RGB6(hull_cube)
    
    #Creatinh the output cube
    indexes_total.data=np.dstack((M3_R540,M3_R1580,M3_sp,M3_ol,M3_cr,M3_fe,M3_ti,M3_clem[0],M3_clem[1],M3_clem[2],M3_spanpx[0],M3_spanpx[1],M3_spanpx[2],M3_BCI_CH,M3_BCII_CH,M3_BDI_CH,M3_BDII_CH,M3_SSI_CH,
                             M3_RGB8_CH[0],M3_RGB8_CH[1],M3_RGB8_CH[2],M3_BAI1000_CH,M3_ASY1000_CH,M3_BAI2000_CH,M3_ASY2000_CH,M3_RGB6_CH[0],M3_RGB6_CH[1],M3_RGB6_CH[2])).transpose(2,0,1)
    
    #Give name to the bands
    bands = ['Reflectance 540 nm','Reflectance 1580 nm','Spinel parameter (Moriarty, 2022)','Olivine parameter','Chromite parameter','Iron oxide parameter','Titanium parameter','Clementine RED','Clementine GREEN','Clementine BLUE','Pyroxene parameter',
             'Spinel parameter (Pieters, 2014)','Anorthosite (Pieters, 2014)','Band center 1 µm CH','Band center 2 µm CH','Band depth 1 µm CH','Band depth 2 µm CH',
             'Spectral slope 1 µm CH','Band depth 1.9 µm CH','Integrated band depth 2 µm CH', 'Integrated band depth 1 µm CH','Band area 1 µm CH','Band assymetry 1 µm CH','Band area 2 µm CH','Band assymetry 2 µm CH','Band depth at 950 nm CH', 'Band depth at 1.05 µm CH','Band depth at 1.25 µm CH' ]
    indexes_final_ch=xa.Dataset()

    for e in range(28):
        indexes_final_ch[bands[e]] = indexes_total[e,:,:]
        
    return(indexes_final_ch)


#All the indexes for the lienar fit method
def indexes_total_LF(M3_cube,wavelengths,order1,order2):
    
    #Filtration
    fourier_cube=MoonIndex.filtration.fourier_filter(M3_cube,60,2)
    gauss_cube=MoonIndex.filtration.gauss_filter(fourier_cube,wavelengths)  #Inputs are the original cube and wavelengths

    #Continuum removal
    lf_cube=MoonIndex.preparation.continuum_removal_lf(gauss_cube,wavelengths,order1,order2)  #Inputs are the filtered cube, wavelengths and the orders of polynomials
    
    indexes_total=gauss_cube[0:28,:,:].copy()
        
    #General indexes
    #R540, reflectance at 540 nm (Zambon et al., 2020)
    M3_R540=R540(gauss_cube) 
    #R1580, reflectance at 540 nm (Besse et al., 2011)
    M3_R1580=R1580(gauss_cube) 
    #Spinel detection index (Moriarty III et al. 2022)
    M3_sp=spinel(gauss_cube)  
    #Olivine detection index  (Corley et al., 2018)
    M3_ol=olivine(gauss_cube)  
    #Chromtie detection index This work
    M3_cr=chromite(gauss_cube)
    #Iron detection index (Wu et al., 2012)
    M3_fe=iron(gauss_cube)
    #TiO detection index (Wu et al., 2012)
    M3_ti=titanium(gauss_cube)
    #Clementine-like RGB. R: R750 nm/R540 nm, G:,R750 nm/R1000 nm, B:R540nm/R750 nm
    M3_clem=clementine(gauss_cube)
    #RGB for mineral ratios. R: Pyroxene ratio, G: Spinel ratio, B:Anorthosite ratio (Pieters et al. 2014)
    M3_spanpx=RGB_spanpx(gauss_cube)
    
    #Creating the minimmums with the linear fit method
    M3_min1000lf,M3_min2000lf=MoonIndex.preparation.find_minimuumslf(lf_cube,wavelengths)
    M3_shoulder0lf,M3_shoulder1lf,M3_shoulder2lf=MoonIndex.preparation.find_shoulders_lf(lf_cube,M3_min1000lf,M3_min2000lf,wavelengths)
    
    #Linear fit indexes
    #BCI, band center ar 1000 nm
    M3_BCI_LF=band_center(M3_min1000lf)
    #BCII, band center ar 2000 nm
    M3_BCII_LF=band_center(M3_min2000lf)
    #BDI, band depth at 1000 nm with the convex hull method
    M3_BDI_LF=band_depth(lf_cube,M3_min1000lf,wavelengths)
    #BDII, band depth at 2000 nm with the convex hull method
    M3_BDII_LF=band_depth(lf_cube,M3_min2000lf,wavelengths)
    #SS1000, Spectral slope between maximun right shoulder and 540nm
    M3_SSI_LF=SSI(gauss_cube,M3_shoulder1lf,wavelengths) 
    #RGB8. R: band depth (BD) 1900, integrated band depth(IBD) 2000, integrated band depth (IBD) 1000
    M3_RGB8_LF=RGB8(gauss_cube,lf_cube)
    #BAI1000
    M3_BAI1000_LF=BA(lf_cube,wavelengths,M3_shoulder0lf,M3_shoulder1lf)
    #ASY1000
    M3_ASY1000_LF=ASY(lf_cube,wavelengths,M3_shoulder0lf, M3_shoulder1lf,M3_min1000lf)
    #BAI2000
    M3_BAI2000_LF=BA(lf_cube,wavelengths,M3_shoulder1lf,M3_shoulder2lf)
    #ASY2000
    M3_ASY2000_LF=ASY(lf_cube,wavelengths,M3_shoulder1lf, M3_shoulder2lf,M3_min2000lf)
    #RGB6
    M3_RGB6_LF=RGB6(lf_cube)
    
    #Creatinh the output cube
    
    indexes_total.data=np.dstack((M3_R540,M3_R1580,M3_sp,M3_ol,M3_cr,M3_fe,M3_ti,M3_clem[0],M3_clem[1],M3_clem[2],M3_spanpx[0],M3_spanpx[1],M3_spanpx[2],M3_BCI_LF,M3_BCII_LF,
                                  M3_BDI_LF,M3_BDII_LF,M3_SSI_LF,M3_RGB8_LF[0],M3_RGB8_LF[1],M3_RGB8_LF[2],M3_BAI1000_LF,M3_ASY1000_LF,M3_BAI2000_LF,M3_ASY2000_LF,M3_RGB6_LF[0],M3_RGB6_LF[1],M3_RGB6_LF[2])).transpose(2,0,1)
    
    #Give name to the bands
    bands = ['Reflectance 540 nm','Reflectance 1580 nm','Spinel parameter (Moriarty, 2022)','Olivine parameter', 'Chromite parameter','Iron oxide parameter','Titanium parameter','Clementine RED','Clementine GREEN','Clementine BLUE','Pyroxene parameter','Spinel parameter (Pieters, 2014)','Anorthosite (Pieters, 2014)','Band center 1 µm LF','Band center 2 µm LF','Band depth 1 µm LF','Band depth 2 µm LF','Spectral slope 1 µm LF','Band depth 1.9 µm LF','Integrated band depth 2 µm LF', 'Integrated band depth 1 µm LF','Band area 1 µm LF','Band assymetry 1 µm LF','Band area 2 µm LF','Band assymetry 2 µm LF','Band depth at 950 nm LF', 'Band depth at 1.05 µm LF','Band depth at 1.25 µm LF']
    indexes_final_lf=xa.Dataset()

    for e in range(28):
        indexes_final_lf[bands[e]] = indexes_total[e,:,:]

    return(indexes_final_lf)


#R540, reflectance at 540 nm
def R540 (fourier_cube):
    cube_R540=fourier_cube[0,:,:]  #The first band corresponds to that wavelength
    cube_R540.data[cube_R540.data==0]=np.nan
    return cube_R540


#R1580, reflectance at 540 nm
def R1580 (fourier_cube):
    cube_R1580=fourier_cube[47,:,:]  #The first band corresponds to that wavelength
    cube_R1580.data[cube_R1580.data==0]=np.nan
    return cube_R1580


#Olivine detection index
def olivine (fourier_filter):
    ol=(((fourier_filter[50,:,:]/((0.1*fourier_filter[21,:,:])+(0.1*fourier_filter[29,:,:])+(0.4*fourier_filter[35,:,:])+(0.4*fourier_filter[42,:,:])))-1))
    return ol


#Anorthosite detection index
def spinel (gauss_cube):
    sp1=((((gauss_cube[31,:,:]-gauss_cube[6,:,:])/500)*1350)+(gauss_cube[31,:,:]))/gauss_cube[73,:,:]
    return sp1


#Chromtie detection index
def chromite (gauss_cube):
    cr=gauss_cube[0,:,:].copy()
    cr.data=((((gauss_cube[36,:,:]-gauss_cube[6,:,:])/600)*1500)+gauss_cube[36,:,:])/gauss_cube[77,:,:]
    return cr


#FeO detection index
def iron (gauss_cube):
    fe=gauss_cube[0,:,:].copy()
    fe=np.arctan(((gauss_cube[19,:,:]/gauss_cube[6,:,:])-1.19)/(gauss_cube[6,:,]-0.08))
    return fe


#TiO detction index
def titanium (gauss_cube):
    ti=gauss_cube[0,:,:].copy()
    ti=np.arctan(((gauss_cube[0,:,:]/gauss_cube[6,:,:])-0.71)/(gauss_cube[6,:,]-0.07))
    return ti

#Clementine-like RGB. R: R750 nm/R540 nm, G:,R750 nm/R1000 nm, B:R540nm/R750 nm
def clementine (fourier_cube):
    clem=fourier_cube[0:3,:,:].copy()    
    B1=fourier_cube[6,:,:]/fourier_cube[0,:,:]  #Selecting the band to divide in the filtered cube
    B2=fourier_cube[6,:,:]/fourier_cube[19,:,:]
    B3=fourier_cube[0,:,:]/fourier_cube[6,:,:]
    clem.data=np.dstack((B1,B2,B3)).transpose(2,0,1)
    clem.data[clem.data > 3]=0
    return clem


#Anorthosite,spinel and pyroxene detection index
def RGB_spanpx (gauss_cube):
    spanpx=gauss_cube[0:3,:,:].copy()  
    px=(gauss_cube[4,:,:]+gauss_cube[29,:,:])/gauss_cube[16,:,:]
    sp=gauss_cube[39,:,:]/gauss_cube[51,:,:]
    an=(gauss_cube[19,:,:]+gauss_cube[44,:,:])/gauss_cube[31,:,:]
    spanpx.data=np.dstack((px,sp,an)).transpose(2,0,1)
    spanpx.data[spanpx.data > 3]=0
    return spanpx

#Band centers
def band_center (minimum):
    band_center=minimum.copy()
    band_center.data[band_center.data==0]=np.nan
    return band_center


#Band depth
def band_depth (hull_cube,minimum,wavelengths3):
    cube_depth=hull_cube[0,:,:].copy()  #Copying the cube to save the results
    stack_depth=[]
    y,z=hull_cube[0,:,:].shape
    wavelengths=wavelengths3[0:74]
    
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
            
            input_depth=hull_cube.data[:,a,b]
                    
            if input_depth[39] == 0:   
                    stack_depth.append(0)
            else:
                input_min=minimum.data[a,b]
                pre_input_minp=np.where(wavelengths==input_min)[0]
                minp=int(pre_input_minp)
            
                band_depth=1 - input_depth[minp]  #Finding the value of the band depth
    
                stack_depth.append(band_depth)
            
    stack_deptha=np.array(stack_depth)
    cube_depth.data=stack_deptha.reshape(y,z)
    cube_depth.data[cube_depth.data==0]=np.nan
    return cube_depth


#SS1000, Spectral slope between maximun right shoulder and 540nm
def SSI (gauss_cube,shoulder1, wavelengths):
    SSI=gauss_cube[0,:,:].copy()
    stack_SSI=[]
    y,z=gauss_cube[0,:,:].shape
    for a in range(gauss_cube.data.shape[1]):
        for b in range(gauss_cube.data.shape[2]):
            
            input_SS1200=gauss_cube.data[:,a,b]
                    
            if input_SS1200[39] == 0:   
                    stack_SSI.append(0)
            else:
                input_shoulder1=shoulder1.data[a,b]
                pre_shoulder1=np.where(wavelengths==input_shoulder1)[0]
                shoulder1p=int(pre_shoulder1)
        
                SS=((input_SS1200[shoulder1p])-input_SS1200[0])/(((wavelengths[shoulder1p])-540.84)*input_SS1200[0])  #Calculating the slope beetwen the R540 and the shoulder
                stack_SSI.append(SS)
        
    stack_SSIa=np.array(stack_SSI)
    SSI.data=stack_SSIa.reshape(y,z)
    SSI.data[SSI.data==0]=np.nan
    return SSI


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


#RGB6. Band depths at 950, 1050 and 1250
def RGB6 (hull_cube):
    RGB6=hull_cube[0:3,:,:].copy()
    RGB6_R=1-hull_cube.data[16,:,:]
    RGB6_G=1-hull_cube.data[21,:,:]
    RGB6_B=1-hull_cube.data[31,:,:]
    RGB6.data=np.dstack((RGB6_R,RGB6_G,RGB6_B)).transpose(2,0,1)
    RGB6.data[RGB6.data==1]=np.nan
    return RGB6


#RGB6. Reflectance at 1580, IBDI, IBDII
def RGB7 (gauss_cube,R1580,IBD1000,IBD200):
    RGB7=gauss_cube[0:3,:,:].copy()
    RGB7.data=np.dstack((R1580,IBD1000,IBD200)).transpose(2,0,1)
    return RGB7


#BAI1000
def BA (hull_cube,wavelengths,shoulder0,shoulder1):
    y,z=hull_cube[0,:,:].shape
    SR=np.diff(wavelengths)  #Finding the spectral resolution, neccesary to find the area
    SR=np.append(39.92,SR)   #Adding the first value

    #Calculating the band area
    BAI=hull_cube[0,:,:].copy()
    stack_BAI=[]
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
            
            s0 = shoulder0.data[a,b]  #The shoulders limits the area calculation
            s1 = shoulder1.data[a,b]
                    
            if s0 == 0:   
                    stack_BAI.append(0)
            else:
                start = np.where(wavelengths == s0)[0][0].item()  #Creating the range
                end = np.where(wavelengths == s1)[0][0].item()
           
                input_SR= SR[start:end]
                input_CCA= hull_cube.data[:,a,b]
            
                sum3=0
                for c in range(start, end):
                
                    sum3 += ((1 - input_CCA[c-start]) * input_SR[c-start])  #Calculates the area by te sum of the spectral resolution multiplied by 1 minus the reflectance
                
                stack_BAI.append(sum3)
        
    stack_BAIa=np.array(stack_BAI)
    BAI.data=stack_BAIa.reshape(y,z)
    BAI.data[BAI.data==0]=np.nan
    return BAI


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
                    
            if s0 == 0:   
                    stack_BAI.append(0)
            else:
                start = np.where(wavelengths == s0)[0][0].item()  #Creating the range
                end = np.where(wavelengths == s1)[0][0].item()
           
                input_SR= SR[start:end]
                input_CCA= fourier_cube.data[:,a,b]
            
                sum3=0
                for c in range(start, end):
                
                    sum3 += ((1 - input_CCA[c-start]) * input_SR[c-start])  #Calculates the area by te sum of the spectral resolution multiplied by 1 minus the reflectance
                
                stack_BAI.append(sum3)
        
    stack_BAIa=np.array(stack_BAI)
    BAI.data=stack_BAIa.reshape(y,z)

    #Creating the RGB
    CCA=fourier_cube[0:3,:,:].copy()
    CCA.data=np.dstack((minimum_1000,minimum_2000,BAI)).transpose(2,0,1)
    CCA.data[CCA.data==0]=np.nan
    return CCA


#Asimetry 1000 nm
def ASY (hull_cube,wavelengths,shoulder0,shoulder1,min1000):
    SR=np.diff(wavelengths)  #Finding the spectral resolution, neccesary to find the area
    SR=np.append(39.92,SR)   #Adding the first value
    
    #Caculating the asymmetry
    y,z=hull_cube[0,:,:].shape
    ASY=hull_cube[0,:,:].copy()

    stack_ASY1=[]
    stack_ASY2=[]
    stack_ASY3=[]
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
            
            
            s00 = shoulder0.data[a,b]  #The asimmetry is also calculated inside the shoulders
            s11 = shoulder1.data[a,b]
            input_min1000=min1000.data[a,b]
            
            if s00 == 0:   
                    stack_ASY1.append(0)
                    stack_ASY2.append(0)
                    
            else:
                start1 = np.where(wavelengths == s00)[0][0].item()  #Definnig the range
                end1 = np.where(wavelengths == s11)[0][0].item()
                middle=np.where(wavelengths == input_min1000)[0][0].item()
           
                input_SR1= SR[start1:middle]
                input_SR2= SR[middle:end1]
                input_CCA= hull_cube.data[:,a,b]
            
                sum4=0
                for c in range(start1, middle):
                
                    sum4 += ((1 - input_CCA[c-start1]) * input_SR1[c-start1])  #Calculating the area of the first half of the zone
                
                stack_ASY1.append(sum4)
            
                sum5=0
                
                for d in range(middle, end1):
                
                    sum5 += ((1 - input_CCA[d-middle]) * input_SR2[d-middle])  #Calculating the area of the second half of the zone
                
                stack_ASY2.append(sum5)         
            
    #Asimetry calculation
    sum_ASY=np.add(stack_ASY1,stack_ASY2)  #Calcualting the total area

    for a in range(len(stack_ASY2)):
        
        if stack_ASY1[a] ==0:
            
            stack_ASY3.append(0)
                
        elif stack_ASY1[a] > stack_ASY2[a]:  #If the left side area is bigger, the asmmetry is negative
                    
            stack_ASY3.append (-(((stack_ASY1[a]-stack_ASY2[a])*100)/sum_ASY[a]))  #The asymmetry is the difference beetwen the two areas when dividing the peak in half, it is given in as a pecentage of the total area
                    
        else:  #If the right side area is bigger, the asmmetry is positive
                
            stack_ASY3.append((stack_ASY2[a]-stack_ASY1[a])*100/sum_ASY[a])

    stack_ASY3a=np.array(stack_ASY3)
    ASY.data=stack_ASY3a.reshape(y,z)
    ASY.data[ASY.data==0]=np.nan
    return ASY


#RGB5. R:ASY, G:BCI, B: BCII, this index combines the band asymmetry at 1000 with the center at 2000 and the band area at 1000
def RGB5 (fourier_cube,wavelengths,shoulder0,shoulder1,min1000,min2000):
    SR=np.diff(wavelengths)  #Finding the spectral resolution, neccesary to find the area
    SR=np.append(39.92,SR)   #Adding the first value
    
    #Caculating the asymmetry
    y,z=fourier_cube[0,:,:].shape
    ASY=fourier_cube[0,:,:].copy()

    stack_ASY1=[]
    stack_ASY2=[]
    stack_ASY3=[]
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            
            s00 = shoulder0.data[a,b]  #The asimmetry is also calculated inside the shoulders
            s11 = shoulder1.data[a,b]
            input_min1000=min1000.data[a,b]
            
            if s00 == 0:   
                    stack_ASY1.append(0)
                    stack_ASY2.append(0)
                    
            else:
                start1 = np.where(wavelengths == s00)[0][0].item()  #Definnig the range
                end1 = np.where(wavelengths == s11)[0][0].item()
                middle=np.where(wavelengths == input_min1000)[0][0].item()
           
                input_SR1= SR[start1:middle]
                input_SR2= SR[middle:end1]
                input_CCA= fourier_cube.data[:,a,b]
            
                sum4=0
                for c in range(start1, middle):
                
                    sum4 += ((1 - input_CCA[c-start1]) * input_SR1[c-start1])  #Calculating the area of the first half of the zone
                
                stack_ASY1.append(sum4)
            
                sum5=0
                
                for d in range(middle, end1):
                
                    sum5 += ((1 - input_CCA[d-middle]) * input_SR2[d-middle])  #Calculating the area of the second half of the zone
                
                stack_ASY2.append(sum5)         
            
    #Asimetry calculation
    sum_ASY=np.add(stack_ASY1,stack_ASY2)  #Calcualting the total area

    for a in range(len(stack_ASY2)):
        
        if stack_ASY1[a] ==0:
            
            stack_ASY3.append(0)
                
        elif stack_ASY1[a] > stack_ASY2[a]:  #If the left side area is bigger, the asmmetry is negative
                    
            stack_ASY3.append (-(((stack_ASY1[a]-stack_ASY2[a])*100)/sum_ASY[a]))  #The asymmetry is the difference beetwen the two areas when dividing the peak in half, it is given in as a pecentage of the total area
                    
        else:  #If the right side area is bigger, the asmmetry is positive
                
            stack_ASY3.append((stack_ASY2[a]-stack_ASY1[a])*100/sum_ASY[a])

    stack_ASY3a=np.array(stack_ASY3)
    ASY.data=stack_ASY3a.reshape(y,z)
    RGB5=fourier_cube[0:3,:,:].copy()
    RGB5.data=np.dstack((ASY,min1000,min2000)).transpose(2,0,1)
    RGB5.data[RGB5.data==0]=np.nan
    return RGB5


#Integrated band depth at 1000 nm
def IBDII(hull_cube):
    y2000,z2000=hull_cube[0,:,:].shape
    IBDII=hull_cube[0,:,:].copy()
    IBDII_slice=hull_cube[49:70,:,:]  #Defines the section to iterate around 2000 nm
    stack_IBDII=[]
    
    for a in range(IBDII_slice.data.shape[1]):
        for b in range(IBDII_slice.data.shape[2]):
            sum1=0
            if IBDII_slice[19,a,b]==0: 
                    stack_IBDII.append(0)
            else:
                for c in range(IBDII_slice.data.shape[0]):
                    input_hull=IBDII_slice.data[:,a,b]
                    sum1 += (1- input_hull[c])  #Summatory
                stack_IBDII.append(sum1)
        
    stack_IBDIIa=np.array(stack_IBDII)
    IBDII.data=stack_IBDIIa.reshape(y2000,z2000)
    IBDII.data[IBDII.data==0]=np.nan
    return IBDII
    
    
#Integrated band depth at 1000 nm
def IBDI(hull_cube):
    y2000,z2000=hull_cube[0,:,:].shape
    IBDI=hull_cube[0,:,:].copy()
    IBDI_slice=hull_cube[8:34,:,:]  #Defines the section to iterate around 2000 nm
    stack_IBDI=[]
    
    for a in range(IBDI_slice.data.shape[1]):
        for b in range(IBDI_slice.data.shape[2]):
            sum1=0
            if IBDI_slice[19,a,b]==0: 
                    stack_IBDI.append(0)
            else:
                for c in range(IBDI_slice.data.shape[0]):
                    input_hull=IBDI_slice.data[:,a,b]
                    sum1 += (1- input_hull[c])  
                stack_IBDI.append(sum1)
        
    stack_IBDIa=np.array(stack_IBDI)
    IBDI.data=stack_IBDIa.reshape(y2000,z2000)
    IBDI.data[IBDI.data==0]=np.nan
    return IBDI

    
#RGB8 Color 1, R: BD 1900, IBD 2000, IBD 1000
def RGB8 (fourier_cube,hull_cube):
    y1000,z1000=hull_cube[0,:,:].shape
    y2000,z2000=hull_cube[0,:,:].shape
    #Band 1. Finds the band depth at 1900 by dividing the reflectance by the continumm value
    RGB81= 1 - (hull_cube[55,:,:])
    RGB81.data[RGB81.data==1]=np.nan
    
    #Band 2 The integrated band depth at 2000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 2000 nm region 
    RGB82=hull_cube[0,:,:].copy()
    RGB82_slice=hull_cube[49:70,:,:]  #Defines the section to iterate around 2000 nm
    stack_RGB82=[]
    
    for a in range(RGB82_slice.data.shape[1]):
        for b in range(RGB82_slice.data.shape[2]):
            sum1=0
            if RGB82_slice[19,a,b]==0: 
                    stack_RGB82.append(0)
            else:
                for c in range(RGB82_slice.data.shape[0]):
                    input_hull=RGB82_slice.data[:,a,b]
                    sum1 += (1- input_hull[c])  #Summatory
                stack_RGB82.append(sum1)
        
    stack_RGB82a=np.array(stack_RGB82)
    RGB82.data=stack_RGB82a.reshape(y2000,z2000)
    
     #Band 3 The integrated band depth at 1000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 1000 nm region
    RGB83=hull_cube[0,:,:].copy()
    RGB83_slice=hull_cube[8:34,:,:]  #Defines the section to iterate around 2000 nm
    stack_RGB83=[]
    
    for a in range(RGB83_slice.data.shape[1]):
        for b in range(RGB83_slice.data.shape[2]):
            sum2=0
            if RGB83_slice[19,a,b]==0: 
                    stack_RGB83.append(0)
            else:
                for c in range(RGB83_slice.data.shape[0]):
                    input_hull4=RGB83_slice.data[:,a,b]
                    sum2 += (1-input_hull4[c])
                stack_RGB83.append(sum2)
            
    stack_RGB83a=np.array(stack_RGB83)
    RGB83.data=stack_RGB83a.reshape(y1000,z1000)
    
    #Making the composite
    RGB8_total=fourier_cube[0:3,:,:].copy()
    RGB8_total.data=np.dstack((RGB81,RGB82,RGB83)).transpose(2,0,1)
    RGB8_total.data[RGB8_total.data==0]=np.nan
    return RGB8_total

##INDEXES WITH THE LIENAR FIT REMOVAL METHOD



#BDI, band depth at the 1000 nm absorption peak, it is obtained by dividing the reflectance by the value of the continnum at that location, always positive
def LFBDI (fourier_cube, hull_cube, wavelengths):
    cube_BDI=hull_cube[0,:,:].copy()  #Copying the cube to save the results
    stack_BDI=[]
    y,z=hull_cube[0,:,:].shape
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            input_BDI=fourier_cube.data[:,a,b]
            input_hull=hull_cube.data[:,a,b]
        
            rfl_1000p=np.where(input_hull[0:29] == min(input_hull[0:29]))[0]  #Finding the value of the reflectance
            rfl_1000=input_BDI[rfl_1000p]
         
            fits_1000BDI=MoonIndex.preparation.continnum_1000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Fitting at 1000 nm
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
            
            input_BDI=fourier_cube.data[:,a,b]
            input_hull=hull_cube.data[:,a,b]
        
            rfl_2000p=np.where(input_hull[35:75] == min(input_hull[35:75]))[0]+35 #Finding the value of the reflectance
            rfl_2000=input_BDI[rfl_2000p]
         
            fits_2000BDI=MoonIndex.preparation.continnum_2000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Fitting at 2000 nm
            fitrfl_2000=np.polyval(fits_2000BDI,wavelengths[rfl_2000p])  #Obtainign the value of the continumm with the fit
            Depth2000=(1-(rfl_2000/fitrfl_2000))  #DOing the division
        
            stack_BDII.append(Depth2000)
            
    stack_BDIIa=np.array(stack_BDII)
    cube_BDII.data=stack_BDIIa.reshape(y,z)
    return cube_BDII


#RGB8 Color 1, R: BD 1900, IBD 2000, IBD 1000
def RGB8_LF (fourier_cube,hull_cube,wavelengths):
    y,z=hull_cube[0,:,:].shape
    #Band 1. Finds the band depth at 1900 by dividing the reflectance by the continumm value
    RGB81=(1 - (fourier_cube[55,:,:]/((fourier_cube[70,:,:]-fourier_cube[39,:,:]/2498-1408)*((1898-1408)+fourier_cube[39,:,:]))))
    
    #Band 2 The integrated band depth at 2000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 2000 nm region 
    RGB82=fourier_cube[0,:,:].copy()
    RGB82_slice=fourier_cube[49:70,:,:]  #Defines the section to iterate around 2000 nm
    stack_RGB82=[]
    
    for a in range(RGB82_slice.data.shape[1]):
        for b in range(RGB82_slice.data.shape[2]):
            for c in range(RGB82_slice.data.shape[0]):
                sum1=0
                input_RGB8=fourier_cube.data[c,a,b]
                input_hull=hull_cube.data[:,a,b]
                
                fits_2000RGB82=MoonIndex.preparation.continnum_2000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Continumm function to get the value
                fitRGB8_2000=np.polyval(fits_2000RGB82, wavelengths[c])
            
                sum1 += (1-(input_RGB8/fitRGB8_2000))  #Summatory
            stack_RGB82.append(sum1)
        
    stack_RGB82a=np.array(stack_RGB82)
    RGB82.data=stack_RGB82a.reshape(y,z)
    
     #Band 3 The integrated band depth at 1000 is calcualted as the summatory of 1 minus the factor beetwen the reflectance and continnum value of the band that makes the 1000 nm region
    RGB83=fourier_cube[0,:,:].copy()
    RGB83_slice=fourier_cube[8:34,:,:]  #Defines the section to iterate around 2000 nm
    stack_RGB83=[]
    
    for a in range(RGB83_slice.data.shape[1]):
        for b in range(RGB83_slice.data.shape[2]):
            for c in range(RGB83_slice.data.shape[0]):
                sum2=0
                input_RGB82=fourier_cube.data[c,a,b]
                input_hull4=hull_cube.data[:,a,b]
            
                fits_2000RGB83=MoonIndex.preparation.continnum_1000(fourier_cube.data, hull_cube.data,wavelengths,b,a)  #Continumm function to get the value
                fitRGB8_1000=np.polyval(fits_2000RGB83, wavelengths[c])
            
                sum2 += (1-(input_RGB82/fitRGB8_1000))
            stack_RGB83.append(sum2)
            
    stack_RGB83a=np.array(stack_RGB83)
    RGB83.data=stack_RGB83a.reshape(y,z)
    
    #Making the composite
    RGB8_total=fourier_cube[0:3,:,:].copy()
    RGB8_total.data=np.dstack((RGB81,RGB82,RGB83)).transpose(2,0,1)
    return RGB8_total