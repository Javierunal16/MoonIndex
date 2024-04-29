import numpy as np
import MoonIndex.preparation
import MoonIndex.filtration
import xarray as xa


###CALCULATING ALL THE INDEXES

def indexes_total_CH(M3_cube,wavelengths):
    '''This function peforms the full procces of creating the indexes using the convex-hull removal method, from the filtering to the indexes generation. The attach_wave (cube_alone,wave) function must still be runned beforehand, but the user can inputs the full cube after that (will take a long time), or crop it with crop_cube (initial_cube,minnx,minny,maxx,maxy) to save time. 
    
    Inputs:
    M3_cube = the cube, 
    wavelengths = the wavelengths.
    
    Outputs:
    An image with all the indexes proccesed (CH).'''
    
    #Filtration
    fourier_cube=MoonIndex.filtration.fourier_filter(M3_cube,60,2)
    #Inputs are the original cube and wavelengths
    gauss_cube=MoonIndex.filtration.gauss_filter(fourier_cube,wavelengths)  

    #Continuum removal
    #Inputs are the filtered cube, the wavelengths, and the distance and prominence of the peaks
    midpoint_cube=MoonIndex.preparation.midpoint(gauss_cube,wavelengths,6,0.002)  
    #Inputs are the filtered cube, wavelengths and the midpoint
    hull_cube=MoonIndex.preparation.convexhull_removal(gauss_cube,wavelengths,midpoint_cube)  
    
    #This copy the original cube to maintain coordiantes, 28 is for the total number of indexes
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
    
    #Giving names to the bands
    bands = ['Reflectance 540 nm','Reflectance 1580 nm','Spinel parameter (Moriarty, 2022)','Olivine parameter','Chromite parameter','Iron oxide parameter','Titanium parameter','Clementine RED','Clementine GREEN','Clementine BLUE','Pyroxene parameter',
             'Spinel parameter (Pieters, 2014)','Anorthosite (Pieters, 2014)','Band center 1 µm CH','Band center 2 µm CH','Band depth 1 µm CH','Band depth 2 µm CH',
             'Spectral slope 1 µm CH','Band depth 1.9 µm CH','Integrated band depth 2 µm CH', 'Integrated band depth 1 µm CH','Band area 1 µm CH','Band assymetry 1 µm CH','Band area 2 µm CH','Band assymetry 2 µm CH','Band depth at 950 nm CH', 'Band depth at 1.05 µm CH','Band depth at 1.25 µm CH' ]
    indexes_final_ch=xa.Dataset()

    for e in range(28):
        indexes_final_ch[bands[e]] = indexes_total[e,:,:]
    return(indexes_final_ch.astype(np.float32))


def indexes_total_SAFO(M3_cube,wavelengths,order1,order2):
    '''This function peforms the full procces of creating the indexes using the second-and-first-order fit removal method, from the filtering to the indexes generation. The attach_wave (cube_alone,wave) function must still be runned beforehand, but the user can inputs the full cube after that (will take a long time), or crop it with crop_cube (initial_cube,minnx,minny,maxx,maxy) to save time. 
    
    Inputs:
    M3_cube = the cube, 
    wavelengths = the wavelengths,
    order1 = polynomial order for the first absorption band,
    order2 = polynomial order for the second absorption band.
    
    Outputs:
    An image with all the indexes proccesed (SAFO).'''
    
    #Filtration
    fourier_cube=MoonIndex.filtration.fourier_filter(M3_cube,60,2)
    #Inputs are the original cube and wavelengths
    gauss_cube=MoonIndex.filtration.gauss_filter(fourier_cube,wavelengths)  

    #Continuum removal
    #Inputs are the filtered cube, wavelengths and the orders of polynomials
    SAFO_cube=MoonIndex.preparation.continuum_removal_SAFO(gauss_cube,wavelengths,order1,order2)  
    #This copy the original cube to maintain coordiantes, 28 is for the total number of indexes
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
    
    #Creating the minimmums with the second-and-first-order fit method
    M3_min1000SAFO,M3_min2000SAFO=MoonIndex.preparation.find_minimums_SAFO(SAFO_cube,wavelengths)
    M3_shoulder0SAFO,M3_shoulder1SAFO,M3_shoulder2SAFO=MoonIndex.preparation.find_shoulders_SAFO(SAFO_cube,M3_min1000SAFO,M3_min2000SAFO,wavelengths)
    
    #second-and-first-order fit indexes
    #BCI, band center ar 1000 nm
    M3_BCI_SAFO=band_center(M3_min1000SAFO)
    #BCII, band center ar 2000 nm
    M3_BCII_SAFO=band_center(M3_min2000SAFO)
    #BDI, band depth at 1000 nm with the second-and-first-order fit method
    M3_BDI_SAFO=band_depth(SAFO_cube,M3_min1000SAFO,wavelengths)
    #BDII, band depth at 2000 nm with the second-and-first-order fit method
    M3_BDII_SAFO=band_depth(SAFO_cube,M3_min2000SAFO,wavelengths)
    #SS1000, Spectral slope between maximun right shoulder and 540nm
    M3_SSI_SAFO=SSI(gauss_cube,M3_shoulder1SAFO,wavelengths) 
    #RGB8. R: band depth (BD) 1900, integrated band depth(IBD) 2000, integrated band depth (IBD) 1000
    M3_RGB8_SAFO=RGB8(gauss_cube,SAFO_cube)
    #BAI1000
    M3_BAI1000_SAFO=BA(SAFO_cube,wavelengths,M3_shoulder0SAFO,M3_shoulder1SAFO)
    #ASY1000
    M3_ASY1000_SAFO=ASY(SAFO_cube,wavelengths,M3_shoulder0SAFO, M3_shoulder1SAFO,M3_min1000SAFO)
    #BAI2000
    M3_BAI2000_SAFO=BA(SAFO_cube,wavelengths,M3_shoulder1SAFO,M3_shoulder2SAFO)
    #ASY2000
    M3_ASY2000_SAFO=ASY(SAFO_cube,wavelengths,M3_shoulder1SAFO, M3_shoulder2SAFO,M3_min2000SAFO)
    #RGB6
    M3_RGB6_SAFO=RGB6(SAFO_cube)
    
    #Creatinh the output cube
    
    indexes_total.data=np.dstack((M3_R540,M3_R1580,M3_sp,M3_ol,M3_cr,M3_fe,M3_ti,M3_clem[0],M3_clem[1],M3_clem[2],M3_spanpx[0],M3_spanpx[1],M3_spanpx[2],M3_BCI_SAFO,M3_BCII_SAFO,
                                  M3_BDI_SAFO,M3_BDII_SAFO,M3_SSI_SAFO,M3_RGB8_SAFO[0],M3_RGB8_SAFO[1],M3_RGB8_SAFO[2],M3_BAI1000_SAFO,M3_ASY1000_SAFO,M3_BAI2000_SAFO,M3_ASY2000_SAFO,M3_RGB6_SAFO[0],M3_RGB6_SAFO[1],M3_RGB6_SAFO[2])).transpose(2,0,1)
    
    #Give name to the bands
    bands = ['Reflectance 540 nm','Reflectance 1580 nm','Spinel parameter (Moriarty, 2022)','Olivine parameter', 'Chromite parameter','Iron oxide parameter','Titanium parameter','Clementine RED','Clementine GREEN','Clementine BLUE','Pyroxene parameter','Spinel parameter (Pieters, 2014)','Anorthosite (Pieters, 2014)','Band center 1 µm SAFO','Band center 2 µm SAFO','Band depth 1 µm SAFO','Band depth 2 µm SAFO','Spectral slope 1 µm SAFO','Band depth 1.9 µm SAFO','Integrated band depth 2 µm SAFO', 'Integrated band depth 1 µm SAFO','Band area 1 µm SAFO','Band assymetry 1 µm SAFO','Band area 2 µm SAFO','Band assymetry 2 µm SAFO','Band depth at 950 nm SAFO', 'Band depth at 1.05 µm SAFO','Band depth at 1.25 µm SAFO']
    indexes_final_SAFO=xa.Dataset()

    for e in range(28):
        indexes_final_SAFO[bands[e]] = indexes_total[e,:,:]
    return(indexes_final_SAFO.astype(np.float32))

###INDIVIDUAL INDEXES

def R540 (gauss_cube):
    '''Creates the reflectance at 540 nm. 
    
    Input: 
    gauss_cube = the filtered cube.
    
    Output:
    R540 index.'''

    #The first band corresponds to that wavelength
    cube_R540=gauss_cube[0,:,:]  
    cube_R540.data[cube_R540.data==0]=np.nan
    return cube_R540


def R1580 (gauss_cube):
    '''Creates the reflectance at 1580 nm. 
    
    Input: 
    gauss_cube = the filtered cube.
    
    Output:
    R1580 index.'''

    #The first band corresponds to that wavelength
    cube_R1580=gauss_cube[47,:,:]  
    cube_R1580.data[cube_R1580.data==0]=np.nan
    return cube_R1580


def olivine (gauss_cube):
    '''Creates the olivine index. 
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    Olivine index.'''
    
    ol=(((gauss_cube[50,:,:]/((0.1*gauss_cube[21,:,:])+(0.1*gauss_cube[29,:,:])+(0.4*gauss_cube[35,:,:])+(0.4*gauss_cube[42,:,:])))-1))
    return ol


def spinel (gauss_cube):
    '''Creates the spinel index. 
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    Spinel index.'''
    
    sp1=((((gauss_cube[31,:,:]-gauss_cube[6,:,:])/500)*1350)+(gauss_cube[31,:,:]))/gauss_cube[73,:,:]
    return sp1


def chromite (gauss_cube):
    '''Creates the chromite index. 
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    Chromite index.'''
    
    cr=gauss_cube[0,:,:].copy()
    cr.data=((((gauss_cube[36,:,:]-gauss_cube[6,:,:])/600)*1500)+gauss_cube[36,:,:])/gauss_cube[77,:,:]
    return cr


def iron (gauss_cube):
    '''Creates the FeO index. 
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    FeO index.'''
    
    fe=gauss_cube[0,:,:].copy()
    fe=np.arctan(((gauss_cube[19,:,:]/gauss_cube[6,:,:])-1.19)/(gauss_cube[6,:,]-0.08))
    return fe


def titanium (gauss_cube):
    '''Creates the TiO index. 
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    TiO index.'''
    
    ti=gauss_cube[0,:,:].copy()
    ti=np.arctan(((gauss_cube[0,:,:]/gauss_cube[6,:,:])-0.71)/(gauss_cube[6,:,]-0.07))
    return ti


def clementine (gauss_cube):
    '''Creates the clementine-like index. 
    
    Input:
    gauss_cube = fitlered cube.
    
    Output:
    The clementine-like RGB composite.'''
    
    clem=gauss_cube[0:3,:,:].copy() 
    #Selecting the band to divide the filtered cube
    B1=gauss_cube[6,:,:]/gauss_cube[0,:,:]  
    B2=gauss_cube[6,:,:]/gauss_cube[19,:,:]
    B3=gauss_cube[0,:,:]/gauss_cube[6,:,:]
    clem.data=np.dstack((B1,B2,B3)).transpose(2,0,1)
    clem.data[clem.data > 3]=0
    return clem


def RGB_spanpx (gauss_cube):
    '''Creates the spanpx (spinel-anorthosite-pyroxene) index. 
    
    Input:
    gauss_cube = fitlered cube.
    
    Output:
    The spanpx RGB composite.'''
    
    spanpx=gauss_cube[0:3,:,:].copy()  
    px=(gauss_cube[4,:,:]+gauss_cube[29,:,:])/gauss_cube[16,:,:]
    sp=gauss_cube[39,:,:]/gauss_cube[51,:,:]
    an=(gauss_cube[19,:,:]+gauss_cube[44,:,:])/gauss_cube[31,:,:]
    spanpx.data=np.dstack((px,sp,an)).transpose(2,0,1)
    spanpx.data[spanpx.data > 3]=0
    return spanpx


def band_center (minimum):
    '''Creates the band minimuum, it works for both absorption bands by changing the corresponding inputs. 
    
    Input: 
    minimum = the minimum image.
    
    Output:
    The minimum of the selected abosprtion band.'''
    
    band_center=minimum.copy()
    band_center.data[band_center.data==0]=np.nan
    return band_center



def band_depth (removed_cube,minimum,wavelengths):
    '''Creates the band depth, it works for both absorption bands by changing the corresponding inputs. 
    
    Inputs:
    removed_cube = continuum-removed cube,
    minimum = the minimum image,
    wavelengths = the wavelegnths.
    
    Output:
    The band depth of the selected abosprtion band.'''

    #Copying the cube to save the results
    cube_depth=removed_cube[0,:,:].copy()  
    stack_depth=[]
    y,z=removed_cube[0,:,:].shape
    wavelengths=wavelengths[0:74]
    
    for a in range(removed_cube.data.shape[1]):
        for b in range(removed_cube.data.shape[2]):
            
            input_depth=removed_cube.data[:,a,b]
                    
            if minimum[a,b] == 0:   
                    stack_depth.append(0)
            else:
                input_min=minimum.data[a,b]
                pre_input_minp=np.where(wavelengths==input_min)[0][0]
                minp=int(pre_input_minp)
                #Finding the value of the band depth
                band_depth=1 - input_depth[minp]  
    
                stack_depth.append(band_depth)
            
    stack_deptha=np.array(stack_depth)
    cube_depth.data=stack_deptha.reshape(y,z)
    cube_depth.data[cube_depth.data==0]=np.nan
    return cube_depth


def SSI (gauss_cube,shoulder1, wavelengths):
    '''Creates the sprectral slope at 1 um index. This is done between the 540 nm band and the left shoudler of the 1 um band.
    
    Inputs: 
    gauss_cube = the filtered cube, 
    shoulder1 = the right shoudler of the 1 um band,
    wavelengths = the wavelengths.
    
    Output:
    The spectral slope at 1 um.'''
    
    SSI=gauss_cube[0,:,:].copy()
    stack_SSI=[]
    y,z=gauss_cube[0,:,:].shape
    for a in range(gauss_cube.data.shape[1]):
        for b in range(gauss_cube.data.shape[2]):
            
            input_SS1200=gauss_cube.data[:,a,b]
                    
            if shoulder1[a,b] == 0:   
                    stack_SSI.append(0)
            else:
                input_shoulder1=shoulder1.data[a,b]
                pre_shoulder1=np.where(wavelengths==input_shoulder1)[0]
                shoulder1p=int(pre_shoulder1)
                #Calculating the slope beetwen the R540 and the shoulder
                SS=((input_SS1200[shoulder1p])-input_SS1200[0])/(((wavelengths[shoulder1p])-0.54084)*0.54084)  
                stack_SSI.append(SS)
        
    stack_SSIa=np.array(stack_SSI)
    SSI.data=stack_SSIa.reshape(y,z)
    SSI.data[SSI.data==0]=np.nan
    return SSI


def RGB1 (gauss_cube,SSI_cube,BDI_cube,BDII_cube):
    '''Creates the RGB1 index. R: SSBI, G: BDI, B: BDII.
    
    Inputs:
    gauss_cube = the filtered cube, 
    SSI_cube = the spectral slope index,
    BDI_cube = the band depth at 1 um,
    BDII_cube = the band depth at 2 um.
    
    Output:
    The RGB1 RGB composite.'''
    
    RGB1=gauss_cube[0:3,:,:].copy()
    RGB1.data=np.dstack((SSI_cube,BDI_cube,BDII_cube)).transpose(2,0,1)
    return RGB1


def RGB2 (gauss_cube,SSI_cube, R540_cube, BCII_cube):
    '''Creates the RGB2 index. R: SSBI, G: R540, B: BCII.
    
    Inputs:
    gauss_cube = the filtered cube, 
    SSI_cube = the spectral slope index,
    R540_cube = the reflectance at 540 nm,
    BCII_cube = the band center at 2 um.
    
    Output:
    The RGB2 RGB composite.'''
    
    RGB2=gauss_cube[0:3,:,:].copy()
    RGB2.data=np.dstack((SSI_cube,R540_cube,BCII_cube)).transpose(2,0,1)
    return RGB2


def RGB3 (gauss_cube,SSI_cube,R540_cube,BCI_cube):
    '''Creates the RGB3 index. R: SSBI, G: R540, B: BCI.
    
    Inputs:
    gauss_cube = the filtered cube, 
    SSI_cube = the spectral slope index,
    R540_cube = the reflectance at 540 nm,
    BCI_cube = the band center at 1 um.
    
    Output:
    The RGB3 RGB composite.'''
    
    RGB3=gauss_cube[0:3,:,:].copy()
    RGB3.data=np.dstack((SSI_cube,R540_cube,BCI_cube)).transpose(2,0,1)
    return RGB3


def RGB6 (removed_cube):
    '''Creates the RGB6 index. R: BD950, G: BD1050, B: BD1250.
    
    Inputs:
    removed_cube = the continuum-removed cube.
    
    Output:
    The RGB6 RGB composite.'''
    
    RGB6=removed_cube[0:3,:,:].copy()
    RGB6_R=1-removed_cube.data[16,:,:]
    RGB6_G=1-removed_cube.data[21,:,:]
    RGB6_B=1-removed_cube.data[31,:,:]
    RGB6.data=np.dstack((RGB6_R,RGB6_G,RGB6_B)).transpose(2,0,1)
    RGB6.data[RGB6.data==1]=np.nan
    return RGB6


def RGB7 (gauss_cube,R1580,IBD1000,IBD2000):
    '''Creates the RGB6 index. R: R1580, G: IBDI, B: IBDII.
    
    Inputs:
    gauss_cube = the filtered cube.
    R1580 = the reflecance at 1580,
    IBD1000 = the integrated band depth at 1 um,
    IBD2000 = the integrated band depth at 2 um.
    
    Output:
    The RGB7 RGB composite.'''
    
    RGB7=gauss_cube[0:3,:,:].copy()
    RGB7.data=np.dstack((R1580,IBD1000,IBD2000)).transpose(2,0,1)
    return RGB7


def BA (removed_cube,wavelengths,shoulder0,shoulder1):
    '''Creates the band area index, it works for both absorption bands by changing the corresponding inputs. 
    
    Inputs: 
    removed_cube = the continuum-removed cube,
    wavelengths = the wavelengths,
    shoulder0 = the left shoulder of the band, 
    shoulder1 = the right shoudler of the band.
    
    Output:
    The band area of the selected absorption band.'''
    
    y,z=removed_cube[0,:,:].shape
    #Finding the spectral resolution, neccesary to find the area
    SR=np.diff(wavelengths) 
    #Adding the first value
    SR=np.append(39.92,SR)   

    #Calculating the band area
    BAI=removed_cube[0,:,:].copy()
    stack_BAI=[]
    for a in range(removed_cube.data.shape[1]):
        for b in range(removed_cube.data.shape[2]):
            #The shoulders limits the area calculation
            s0 = shoulder0.data[a,b]  
            s1 = shoulder1.data[a,b]
                    
            if s0 == 0 or s1 == 0:   
                    stack_BAI.append(0)
            else:
                 #Creating the range
                start = np.where(wavelengths == s0)[0][0].item() 
                end = np.where(wavelengths == s1)[0][0].item()
           
                input_SR= SR[start:end]
                input_CCA= removed_cube.data[:,a,b]
            
                sum3=0
                for c in range(start, end):
                    #Calculates the area by te sum of the spectral resolution multiplied by 1 minus the reflectance
                    sum3 += ((1 - input_CCA[c-start]) * input_SR[c-start])  
                
                stack_BAI.append(sum3)
        
    stack_BAIa=np.array(stack_BAI)
    BAI.data=stack_BAIa.reshape(y,z)
    BAI.data[BAI.data==0]=np.nan
    return BAI


def RGB4 (gauss_cube,wavelengths,shoulder0,shoulder1,minimum_1000,minimum_2000):
    '''Creates the RGB4 index. R: BCI, G: BCII, B: BAI.
    
    Inputs:
    gauss_cube = the filtered cube,
    wavelengths = the wavelengths,
    shoulder0 = the left shoulder of the band, 
    shoulder1 = the right shoudler of the band,
    minimum_1000 = the band minimum at 1 um,
    minimum_2000 = the band minimum at 2 um.
    
    Output:
    The RGB4 RGB composite.'''
    
    y,z=gauss_cube[0,:,:].shape
    #Finding the spectral resolution, neccesary to find the area
    SR=np.diff(wavelengths)  
    #Adding the first value
    SR=np.append(39.92,SR)  

    #Calculating the band area
    BAI=gauss_cube[0,:,:].copy()
    stack_BAI=[]
    for a in range(gauss_cube.data.shape[1]):
        for b in range(gauss_cube.data.shape[2]):
            #The shoulders limits the area calculation
            s0 = shoulder0.data[a,b]  
            s1 = shoulder1.data[a,b]
                    
            if s0 == 0 or s1 == 0:   
                    stack_BAI.append(0)
            else:
                #Creating the range
                start = np.where(wavelengths == s0)[0][0].item()  
                end = np.where(wavelengths == s1)[0][0].item()
           
                input_SR= SR[start:end]
                input_CCA= gauss_cube.data[:,a,b]
            
                sum3=0
                for c in range(start, end):
                    #Calculates the area by te sum of the spectral resolution multiplied by 1 minus the reflectance
                    sum3 += ((1 - input_CCA[c-start]) * input_SR[c-start])  
                
                stack_BAI.append(sum3)
        
    stack_BAIa=np.array(stack_BAI)
    BAI.data=stack_BAIa.reshape(y,z)

    #Creating the RGB
    CCA=gauss_cube[0:3,:,:].copy()
    CCA.data=np.dstack((minimum_1000,minimum_2000,BAI)).transpose(2,0,1)
    CCA.data[CCA.data==0]=np.nan
    return CCA


#Asimetry 1000 nm
def ASY (removed_cube,wavelengths,shoulder0,shoulder1,min1000):
    '''Creates the band asymmetry index, it works for both absorption bands by changing the corresponding inputs. 
    
    Inputs: 
    removed_cube = the continuum-removed cube,
    wavelengths = the wavelengths,
    shoulder0 = the left shoudler of the band, 
    shoulder1 = the right shoudler of the band, 
    min1000 = the minimum of the selected band.
    
    Output:
    Band asymmetry of the selected band.'''
    
    #Finding the spectral resolution, neccesary to find the area
    SR=np.diff(wavelengths)  
    #Adding the first value
    SR=np.append(39.92,SR)   
    
    #Caculating the asymmetry
    y,z=removed_cube[0,:,:].shape
    ASY=removed_cube[0,:,:].copy()

    stack_ASY1=[]
    stack_ASY2=[]
    stack_ASY3=[]
    for a in range(removed_cube.data.shape[1]):
        for b in range(removed_cube.data.shape[2]):
            
            #The asymmetry is also calculated inside the shoulders
            s00 = shoulder0.data[a,b]  
            s11 = shoulder1.data[a,b]
            input_min1000=min1000.data[a,b]
            
            if s00 == 0 or s11 == 0:   
                    stack_ASY1.append(0)
                    stack_ASY2.append(0)
                    
            else:
                #Definnig the range
                start1 = np.where(wavelengths == s00)[0][0].item()  
                end1 = np.where(wavelengths == s11)[0][0].item()
                middle=np.where(wavelengths == input_min1000)[0][0].item()
           
                input_SR1= SR[start1:middle]
                input_SR2= SR[middle:end1]
                input_CCA= removed_cube.data[:,a,b]
            
                sum4=0
                for c in range(start1, middle):
                    #Calculating the area of the first half of the zone
                    sum4 += ((1 - input_CCA[c-start1]) * input_SR1[c-start1])  
                
                stack_ASY1.append(sum4)
            
                sum5=0
                
                for d in range(middle, end1):
                    #Calculating the area of the second half of the zone
                    sum5 += ((1 - input_CCA[d-middle]) * input_SR2[d-middle])  
                
                stack_ASY2.append(sum5)         
            
    #Asimetry calculation
    #Calcualting the total area
    sum_ASY=np.add(stack_ASY1,stack_ASY2)  

    for a in range(len(stack_ASY2)):
        
        if stack_ASY1[a] ==0:
            
            stack_ASY3.append(0)
        #If the left side area is bigger, the asymmetry is negative  
        elif stack_ASY1[a] > stack_ASY2[a]:  
            #The asymmetry is the difference between the two areas when dividing the peak in half, it is given as a pecentage of the total area
            stack_ASY3.append (-(((stack_ASY1[a]-stack_ASY2[a])*100)/sum_ASY[a]))  
                    
        else:  
            #If the right side area is bigger, the asymmetry is positive    
            stack_ASY3.append((stack_ASY2[a]-stack_ASY1[a])*100/sum_ASY[a])

    stack_ASY3a=np.array(stack_ASY3)
    ASY.data=stack_ASY3a.reshape(y,z)
    ASY.data[ASY.data==0]=np.nan
    return ASY


def RGB5 (gauss_cube,wavelengths,shoulder0,shoulder1,min1000,min2000):
    '''Creates the RGB5 index. R: ASY, G: BCI, B: BCII.
    
    Inputs:
    gauss_cube = the filtered cube.
    wavelengths = the wavelengths,
    shoulder0 = the left shoudler of the band, 
    shoulder1 = the right shoudler of the band, 
    min1000 = the minimum at 1 um,
    min2000 = the minimum ar 2 um.
    
    Output:
    The RGB5 RGB composite.'''
    
    #Finding the spectral resolution, neccesary to find the area
    SR=np.diff(wavelengths) 
    #Adding the first value
    SR=np.append(39.92,SR)   
    
    #Caculating the asymmetry
    y,z=gauss_cube[0,:,:].shape
    ASY=gauss_cube[0,:,:].copy()

    stack_ASY1=[]
    stack_ASY2=[]
    stack_ASY3=[]
    for a in range(gauss_cube.data.shape[1]):
        for b in range(gauss_cube.data.shape[2]):
            
            #The asymmetry is also calculated inside the shoulders
            s00 = shoulder0.data[a,b]  
            s11 = shoulder1.data[a,b]
            input_min1000=min1000.data[a,b]
            
            if s00 == 0 or s11 == 0:   
                    stack_ASY1.append(0)
                    stack_ASY2.append(0)
                    
            else:
                #Definnig the range
                start1 = np.where(wavelengths == s00)[0][0].item()  
                end1 = np.where(wavelengths == s11)[0][0].item()
                middle=np.where(wavelengths == input_min1000)[0][0].item()
           
                input_SR1= SR[start1:middle]
                input_SR2= SR[middle:end1]
                input_CCA= gauss_cube.data[:,a,b]
            
                sum4=0
                for c in range(start1, middle):
                    #Calculating the area of the first half of the zone
                    sum4 += ((1 - input_CCA[c-start1]) * input_SR1[c-start1])  
                
                stack_ASY1.append(sum4)
            
                sum5=0
                
                for d in range(middle, end1):
                    #Calculating the area of the second half of the zone
                    sum5 += ((1 - input_CCA[d-middle]) * input_SR2[d-middle])  
                
                stack_ASY2.append(sum5)         
            
    #Asimetry calculation
    #Calcualting the total area
    sum_ASY=np.add(stack_ASY1,stack_ASY2)  

    for a in range(len(stack_ASY2)):
        
        if stack_ASY1[a] ==0:
            
            stack_ASY3.append(0)
        #If the left side area is bigger, the asmmetry is negative   
        elif stack_ASY1[a] > stack_ASY2[a]:  
            #The asymmetry is the difference beetwen the two areas when dividing the peak in half, it is given in as a pecentage of the total area        
            stack_ASY3.append (-(((stack_ASY1[a]-stack_ASY2[a])*100)/sum_ASY[a]))  
                    
        else:  
            #If the right side area is bigger, the asymmetry is positive    
            stack_ASY3.append((stack_ASY2[a]-stack_ASY1[a])*100/sum_ASY[a])

    stack_ASY3a=np.array(stack_ASY3)
    ASY.data=stack_ASY3a.reshape(y,z)
    RGB5=gauss_cube[0:3,:,:].copy()
    RGB5.data=np.dstack((ASY,min1000,min2000)).transpose(2,0,1)
    RGB5.data[RGB5.data==0]=np.nan
    return RGB5


def IBDII(removed_cube):
    '''Calculates the integrated band depth around the 2 um band. Internal process. 
    
    Input:
    removed_cube = the continuum-removed cube.
    
    Output:
    Integrated band dpeth at 2 um.'''
    
    y2000,z2000=removed_cube[0,:,:].shape
    IBDII=removed_cube[0,:,:].copy()
    #Defines the section to iterate around 2000 nm
    IBDII_slice=removed_cube[49:70,:,:]  
    stack_IBDII=[]
    
    for a in range(IBDII_slice.data.shape[1]):
        for b in range(IBDII_slice.data.shape[2]):
            sum1=0
            if IBDII_slice[19,a,b]==0: 
                    stack_IBDII.append(0)
            else:
                for c in range(IBDII_slice.data.shape[0]):
                    input_removed=IBDII_slice.data[:,a,b]
                    #Summatory
                    sum1 += (1- input_removed[c]) 
                stack_IBDII.append(sum1)
        
    stack_IBDIIa=np.array(stack_IBDII)
    IBDII.data=stack_IBDIIa.reshape(y2000,z2000)
    IBDII.data[IBDII.data==0]=np.nan
    return IBDII
    
    
def IBDI(removed_cube):
    '''Calculates the integrated band depth around the 1 um band. Internal process. 
    
    Input:
    removed_cube = the continuum-removed cube.
    
    Output:
    Integrated band dpeth at 1 um.'''
    
    y2000,z2000=removed_cube[0,:,:].shape
    IBDI=removed_cube[0,:,:].copy()
    #Defines the section to iterate around 1000 nm
    IBDI_slice=removed_cube[8:34,:,:]  
    stack_IBDI=[]
    
    for a in range(IBDI_slice.data.shape[1]):
        for b in range(IBDI_slice.data.shape[2]):
            sum1=0
            if IBDI_slice[19,a,b]==0: 
                    stack_IBDI.append(0)
            else:
                for c in range(IBDI_slice.data.shape[0]):
                    input_removed=IBDI_slice.data[:,a,b]
                    #Summatory
                    sum1 += (1- input_removed[c])  
                stack_IBDI.append(sum1)
        
    stack_IBDIa=np.array(stack_IBDI)
    IBDI.data=stack_IBDIa.reshape(y2000,z2000)
    IBDI.data[IBDI.data==0]=np.nan
    return IBDI


def RGB8 (gauss_cube,removed_cube):
    '''Creates the RGB5 index. R: BD1900, G: IBD2000, B: IBD1000.
    
    Inputs:
    gauss_cube = the filtered cube.
    removed_cube = continuum-removed cube.
    
    Output:
    The RGB8 RGB composite.'''
    
    y1000,z1000=removed_cube[0,:,:].shape
    y2000,z2000=removed_cube[0,:,:].shape
    #Band 1. Finds the band depth at 1900 by dividing the reflectance by the continumm value
    RGB81= 1 - (removed_cube[55,:,:])
    RGB81.data[RGB81.data==1]=np.nan
    
    #Band 2 The integrated band depth at 2000 is calcualted as the summatory of 1 minus the factor between the reflectance and continuum value of the band that makes the 2000 nm region 
    RGB82=removed_cube[0,:,:].copy()
    #Defines the section to iterate around 2000 nm
    RGB82_slice=removed_cube[49:70,:,:]  
    stack_RGB82=[]
    
    for a in range(RGB82_slice.data.shape[1]):
        for b in range(RGB82_slice.data.shape[2]):
            sum1=0
            if RGB82_slice[19,a,b]==0: 
                    stack_RGB82.append(0)
            else:
                for c in range(RGB82_slice.data.shape[0]):
                    input_removed=RGB82_slice.data[:,a,b]
                    #Summatory
                    sum1 += (1- input_removed[c])  
                stack_RGB82.append(sum1)
        
    stack_RGB82a=np.array(stack_RGB82)
    RGB82.data=stack_RGB82a.reshape(y2000,z2000)
    
     #Band 3 The integrated band depth at 1000 is calcualted as the summatory of 1 minus the factor between the reflectance and continuum value of the band that makes the 1000 nm region
    RGB83=removed_cube[0,:,:].copy()
    #Defines the section to iterate around 1000 nm
    RGB83_slice=removed_cube[8:34,:,:]  
    stack_RGB83=[]
    
    for a in range(RGB83_slice.data.shape[1]):
        for b in range(RGB83_slice.data.shape[2]):
            sum2=0
            if RGB83_slice[19,a,b]==0: 
                    stack_RGB83.append(0)
            else:
                for c in range(RGB83_slice.data.shape[0]):
                    input_removed4=RGB83_slice.data[:,a,b]
                    sum2 += (1-input_removed4[c])
                stack_RGB83.append(sum2)
            
    stack_RGB83a=np.array(stack_RGB83)
    RGB83.data=stack_RGB83a.reshape(y1000,z1000)
    
    #Making the composite
    RGB8_total=gauss_cube[0:3,:,:].copy()
    RGB8_total.data=np.dstack((RGB81,RGB82,RGB83)).transpose(2,0,1)
    RGB8_total.data[RGB8_total.data==0]=np.nan
    return RGB8_total