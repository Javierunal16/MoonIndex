import numpy as np
import MoonIndex.preparation
import MoonIndex.filtration
import xarray as xa
from joblib import Parallel, delayed
import multiprocessing
import time

###CALCULATING ALL THE INDEXES

def indexes_total_CH(M3_cube,wavelengths, n_jobs=None):
    '''This function performs the full process of creating the indexes using the convex-hull removal method, from the filtering to the indexes generation. The attach_wave (cube_alone,wave) function must still be runned beforehand, but the user can input the full cube after that (it will take a long time), or crop it with crop_cube (initial_cube,minnx,minny,maxx,maxy) to save time. 
    
    Inputs:
    M3_cube = the cube, 
    wavelengths = the wavelengths.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).
    
    Outputs:
    An image with all the indexes processed (CH).'''
    start_total = time.time()

    def log_time(msg, t0):
        print(f"{msg}: {time.time() - t0:.2f} s")

    print("=== Starting the proccesing using CH ===")

    # Filtering
    t0 = time.time()
    fourier_cube = MoonIndex.filtration.fourier_filter(M3_cube, 60, 2)
    log_time("Fourier filtering", t0)

    t0 = time.time()
    gauss_cube = MoonIndex.filtration.gauss_filter(fourier_cube, wavelengths)
    log_time("Gaussian filtering", t0)

    # Continuum removal
    t0 = time.time()
    midpoint_cube = MoonIndex.preparation.midpoint(gauss_cube, wavelengths, 6, 0.002, block_size=100, n_jobs=n_jobs)
    log_time("Midpoint calculation", t0)

    t0 = time.time()
    hull_cube = MoonIndex.preparation.convexhull_removal(gauss_cube, wavelengths, midpoint_cube, block_size=100, n_jobs=n_jobs)
    log_time("Convex hull removal", t0)

    # Copy cube
    indexes_total = gauss_cube[0:28, :, :].copy()

    # Minima and shoulders
    t0 = time.time()
    M3_min1000ch, M3_min2000ch = MoonIndex.preparation.find_minimums_ch(hull_cube, midpoint_cube, wavelengths, block_size=100, n_jobs=n_jobs)
    log_time("Minimums calculation", t0)

    t0 = time.time()
    M3_shoulder0ch, M3_shoulder1ch, M3_shoulder2ch, M3_shoulder3ch = MoonIndex.preparation.find_shoulders_ch(
        hull_cube, midpoint_cube, M3_min1000ch, M3_min2000ch, wavelengths, block_size=100, n_jobs=n_jobs)
    log_time("Shoulders calculation", t0)

    # General indexes
    t0 = time.time()
    M3_R540 = R540(gauss_cube)
    M3_R1580 = R1580(gauss_cube)
    M3_sp = spinel(gauss_cube)
    M3_ol = olivine(gauss_cube)
    M3_cr = chromite(gauss_cube)
    M3_fe = iron(gauss_cube)
    M3_ti = titanium(gauss_cube)
    M3_clem = clementine(gauss_cube)
    M3_spanpx = RGB_spanpx(gauss_cube)
    log_time("General indexes", t0)

    # Convex hull indexes
    t0 = time.time()
    M3_BCI_CH = band_center(M3_min1000ch)
    M3_BCII_CH = band_center(M3_min2000ch)
    M3_BDI_CH = band_depth(hull_cube, M3_min1000ch, wavelengths,block_size=100, n_jobs=n_jobs)
    M3_BDII_CH = band_depth(hull_cube, M3_min2000ch, wavelengths,block_size=100, n_jobs=n_jobs)
    log_time("Band measurements", t0)

    t0 = time.time()
    M3_SSI_CH = SSI(gauss_cube, M3_shoulder1ch, wavelengths, block_size=100, n_jobs=n_jobs)
    M3_RGB8_CH = RGB8(gauss_cube, hull_cube, block_size=100, n_jobs=n_jobs)
    M3_RGB6_CH = RGB6(hull_cube)
    log_time("Others", t0)

    t0 = time.time()
    M3_BAI1000_CH = BA(hull_cube, wavelengths, M3_shoulder0ch, M3_shoulder1ch)
    M3_ASY1000_CH = ASY(hull_cube, wavelengths, M3_shoulder0ch, M3_shoulder1ch, M3_min1000ch,block_size=100, n_jobs=n_jobs)
    M3_BAI2000_CH = BA(hull_cube, wavelengths, M3_shoulder2ch, M3_shoulder3ch)
    M3_ASY2000_CH = ASY(hull_cube, wavelengths, M3_shoulder2ch, M3_shoulder3ch, M3_min2000ch,block_size=100, n_jobs=n_jobs)
    log_time("Area and asymmetry", t0)

    # Output stack
    t0 = time.time()
    indexes_total.data = np.dstack((
        M3_R540, M3_R1580, M3_sp, M3_ol, M3_cr, M3_fe, M3_ti,
        M3_clem[0], M3_clem[1], M3_clem[2],
        M3_spanpx[0], M3_spanpx[1], M3_spanpx[2],
        M3_BCI_CH, M3_BCII_CH, M3_BDI_CH, M3_BDII_CH, M3_SSI_CH,
        M3_RGB8_CH[0], M3_RGB8_CH[1], M3_RGB8_CH[2],
        M3_BAI1000_CH, M3_ASY1000_CH, M3_BAI2000_CH, M3_ASY2000_CH,
        M3_RGB6_CH[0], M3_RGB6_CH[1], M3_RGB6_CH[2]
    )).transpose(2, 0, 1)
    log_time("Stacking output bands", t0)

    # Naming bands
    bands = ['Reflectance 540 nm', 'Reflectance 1580 nm', 'Spinel parameter (Moriarty, 2022)',
             'Olivine parameter', 'Chromite parameter', 'Iron oxide parameter', 'Titanium parameter',
             'Clementine RED', 'Clementine GREEN', 'Clementine BLUE', 'Pyroxene parameter',
             'Spinel parameter (Pieters, 2014)', 'Anorthosite (Pieters, 2014)', 'Band center 1 µm CH',
             'Band center 2 µm CH', 'Band depth 1 µm CH', 'Band depth 2 µm CH', 'Spectral slope 1 µm CH',
             'Band depth 1.9 µm CH', 'Integrated band depth 2 µm CH', 'Integrated band depth 1 µm CH',
             'Band area 1 µm CH', 'Band assymetry 1 µm CH', 'Band area 2 µm CH', 'Band assymetry 2 µm CH',
             'Band depth at 950 nm CH', 'Band depth at 1.05 µm CH', 'Band depth at 1.25 µm CH']

    indexes_final_ch = xa.Dataset()
    for e in range(28):
        indexes_final_ch[bands[e]] = indexes_total[e, :, :]

    log_time("Total time", start_total)
    print("=== Finsihing calculation ===")

    return indexes_final_ch.astype(np.float32)


def indexes_total_SAFO(M3_cube, wavelengths, order1, order2, n_jobs=None):
    '''This function performs the full process of creating the indexes using the second-and-first-order fit removal method, from the filtering to the indexes generation. The attach_wave (cube_alone,wave) function must still be run beforehand, but the user can input the full cube after that (will take a long time), or crop it with crop_cube (initial_cube,minnx,minny,maxx,maxy) to save time. 
    
    Inputs:
    M3_cube = the cube, 
    wavelengths = the wavelengths,
    order1 = polynomial order for the first absorption band,
    order2 = polynomial order for the second absorption band.
    n_jobs= Caution. Amount of cores for parallelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores). If None, defaults to 1 (no parallelism).
    
    Outputs:
    An image with all the indexes processed (SAFO).'''
    
    start_total = time.time()

    def log_time(msg, t0):
        print(f"{msg}: {time.time() - t0:.2f} s", flush=True)

    print("=== Starting the processing using SAFO ===", flush=True)

    # Filtering
    t0 = time.time()
    fourier_cube = MoonIndex.filtration.fourier_filter(M3_cube, 60, 2)
    gauss_cube = MoonIndex.filtration.gauss_filter(fourier_cube, wavelengths)  
    log_time("Filtering", t0)

    # Continuum removal
    t0 = time.time()
    SAFO_cube = MoonIndex.preparation.continuum_removal_SAFO(
        gauss_cube, wavelengths, order1, order2, block_size=100, n_jobs=n_jobs
    )
    log_time("Continuum removal (SAFO)", t0)

    # Copy to output cube
    indexes_total = gauss_cube[0:28, :, :].copy()

    # General indexes
    t0 = time.time()
    M3_R540 = R540(gauss_cube) 
    M3_R1580 = R1580(gauss_cube) 
    M3_sp = spinel(gauss_cube)  
    M3_ol = olivine(gauss_cube)  
    M3_cr = chromite(gauss_cube)
    M3_fe = iron(gauss_cube)
    M3_ti = titanium(gauss_cube)
    M3_clem = clementine(gauss_cube)
    M3_spanpx = RGB_spanpx(gauss_cube)
    log_time("General indexes", t0)

    # Find minimums and shoulders
    t0 = time.time()
    M3_min1000SAFO, M3_min2000SAFO = MoonIndex.preparation.find_minimums_SAFO(SAFO_cube, wavelengths, block_size=100, n_jobs=n_jobs)
    M3_shoulder0SAFO, M3_shoulder1SAFO, M3_shoulder2SAFO = MoonIndex.preparation.find_shoulders_SAFO(
        SAFO_cube, M3_min1000SAFO, M3_min2000SAFO, wavelengths, block_size=100, n_jobs=n_jobs)
    log_time("Minimums and shoulders", t0)

    # SAFO indexes
    t0 = time.time()
    M3_BCI_SAFO = band_center(M3_min1000SAFO)
    M3_BCII_SAFO = band_center(M3_min2000SAFO)
    M3_BDI_SAFO = band_depth(SAFO_cube, M3_min1000SAFO, wavelengths, block_size=100, n_jobs=n_jobs)
    M3_BDII_SAFO = band_depth(SAFO_cube, M3_min2000SAFO, wavelengths,block_size=100, n_jobs=n_jobs)
    log_time("Band measurements", t0)

    t0 = time.time()
    M3_SSI_SAFO = SSI(gauss_cube, M3_shoulder1SAFO, wavelengths, block_size=100, n_jobs=n_jobs)
    M3_RGB6_SAFO = RGB6(SAFO_cube)
    M3_RGB8_SAFO = RGB8(gauss_cube, SAFO_cube, block_size=100, n_jobs=n_jobs)
    log_time("Others", t0)

    t0 = time.time()
    M3_BAI1000_SAFO = BA(SAFO_cube, wavelengths, M3_shoulder0SAFO, M3_shoulder1SAFO)
    M3_ASY1000_SAFO = ASY(SAFO_cube, wavelengths, M3_shoulder0SAFO, M3_shoulder1SAFO, M3_min1000SAFO,block_size=100, n_jobs=n_jobs)
    M3_BAI2000_SAFO = BA(SAFO_cube, wavelengths, M3_shoulder1SAFO, M3_shoulder2SAFO)
    M3_ASY2000_SAFO = ASY(SAFO_cube, wavelengths, M3_shoulder1SAFO, M3_shoulder2SAFO, M3_min2000SAFO,block_size=100, n_jobs=n_jobs)
    log_time("Area and asymmetry", t0)

    # Output stack
    t0 = time.time()
    indexes_total.data = np.dstack((
        M3_R540, M3_R1580, M3_sp, M3_ol, M3_cr, M3_fe, M3_ti,
        M3_clem[0], M3_clem[1], M3_clem[2],
        M3_spanpx[0], M3_spanpx[1], M3_spanpx[2],
        M3_BCI_SAFO, M3_BCII_SAFO, M3_BDI_SAFO, M3_BDII_SAFO, M3_SSI_SAFO,
        M3_RGB8_SAFO[0], M3_RGB8_SAFO[1], M3_RGB8_SAFO[2],
        M3_BAI1000_SAFO, M3_ASY1000_SAFO, M3_BAI2000_SAFO, M3_ASY2000_SAFO,
        M3_RGB6_SAFO[0], M3_RGB6_SAFO[1], M3_RGB6_SAFO[2]
    )).transpose(2, 0, 1)
    log_time("Stacking output cube", t0)

    # Band naming
    t0 = time.time()
    bands = [
        'Reflectance 540 nm', 'Reflectance 1580 nm', 'Spinel parameter (Moriarty, 2022)',
        'Olivine parameter', 'Chromite parameter', 'Iron oxide parameter', 'Titanium parameter',
        'Clementine RED', 'Clementine GREEN', 'Clementine BLUE',
        'Pyroxene parameter', 'Spinel parameter (Pieters, 2014)', 'Anorthosite (Pieters, 2014)',
        'Band center 1 µm SAFO', 'Band center 2 µm SAFO',
        'Band depth 1 µm SAFO', 'Band depth 2 µm SAFO',
        'Spectral slope 1 µm SAFO',
        'Band depth 1.9 µm SAFO', 'Integrated band depth 2 µm SAFO', 'Integrated band depth 1 µm SAFO',
        'Band area 1 µm SAFO', 'Band asymmetry 1 µm SAFO', 'Band area 2 µm SAFO', 'Band asymmetry 2 µm SAFO',
        'Band depth at 950 nm SAFO', 'Band depth at 1.05 µm SAFO', 'Band depth at 1.25 µm SAFO'
    ]

    indexes_final_SAFO = xa.Dataset()
    for e in range(28):
        indexes_final_SAFO[bands[e]] = indexes_total[e, :, :]
    log_time("Dataset creation", t0)

    log_time("Total time SAFO", start_total)
    print("=== Finsihing calculation ===")
    return indexes_final_SAFO.astype(np.float32)

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
    '''Creates the FeO index. Which return iron oxide abundance in percentage. 
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    FeO index.'''
    
    fe=gauss_cube[0,:,:].copy()
    fep=-np.arctan(((gauss_cube[16,:,:]/gauss_cube[6,:,:])-1.26)/(gauss_cube[6,:,]-0.01))
    fe.data=8.878 * (fep**1.8732)
    return fe


def titanium (gauss_cube):
    '''Creates the TiO index. Which returns titanium oxide abundance in percentage.
    
    Input:
    gauss_cube = filtered cube.
    
    Output: 
    TiO index.'''
    
    ti=gauss_cube[0,:,:].copy()
    tip=np.arctan(((gauss_cube[0,:,:]/gauss_cube[6,:,:])-0.45)/(gauss_cube[6,:,]-0.05))
    ti.data=2.6275 * (tip**4.2964)
    return ti


def clementine (gauss_cube):
    '''Creates the clementine-like index. 
    
    Input:
    gauss_cube = filtered cube.
    
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
    gauss_cube = filtered cube.
    
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
    '''Creates the band minimum, it works for both absorption bands by changing the corresponding inputs. 
    
    Input: 
    minimum = the minimum image.
    
    Output:
    The minimum of the selected abosprtion band.'''
    
    band_center=minimum.copy()
    band_center.data[band_center.data==0]=np.nan
    return band_center



def band_depth(removed_cube, minimum, wavelengths, block_size=100, n_jobs=None):
    '''Creates the band depth, it works for both absorption bands by changing the corresponding inputs. 
    
    Inputs:
    removed_cube = continuum-removed cube,
    minimum = the minimum image,
    wavelengths = the wavelengths.
    block_size= Size of the block during the paralelization.
    n_jobs= Amount of cores for paralelization. Use -1 for all cores, or negative to leave some cores free.
    
    Output:
    The band depth of the selected absorption band.'''

    y, z = removed_cube[0, :, :].shape
    wavelengths = wavelengths[0:74]
    cube_depth = removed_cube[0, :, :].copy()

    # Resolve number of jobs
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        input_depth = removed_cube.data[:, a, b]
        if minimum[a, b] == 0:
            return 0
        else:
            input_min = minimum.data[a, b]
            pre_input_minp = np.where(wavelengths == input_min)[0][0]
            minp = int(pre_input_minp)
            band_depth_val = 1 - input_depth[minp]
            return band_depth_val

    def process_block(r_start, r_end, c_start, c_end):
        block_result = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                block_result[a - r_start, b - c_start] = process_pixel(a, b)
        return block_result

    blocks = []
    for r in range(0, y, block_size):
        for c in range(0, z, block_size):
            r_end = min(r + block_size, y)
            c_end = min(c + block_size, z)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    full_array = np.zeros((y, z), dtype=np.float32)
    for idx, (r_start, r_end, c_start, c_end) in enumerate(blocks):
        full_array[r_start:r_end, c_start:c_end] = results[idx]

    cube_depth.data = full_array
    cube_depth.data[cube_depth.data == 0] = np.nan

    return cube_depth


def SSI(gauss_cube, shoulder1, wavelengths, block_size=100, n_jobs=None):
    '''Creates the spectral slope at 1 um index. This is done between the 540 nm band and the left shoulder of the 1 um band.

    Inputs: 
    gauss_cube = the filtered cube, 
    shoulder1 = the right shoulder of the 1 um band,
    wavelengths = the wavelengths.
    block_size = size of processing block for parallelization.
    n_jobs = number of parallel jobs (cores).

    Output:
    The spectral slope at 1 um.'''

    SSI = gauss_cube[0, :, :].copy()
    y, z = gauss_cube[0, :, :].shape

    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_block(r_start, r_end, c_start, c_end):
        block_result = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                if shoulder1[a, b] == 0:
                    block_result[a - r_start, b - c_start] = 0
                else:
                    input_SS1200 = gauss_cube.data[:, a, b]
                    input_shoulder1 = shoulder1.data[a, b]
                    pre_shoulder1 = np.where(wavelengths == input_shoulder1)[0]
                    if len(pre_shoulder1) == 0:
                        block_result[a - r_start, b - c_start] = 0
                        continue
                    shoulder1p = int(pre_shoulder1[0])
                    SS = ((input_SS1200[shoulder1p]) - input_SS1200[0]) / (((wavelengths[shoulder1p]) - 0.54084) * 0.54084)
                    block_result[a - r_start, b - c_start] = SS
        return block_result

    blocks = []
    for r in range(0, y, block_size):
        for c in range(0, z, block_size):
            r_end = min(r + block_size, y)
            c_end = min(c + block_size, z)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    full_result = np.zeros((y, z), dtype=np.float32)
    for idx, (r_start, r_end, c_start, c_end) in enumerate(blocks):
        full_result[r_start:r_end, c_start:c_end] = results[idx]

    SSI.data = full_result
    SSI.data[SSI.data == 0] = np.nan

    return SSI


def RGB1 (gauss_cube,SSI_cube,BDI_cube,BDII_cube):
    '''Creates the RGB1 index. R: SSBI, G: BDI, B: BDII.
    
    Inputs:
    gauss_cube = the filtered cube, 
    SSI_cube = the spectral slope index,
    BDI_cube = the band depth at 1 um,
    BDII_cube = the band depth at 2 um.
    
    Output:
    The RGB1 composite.'''
    
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
    The RGB2 composite.'''
    
    RGB2=gauss_cube[0:3,:,:].copy()
    RGB2.data=np.dstack((SSI_cube,R540_cube,BCII_cube)).transpose(2,0,1)
    return RGB2


def RGB3 (gauss_cube,SSI_cube,R540_cube,BDI_cube):
    '''Creates the RGB3 index. R: SSBI, G: R540, B: BDI.
    
    Inputs:
    gauss_cube = the filtered cube, 
    SSI_cube = the spectral slope index,
    R540_cube = the reflectance at 540 nm,
    BDI_cube = the band depth at 1 um.
    
    Output:
    The RGB3 composite.'''
    
    RGB3=gauss_cube[0:3,:,:].copy()
    RGB3.data=np.dstack((SSI_cube,R540_cube,BDI_cube)).transpose(2,0,1)
    return RGB3


def RGB6 (removed_cube):
    '''Creates the RGB6 index. R: BD950, G: BD1050, B: BD1250.
    
    Inputs:
    removed_cube = the continuum-removed cube.
    
    Output:
    The RGB6 composite.'''
    
    RGB6=removed_cube[0:3,:,:].copy()
    RGB6_R=1-removed_cube.data[16,:,:]
    RGB6_G=1-removed_cube.data[21,:,:]
    RGB6_B=1-removed_cube.data[31,:,:]
    RGB6.data=np.dstack((RGB6_R,RGB6_G,RGB6_B)).transpose(2,0,1)
    RGB6.data[RGB6.data==1]=np.nan
    return RGB6


def RGB7 (gauss_cube,IBD1000,IBD2000, R1580):
    '''Creates the RGB6 index. R: IBDI, G: IBDII, B: R1580.
    
    Inputs:
    gauss_cube = the filtered cube.
    IBD1000 = the integrated band depth at 1 um,
    IBD2000 = the integrated band depth at 2 um,
    R1580 = the reflectance at 1580.
    
    Output:
    The RGB7 composite.'''
    
    RGB7=gauss_cube[0:3,:,:].copy()
    RGB7.data=np.dstack((IBD1000,IBD2000,R1580)).transpose(2,0,1)
    return RGB7


def BA (removed_cube,wavelengths,shoulder0,shoulder1):
    '''Creates the band area index, it works for both absorption bands by changing the corresponding inputs. 
    
    Inputs: 
    removed_cube = the continuum-removed cube,
    wavelengths = the wavelengths,
    shoulder0 = the left shoulder of the band, 
    shoulder1 = the right shoulder of the band.
    
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
    shoulder1 = the right shoulder of the band,
    minimum_1000 = the band minimum at 1 um,
    minimum_2000 = the band minimum at 2 um.
    
    Output:
    The RGB4 composite.'''
    
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


def ASY(removed_cube, wavelengths, shoulder0, shoulder1, min1000, block_size=100, n_jobs=None):
    '''Creates the band asymmetry index. Works for absorption bands by specifying appropriate shoulders and band minimum.

    Inputs: 
    removed_cube = the continuum-removed cube (reflectance values from 0 to 1),
    wavelengths = array of wavelength values,
    shoulder0 = left shoulder wavelength of the band,
    shoulder1 = right shoulder wavelength of the band,
    min1000 = wavelength of the band minimum.
    block_size = block size for parallel processing.
    n_jobs = number of cores to use. -1 for all cores, negative leaves that many cores free.

    Output:
    2D image of band asymmetry values for each pixel.
    '''

    # Compute spectral resolution
    SR = np.diff(wavelengths)
    SR = np.append(39.92, SR)  # Add the first resolution manually

    y, z = removed_cube[0, :, :].shape
    ASY = removed_cube[0, :, :].copy()

    # Configure number of parallel jobs
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    # Define block processor
    def process_block(r_start, r_end, c_start, c_end):
        block_result = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                if removed_cube[39, a, b] == 0:
                    block_result[a - r_start, b - c_start] = 0
                    continue

                s00 = shoulder0.data[a, b]
                s11 = shoulder1.data[a, b]
                input_min1000 = min1000.data[a, b]

                if s00 == 0 or s11 == 0:
                    block_result[a - r_start, b - c_start] = 0
                else:
                    try:
                        start1 = np.where(wavelengths == s00)[0][0].item()
                        end1 = np.where(wavelengths == s11)[0][0].item()
                        middle = np.where(wavelengths == input_min1000)[0][0].item()

                        input_SR1 = SR[start1:middle + 1]
                        input_SR2 = SR[middle + 1:end1]
                        input_CCA = removed_cube.data[:, a, b]

                        sum4 = sum((1 - input_CCA[c]) * SR[c] for c in range(start1, middle + 1))
                        sum5 = sum((1 - input_CCA[d]) * SR[d] for d in range(middle + 1, end1))

                        total = sum4 + sum5
                        if sum4 == 0 or total == 0:
                            block_result[a - r_start, b - c_start] = 0
                        elif sum4 > sum5:
                            block_result[a - r_start, b - c_start] = -((sum4 - sum5) * 100 / total)
                        else:
                            block_result[a - r_start, b - c_start] = ((sum5 - sum4) * 100 / total)
                    except:
                        block_result[a - r_start, b - c_start] = 0
        return block_result

    # Create spatial blocks
    blocks = []
    for r in range(0, y, block_size):
        for c in range(0, z, block_size):
            r_end = min(r + block_size, y)
            c_end = min(c + block_size, z)
            blocks.append((r, r_end, c, c_end))

    # Parallel execution of blocks
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    # Assemble results
    ASY_values = np.zeros((y, z), dtype=np.float32)
    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        ASY_values[r_start:r_end, c_start:c_end] = results[idx]
        idx += 1

    # Replace 0s with NaN (optional)
    ASY_values[ASY_values == 0] = np.nan
    ASY.data = ASY_values

    return ASY


def RGB5(gauss_cube, wavelengths, shoulder0, shoulder1, min1000, min2000):
    '''Creates the RGB5 index. R: ASY, G: BCI, B: BCII.
    
    Inputs:
    gauss_cube = the filtered cube.
    wavelengths = the wavelengths,
    shoulder0 = the left shoulder of the band, 
    shoulder1 = the right shoulder of the band, 
    min1000 = the minimum at 1 um,
    min2000 = the minimum at 2 um.
    
    Output:
    The RGB5 RGB composite.'''
    
    # Compute ASY directly using the existing ASY() function
    ASY_img = ASY(gauss_cube, wavelengths, shoulder0, shoulder1, min1000)
    
    # Create copy of first 3 bands (shape/metadata preserved)
    RGB5 = gauss_cube[0:3, :, :].copy()
    
    # Combine the ASY, min1000 and min2000 values into an RGB composite
    RGB5.data = np.dstack((ASY_img.data, min1000.data, min2000.data)).transpose(2, 0, 1)
    
    # Optional: replace zeroes with NaNs
    RGB5.data[RGB5.data == 0] = np.nan

    return RGB5


def IBDII(removed_cube):
    '''Calculates the integrated band depth around the 2 um band. Internal process. 
    
    Input:
    removed_cube = the continuum-removed cube.
    
    Output:
    Integrated band depth at 2 um.'''
    
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
    Integrated band depth at 1 um.'''
    
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


def RGB8(gauss_cube, removed_cube, block_size=50, n_jobs=None):
    '''Creates the RGB8 index. R: BD1900, G: IBD2000, B: IBD1000.
    
    Inputs:
    gauss_cube = the filtered cube.
    removed_cube = continuum-removed cube.
    block_size = size of processing block for parallelization.
    n_jobs = number of parallel jobs (cores).
    
    Output:
    The RGB8 RGB composite.'''

    y, z = removed_cube[0, :, :].shape

    # Band 1: Band depth at 1900 nm
    RGB81 = 1 - removed_cube[55, :, :]
    RGB81.data[RGB81.data == 1] = np.nan

    # Prepare containers for bands 2 and 3
    RGB82 = removed_cube[0, :, :].copy()
    RGB83 = removed_cube[0, :, :].copy()

    # Slices for bands 2 and 3
    RGB82_slice = removed_cube[49:70, :, :]
    RGB83_slice = removed_cube[8:34, :, :]

    # Resolve n_jobs
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_block_82(r_start, r_end, c_start, c_end):
        block_result = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                if RGB82_slice[19, a, b] == 0:
                    block_result[a - r_start, b - c_start] = 0
                else:
                    input_removed = RGB82_slice.data[:, a, b]
                    block_result[a - r_start, b - c_start] = np.sum(1 - input_removed)
        return block_result

    def process_block_83(r_start, r_end, c_start, c_end):
        block_result = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                if RGB83_slice[19, a, b] == 0:
                    block_result[a - r_start, b - c_start] = 0
                else:
                    input_removed4 = RGB83_slice.data[:, a, b]
                    block_result[a - r_start, b - c_start] = np.sum(1 - input_removed4)
        return block_result

    blocks = []
    for r in range(0, y, block_size):
        for c in range(0, z, block_size):
            r_end = min(r + block_size, y)
            c_end = min(c + block_size, z)
            blocks.append((r, r_end, c, c_end))

    results_82 = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block_82)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )
    results_83 = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block_83)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    full_82 = np.zeros((y, z), dtype=np.float32)
    full_83 = np.zeros((y, z), dtype=np.float32)

    for idx, (r_start, r_end, c_start, c_end) in enumerate(blocks):
        full_82[r_start:r_end, c_start:c_end] = results_82[idx]
        full_83[r_start:r_end, c_start:c_end] = results_83[idx]

    RGB82.data = full_82
    RGB83.data = full_83

    # Compose RGB cube
    RGB8_total = gauss_cube[0:3, :, :].copy()
    RGB8_total.data = np.dstack((RGB81, RGB82, RGB83)).transpose(2, 0, 1)
    RGB8_total.data[RGB8_total.data == 0] = np.nan

    return RGB8_total