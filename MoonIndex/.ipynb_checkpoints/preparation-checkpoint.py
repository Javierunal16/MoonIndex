import numpy as np
import pysptools.spectro as spectro
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy as sp
from joblib import Parallel, delayed
import multiprocessing


###DATA PRETARION

def attach_wave (initial_cube,wavelengths):
    '''This function eliminates the first two empty bands, turn all anomalous values to no data, and attach the wavelengths. 
    
    Inputs:
    initial_cube = M3 cube, 
    wavelengths = wavelengths file.
    
    Outputs:
    Prepared cube.'''

    #Removing bands
    initial_cube2=initial_cube[2:85,:,:]
    #Removing nodata
    initial_cube2.data[initial_cube2.data < -1000]= 0
    initial_cube2.data[initial_cube2.data > 1000]= 0
    #Attaching wavelength
    initial_cube2.coords['wavelength'] = ('band', wavelengths)
    cube_wave = initial_cube2.swap_dims({'band':'wavelength'})
    return cube_wave


def crop_cube (initial_cube,minnx,minny,maxxx,maxxy):
    '''Crop the prepared cube to a desired location, using the units of the coordinate system. Be caurefull to select values inside the coordinates of the image. 
    
    Inputs: 
    initial_cube = prepared cube, 
    minnx = lowest x limiting value,
    minny = lowest y limiting value,
    maxxx = highest x limiting value, 
    maxxy = highest y limiting value.
    
    Outputs:
    Cropped cube.'''

    #Check input coordinates
    if minnx < initial_cube.coords['x'].min().data or minny < initial_cube.coords['y'].min().data or maxxx > initial_cube.coords['x'].max().data or maxxy > initial_cube.coords['y'].max().data:
        raise ValueError("Invalid coordinate")
    #Assigning coordinates
    croped_cube=initial_cube.rio.clip_box(minx=minnx,miny=minny,maxx=maxxx,maxy=maxxy)
    #Creating the small rectangle
    rect_crop=patches.Rectangle((minnx,minny),(maxxx-minnx),(maxxy-minny),linewidth=1, edgecolor='r', facecolor='none')
    plot0, (axs, axs1)=plt.subplots(1,2, figsize=(5,20))
    #Plotting the cube
    initial_cube[0:3,:,:].plot.imshow(ax=axs,robust=True,add_labels=False)
    axs.add_patch(rect_crop)
    axs.set_aspect(1)
    axs.set_title('Full cube')
    #Plotting the cropped cube
    croped_cube[0:3,:,:].plot.imshow(ax=axs1,robust=True,add_labels=False)
    axs1.set_aspect(1)
    axs1.set_title('Cropped cube')
    plt.tight_layout() 
    return croped_cube

def crop_cube_size (initial_cube,cx1,cy1,cx2,cy2):
    '''Crop the prepared cube to a desired location, usign the number of lines and columns of the file. Be carefull to select values inside the image size. 
    
    Inputs: 
    initial_cube = prepared cube, 
    cx1 = lowest x limiting value,
    cy1 = highest x limiting value,
    cx2 = lowest y limiting value,
    cy2 = highest y limiting value.
    
    Outputs:
    Cropped cube.'''
    #Check input coordinates
    if cx1 < 0 or cy1 < 0 or cx2 > initial_cube.shape[2] or cy2 > initial_cube.shape[1]:
        raise ValueError("Invalid coordinate")
    M3_cubecrop=initial_cube[:,cy1:cy2,cx1:cx2]
    rect_crop=patches.Rectangle((cx1,cy1),(cx2-cx1),(cy2-cy1),linewidth=1, edgecolor='r', facecolor='none')

    plot0, ax=plt.subplots(1,2, figsize=(5,20))
    #To plot using the dimensions of the cube instead of the coordinates we should use only one band, in this case the 540 nm band
    ax[0].imshow(initial_cube[5,:,:], cmap='gray')
    ax[0].set_title('Full cube')
    ax[0].add_patch(rect_crop)
    ax[1].imshow(M3_cubecrop[5,:,:], cmap='gray')
    ax[1].set_title('Cropped cube')
    return M3_cubecrop

###CONTINUUM-REMOVAL

import multiprocessing
from joblib import Parallel, delayed

def midpoint(filtered_cube, wavelengths, peak_distance, peak_prominence, block_size=100, n_jobs=None):
    '''Finds the tie-point to limit the two absorption bands, used when the slope of the spectra is too steep. 
    It uses an automatic function to detect local peaks.
    
    Inputs:
    filtered_cube = filtered cube,
    wavelengths = wavelengths,
    peak_distance = the minimum distance between peaks (6 is recommended),
    peak_prominence = the minimum prominence of the peaks (0.002 is recommended).
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).
    
    Outputs:
    Tie-points cube.'''

    x_midpoint, y_midpoint, z_midpoint = filtered_cube.shape
    midpoint_cube = filtered_cube[0, :, :].copy()

    # Resolve number of jobs
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        cube_filtered = filtered_cube[20:60, a, b]
        if cube_filtered[39] == 0:
            return 0
        else:
            midpoint_y = np.array(cube_filtered[0])
            midpoint_y2 = np.append(midpoint_y, cube_filtered[39])
            midpoint_x = np.array(wavelengths[20])
            midpoint_x2 = np.append(midpoint_x, wavelengths[59])
            midpoint_fit = np.polyfit(midpoint_x2, midpoint_y2, 1)
            midpoint_polival = np.polyval(midpoint_fit, wavelengths[20:60])
            dif_cube = cube_filtered - midpoint_polival
            peaks, _ = sp.signal.find_peaks(dif_cube, distance=peak_distance, prominence=peak_prominence)
            if len(peaks) == 0:
                return wavelengths[42]
            else:
                return wavelengths[peaks + 20][-1]

    def process_block(r_start, r_end, c_start, c_end):
        block_result = np.zeros((r_end - r_start, c_end - c_start), dtype=np.float32)
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                block_result[a - r_start, b - c_start] = process_pixel(a, b)
        return block_result

    blocks = []
    for r in range(0, y_midpoint, block_size):
        for c in range(0, z_midpoint, block_size):
            r_end = min(r + block_size, y_midpoint)
            c_end = min(c + block_size, z_midpoint)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    full_array = np.zeros((y_midpoint, z_midpoint), dtype=np.float32)
    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        full_array[r_start:r_end, c_start:c_end] = results[idx]
        idx += 1

    midpoint_cube.data = full_array
    return midpoint_cube

##Convex hull method
 
def convexhull_removal(filtered_cube, wavelengths_full, mid_point, block_size=100, n_jobs=None):
    '''Remove the continuum of the spectra using the convex-hull method. 
    
    Inputs:
    filtered_cube = filtered cube, 
    wavelengths_full = wavelengths,
    mid_point = tie-point cube.
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).
    
    Outputs:
    Continuum removed cube by convex hull(CH).'''
    
    hull_cube = filtered_cube[:,:,:].copy()
    wavelengths = wavelengths_full
    bands, rows, cols = hull_cube.shape

    # Resolve number of jobs
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        input_filtered = filtered_cube.data[:, a, b]
        input_midpoint = mid_point.data[a, b]

        if input_filtered[39] == 0:
            return np.zeros(bands, dtype=np.float32)

        try:
            add_point = np.where(wavelengths == input_midpoint)[0]
            add_array2 = np.vstack((wavelengths[add_point], input_filtered[add_point])).T
            points = np.c_[wavelengths, input_filtered]
            augmented = np.concatenate([
                points, 
                [(wavelengths[0], np.min(input_filtered) - 1), 
                 (wavelengths[-1], np.min(input_filtered) - 1)]
            ])
            hull = sp.spatial.ConvexHull(augmented)
            hull_vertices = [v for v in hull.vertices if v < len(points)]
            pre_continuum_points2 = points[np.sort(hull_vertices)]
            pre_continuum_points2 = np.concatenate((pre_continuum_points2, add_array2), axis=0)
            pre_continuum_points2 = np.unique(pre_continuum_points2, axis=0)
            pre_continuum_points2.sort(axis=0)
            continuum_function = sp.interpolate.interp1d(
                pre_continuum_points2[:, 0],
                pre_continuum_points2[:, 1],
                bounds_error=False,
                fill_value="extrapolate"
            )
            continuum = continuum_function(wavelengths)
            result = input_filtered / continuum
            result[result >= 1] = 1
            return np.nan_to_num(result).astype(np.float32)
        except:
            return np.zeros(bands, dtype=np.float32)

    def process_block(r_start, r_end, c_start, c_end):
        block_result = []
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                block_result.append(process_pixel(a, b))
        block_array = np.array(block_result, dtype=np.float32).reshape(r_end - r_start, c_end - c_start, bands)
        return block_array

    blocks = []
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    full_array = np.zeros((bands, rows, cols), dtype=np.float32)
    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        block_array = results[idx]
        idx += 1
        full_array[:, r_start:r_end, c_start:c_end] = block_array.transpose(2, 0, 1)

    hull_cube.data = full_array
    return hull_cube

def find_minimums_ch(hull_cube, midpoint, wavelengths, block_size=100, n_jobs=None):
    '''This function finds the minimums around the 1 um and 2 um bands for the convex hull method. 
    
    Inputs:
    hull_cube = continuum-removed cube (CH),
    midpoint = tie-point,
    wavelengths = wavelengths.
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).
    
    Outputs:
    Minimum at 1 um and minimum at 2 um cubes.'''

    min1000 = hull_cube[0, :, :].copy()
    min2000 = hull_cube[0, :, :].copy()
    ymin, zmin = hull_cube[0, :, :].shape
    bands = hull_cube.shape[0]
    ofset = 5

    # Configurar núcleos para paralelización
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        # Default output: zeros
        val_min1000 = 0
        val_min2000 = 0
        input_hull = hull_cube.data[:, a, b]

        if input_hull[39] != 0:
            try:
                input_midpoint = midpoint.data[a, b]
                midpointp = int(np.where(wavelengths == input_midpoint)[0][0])

                # Minimum around 1 um band
                minimum_1000 = np.argmin(input_hull[0:midpointp])
                fitxp = max(0, minimum_1000 - ofset)
                fitxp2 = minimum_1000 + ofset
                if fitxp2 > midpointp:
                    fitxp2 = midpointp - 1
                fitx = wavelengths[fitxp:fitxp2]
                fity = input_hull[fitxp:fitxp2]

                fit_1000 = np.polyfit(fitx, fity, 2)
                polyval_1000 = np.polyval(fit_1000, wavelengths[fitxp:fitxp2])
                min1000p = np.where(polyval_1000 == np.min(polyval_1000))[0][0]
                final_1000 = wavelengths[min1000p + fitxp]

                if input_hull[min1000p + fitxp] < 0.974:
                    val_min1000 = final_1000
                else:
                    val_min1000 = 0

                # Minimum around 2 um band
                minimum_2000 = np.argmin(input_hull[midpointp:74]) + midpointp
                fit_start = max(0, minimum_2000 - ofset)
                fit_end = min(bands, minimum_2000 + ofset + 1)
                fitx2 = wavelengths[fit_start:fit_end]
                fity2 = input_hull[fit_start:fit_end]

                fit_2000 = np.polyfit(fitx2, fity2, 2)
                polyval_2000 = np.polyval(fit_2000, wavelengths[fit_start:fit_end])
                min2000p = np.where(polyval_2000 == np.min(polyval_2000))[0][0]

                final_index = min(max(min2000p + fit_start, midpointp + 1), 73)
                final_2000 = wavelengths[final_index]

                if input_hull[min2000p + fit_start] < 0.983:
                    val_min2000 = final_2000
                else:
                    val_min2000 = 0
            except:
                pass

        return val_min1000, val_min2000

    def process_block(r_start, r_end, c_start, c_end):
        min1000_block = []
        min2000_block = []
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                m1000, m2000 = process_pixel(a, b)
                min1000_block.append(m1000)
                min2000_block.append(m2000)
        shape = (r_end - r_start, c_end - c_start)
        return (np.array(min1000_block, dtype=np.float32).reshape(shape),
                np.array(min2000_block, dtype=np.float32).reshape(shape))

    blocks = []
    for r in range(0, ymin, block_size):
        for c in range(0, zmin, block_size):
            r_end = min(r + block_size, ymin)
            c_end = min(c + block_size, zmin)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    arr_min1000 = np.zeros((ymin, zmin), dtype=np.float32)
    arr_min2000 = np.zeros((ymin, zmin), dtype=np.float32)

    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        min1000_block, min2000_block = results[idx]
        arr_min1000[r_start:r_end, c_start:c_end] = min1000_block
        arr_min2000[r_start:r_end, c_start:c_end] = min2000_block
        idx += 1

    # Corrección de valores iguales a wavelengths[0]
    arr_min1000[arr_min1000 == wavelengths[0]] = wavelengths[18]

    min1000.data = arr_min1000
    min2000.data = arr_min2000

    return min1000, min2000 


def find_shoulders_ch(hull_cube, midpoint, min_1000, min_2000, wavelengths3, block_size=100, n_jobs=None):
    '''Find the shoulders around the minimums at 1 um and 2 um for the convex hull method. 
    
    Inputs:
    hull_cube = continuum removed cube (CH),
    midpoint = tie-point,
    min_1000 = the minimum at 1 um cube, 
    min_2000 = the minimum at 2 um cube,
    wavelengths3 = wavelengths.
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).
    
    Outputs:
    Left and right shoulders of the 1 um absorption band, left and right shoulder of the 2 um absorption band.'''

    # Crear cubos copia para los resultados
    shoulder0 = hull_cube[0, :, :].copy()
    shoulder1 = hull_cube[0, :, :].copy()
    shoulder2 = hull_cube[0, :, :].copy()
    shoulder3 = hull_cube[0, :, :].copy()
    y5, z5 = hull_cube[0, :, :].shape
    wavelengths = wavelengths3
    bands = hull_cube.shape[0]

    # Determinar núcleos para paralelización
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        result = [0, 0, 0, 0]  # [shoulder0, shoulder1, shoulder2, shoulder3]
        input_hull = hull_cube.data[:, a, b]

        try:
            input_mid = midpoint.data[a, b]
            midpoint_p = np.where(wavelengths == input_mid)[0][0]
        except:
            return result

        if min_1000[a, b] != 0:
            try:
                input_min1000 = min_1000.data[a, b]
                min1000_p = np.where(wavelengths == input_min1000)[0][0]
                ofset = 3

                # Left shoulder 1 μm
                shoulder_0 = np.where(input_hull[0:min1000_p] == max(input_hull[0:min1000_p]))[0][-1]
                fitxp0 = max(0, shoulder_0 - ofset)
                fitx0 = wavelengths[fitxp0:shoulder_0 + ofset]
                fity0 = input_hull[fitxp0:shoulder_0 + ofset]
                fit_0 = np.polyfit(fitx0, fity0, 2)
                polyval_0 = np.polyval(fit_0, wavelengths[fitxp0:shoulder_0 + ofset + 1])
                max0p = np.where(polyval_0 == max(polyval_0))[0][0]
                result[0] = wavelengths[max0p + fitxp0]

                # Right shoulder 1 μm
                shoulder_1 = np.where(input_hull[min1000_p:midpoint_p] == max(input_hull[min1000_p:midpoint_p]))[0][-1] + min1000_p
                fitxp1 = max(0, shoulder_1 - ofset)
                fitx1 = wavelengths[fitxp1:shoulder_1 + ofset]
                fity1 = input_hull[fitxp1:shoulder_1 + ofset]
                fit_1 = np.polyfit(fitx1, fity1, 2)
                polyval_1 = np.polyval(fit_1, wavelengths[fitxp1:shoulder_1 + ofset + 1])
                max1p = np.where(polyval_1 == max(polyval_1))[0][0]
                result[1] = wavelengths[max1p + fitxp1]
            except:
                pass

        if min_2000[a, b] != 0:
            try:
                input_min2000 = min_2000.data[a, b]
                min2000_p = np.where(wavelengths == input_min2000)[0][0]

                # Left shoulder 2 μm
                if midpoint_p - min2000_p < 0:
                    shoulder_2 = np.where(input_hull[midpoint_p:min2000_p] == max(input_hull[midpoint_p:min2000_p]))[0][-1] + midpoint_p
                    result[2] = wavelengths[shoulder_2]
                else:
                    result[2] = wavelengths[midpoint_p]

                # Right shoulder 2 μm
                shoulder_3 = np.where(input_hull[min2000_p:74] == max(input_hull[min2000_p:74]))[0][-1] + min2000_p
                result[3] = wavelengths[shoulder_3]
            except:
                pass

        return result

    def process_block(r_start, r_end, c_start, c_end):
        block_sh0 = []
        block_sh1 = []
        block_sh2 = []
        block_sh3 = []
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                s0, s1, s2, s3 = process_pixel(a, b)
                block_sh0.append(s0)
                block_sh1.append(s1)
                block_sh2.append(s2)
                block_sh3.append(s3)
        shape = (r_end - r_start, c_end - c_start)
        return (
            np.array(block_sh0, dtype=np.float32).reshape(shape),
            np.array(block_sh1, dtype=np.float32).reshape(shape),
            np.array(block_sh2, dtype=np.float32).reshape(shape),
            np.array(block_sh3, dtype=np.float32).reshape(shape)
        )

    # Crear bloques
    blocks = []
    for r in range(0, y5, block_size):
        for c in range(0, z5, block_size):
            r_end = min(r + block_size, y5)
            c_end = min(c + block_size, z5)
            blocks.append((r, r_end, c, c_end))

    # Paralelizar
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    # Reconstruir los cubos
    shoulder0_arr = np.zeros((y5, z5), dtype=np.float32)
    shoulder1_arr = np.zeros((y5, z5), dtype=np.float32)
    shoulder2_arr = np.zeros((y5, z5), dtype=np.float32)
    shoulder3_arr = np.zeros((y5, z5), dtype=np.float32)

    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        sh0, sh1, sh2, sh3 = results[idx]
        shoulder0_arr[r_start:r_end, c_start:c_end] = sh0
        shoulder1_arr[r_start:r_end, c_start:c_end] = sh1
        shoulder2_arr[r_start:r_end, c_start:c_end] = sh2
        shoulder3_arr[r_start:r_end, c_start:c_end] = sh3
        idx += 1

    # Asignar a cubos originales
    shoulder0.data = shoulder0_arr
    shoulder1.data = shoulder1_arr
    shoulder2.data = shoulder2_arr
    shoulder3.data = shoulder3_arr

    return (shoulder0, shoulder1, shoulder2, shoulder3)

##second-and-first-order fit
def continuum_removal_SAFO(filtered_cube, wavelengths, order1, order2, block_size=100, n_jobs=None):
    '''Remove the continuum of the spectra using the second-and-first-order fit method. The limits for the fits are manually defined using values established
    in the literature.
    
    Inputs:
    filtered_cube = fitlered cube, 
    wavelengths = wavelengths,
    order1 = polynomial order for the first absorption band.
    order2 = polynomial order for the second absorption band.
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).   
    
    Outputs:
    Continuum removed cube by second-and-first-order fit (SAFO).'''

    # Copy input cube to preserve original
    SAFO_cube = filtered_cube[:,:,:].copy()
    bands, rows, cols = SAFO_cube.shape

    # Resolve number of jobs
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        # Extract spectrum for pixel (a, b)
        spectrum = filtered_cube.data[:, a, b]

        if spectrum[39] == 0:
            return np.zeros(83, dtype=np.float32)
        
        # Fit continuum at 1000 nm (second order by default)
        fitx1 = wavelengths[1:7]
        fitx2 = wavelengths[39:42]
        fity1 = spectrum[1:7]
        fity2 = spectrum[39:42]
        x_fit_1000 = np.hstack((fitx1, fitx2))
        y_fit_1000 = np.hstack((fity1, fity2))
        poly_1000 = np.polyfit(x_fit_1000, y_fit_1000, order1)
        continuum_1000 = np.polyval(poly_1000, wavelengths[0:42])

        # Fit continuum at 2000 nm (first order by default)
        x_fit_2000 = np.hstack((fitx2, wavelengths[73]))
        y_fit_2000 = np.hstack((fity2, spectrum[73]))
        poly_2000 = np.polyfit(x_fit_2000, y_fit_2000, order2)
        continuum_2000 = np.polyval(poly_2000, wavelengths[42:83])

        # Join the two parts of the continuum and apply the division
        continuum = np.hstack((continuum_1000, continuum_2000))
        result = spectrum / continuum
        result[result > 1] = 1

        return np.nan_to_num(result).astype(np.float32)

    def process_block(r_start, r_end, c_start, c_end):
        block_result = []
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                block_result.append(process_pixel(a, b))
        return np.array(block_result, dtype=np.float32).reshape(r_end - r_start, c_end - c_start, 83)

    # Define spatial blocks
    blocks = []
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            blocks.append((r, r_end, c, c_end))

    # Process blocks in parallel
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    # Reconstruct the cube
    full_array = np.zeros((83, rows, cols), dtype=np.float32)
    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        block = results[idx]
        idx += 1
        full_array[:, r_start:r_end, c_start:c_end] = block.transpose(2, 0, 1)

    SAFO_cube.data = full_array
    return SAFO_cube


def find_minimums_SAFO(SAFO_cube, wavelengths, block_size=100, n_jobs=None):
    '''This function finds the minimums around the 1 um and 2 um bands for the second-and-first-order fit method. 
    
    Inputs:
    SAFO_cube = continuum-removed cube (SAFO),
    wavelengths = wavelengths.
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).
    
    Outputs:
    Minimum at 1 um and minimum at 2 um cubes.'''


    min_1000SAFO = SAFO_cube[0,:,:].copy()
    min_2000SAFO = SAFO_cube[0,:,:].copy()
    y, z = SAFO_cube[0,:,:].shape
    bands = SAFO_cube.shape[0]
    ofsetSAFO = 5

    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        val_min_1000 = 0
        val_min_2000 = 0
        min_SAFO = SAFO_cube.data[:, a, b]

        if min_SAFO[39] != 0:
            try:
                # Minimum 1 um
                minimum_1000SAFO = np.argmin(min_SAFO[7:39]) + 7
                fitxpSAFO = max(0, minimum_1000SAFO - ofsetSAFO)
                fitxp2SAFO = minimum_1000SAFO + ofsetSAFO
                if fitxp2SAFO > 39:
                    fitxp2SAFO = 38
                fitxSAFO = wavelengths[int(fitxpSAFO):int(fitxp2SAFO)]
                fitySAFO = min_SAFO[int(fitxpSAFO):int(fitxp2SAFO)]
                fit_1000SAFO = np.polyfit(fitxSAFO, fitySAFO, 2)
                polyval_1000SAFO = np.polyval(fit_1000SAFO, wavelengths[int(fitxpSAFO):int(fitxp2SAFO)])
                min1000pSAFO = np.argmin(polyval_1000SAFO)
                final_1000SAFO = wavelengths[min1000pSAFO + int(fitxpSAFO)]

                if min_SAFO[min1000pSAFO + int(fitxpSAFO)] < 0.98:
                    val_min_1000 = final_1000SAFO
                else:
                    val_min_1000 = 0

                # Minimum 2 um
                minimum_2000SAFO = np.argmin(min_SAFO[39:74]) + 39
                min2000 = minimum_2000SAFO + ofsetSAFO
                if min2000 > 73:
                    min2000 = 73
                fitx2 = wavelengths[int(minimum_2000SAFO - ofsetSAFO):int(min2000)]
                fity2 = min_SAFO[int(minimum_2000SAFO - ofsetSAFO):int(min2000)]
                fit_2000SAFO = np.polyfit(fitx2, fity2, 2)
                polyval_2000SAFO = np.polyval(fit_2000SAFO, wavelengths[int(minimum_2000SAFO - ofsetSAFO):int(minimum_2000SAFO + ofsetSAFO + 1)])
                min2000pSAFO = np.argmin(polyval_2000SAFO)
                wave_index2000 = min2000pSAFO + minimum_2000SAFO - ofsetSAFO

                wave_index2000 = min(max(wave_index2000, 39), 73)
                final_2000SAFO = wavelengths[wave_index2000]

                if min_SAFO[wave_index2000] < 0.98:
                    val_min_2000 = final_2000SAFO
                else:
                    val_min_2000 = 0
            except:
                pass

        return val_min_1000, val_min_2000

    def process_block(r_start, r_end, c_start, c_end):
        min1000_block = []
        min2000_block = []
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                m1000, m2000 = process_pixel(a, b)
                min1000_block.append(m1000)
                min2000_block.append(m2000)
        shape = (r_end - r_start, c_end - c_start)
        return (np.array(min1000_block, dtype=np.float32).reshape(shape),
                np.array(min2000_block, dtype=np.float32).reshape(shape))

    blocks = []
    for r in range(0, y, block_size):
        for c in range(0, z, block_size):
            r_end = min(r + block_size, y)
            c_end = min(c + block_size, z)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    arr_min_1000SAFO = np.zeros((y, z), dtype=np.float32)
    arr_min_2000SAFO = np.zeros((y, z), dtype=np.float32)

    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        min1000_block, min2000_block = results[idx]
        arr_min_1000SAFO[r_start:r_end, c_start:c_end] = min1000_block
        arr_min_2000SAFO[r_start:r_end, c_start:c_end] = min2000_block
        idx += 1

    # Corrección según el código original
    arr_min1000SAFOa = arr_min_1000SAFO
    arr_min1000SAFOa[arr_min1000SAFOa == wavelengths[0]] = wavelengths[18]

    min_1000SAFO.data = arr_min1000SAFOa
    min_2000SAFO.data = arr_min_2000SAFO

    return min_1000SAFO, min_2000SAFO


def find_shoulders_SAFO(SAFO_cube, min_1000SAFO, min_2000SAFO, wavelengths, block_size=100, n_jobs=None):
    '''Find the shoulders around the minimums at 1 um and 2 um for the second-and-first-order fit method. 
    
    Inputs:
    SAFO_cube = continuum removed cube (SAFO),
    min_1000SAFO = the minimum at 1 um cube, 
    min_2000SAFO = the minimum at 2 um cube,
    wavelengths = wavelengths.
    block_size= Size of the block during the paralelization.
    n_jobs= Caution. Amount of cores for paralelization. Use -1 for all cores, or a negative number to leave some cores 
    free (e.g., -2 means all but 2 cores).If None, defaults to 1 (no parallelism).  
    
    Outputs:
    Left and right shoulders of the 1 um absorption band, left and right shoulder of the 2 um absorption band (the rigth shoulder of the 1 um is the same as the left shoulder of the 2 um absorption band).'''
    
    shoulder0SAFO = SAFO_cube[0, :, :].copy()
    shoulder1SAFO = SAFO_cube[0, :, :].copy()
    shoulder2SAFO = SAFO_cube[0, :, :].copy()
    y, z = SAFO_cube[0, :, :].shape
    bands = SAFO_cube.shape[0]
    ofsetSAFO = 3

    # Configurar núcleos para paralelización
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)

    def process_pixel(a, b):
        s0, s1, s2 = 0, 0, 0
        input_shoulderSAFO = SAFO_cube.data[:, a, b]

        if min_1000SAFO[a, b] != 0:
            try:
                input_min1000SAFO = min_1000SAFO.data[a, b]
                min1000pSAFO = np.where(wavelengths == input_min1000SAFO)[0][0]
                shoulder_0SAFO = np.where(input_shoulderSAFO[0:min1000pSAFO] == max(input_shoulderSAFO[0:min1000pSAFO]))[0][-1]
                fitxp0SAFO = max(0, shoulder_0SAFO - ofsetSAFO)
                fitx0SAFO = wavelengths[fitxp0SAFO:shoulder_0SAFO + ofsetSAFO]
                fity0SAFO = input_shoulderSAFO[fitxp0SAFO:shoulder_0SAFO + ofsetSAFO]
                fit_0SAFO = np.polyfit(fitx0SAFO, fity0SAFO, 2)
                polyval_0SAFO = np.polyval(fit_0SAFO, wavelengths[fitxp0SAFO:shoulder_0SAFO + ofsetSAFO + 1])
                max0pSAFO = np.where(polyval_0SAFO == max(polyval_0SAFO))[0][0]
                s0 = wavelengths[max0pSAFO + fitxp0SAFO]
            except:
                pass

        if min_2000SAFO[a, b] != 0:
            try:
                input_min2000SAFO = min_2000SAFO.data[a, b]
                min2000pSAFO = np.where(wavelengths == input_min2000SAFO)[0][0]
                if min_1000SAFO[a, b] != 0:
                    min1000pSAFO = np.where(wavelengths == min_1000SAFO.data[a, b])[0][0]
                else:
                    min1000pSAFO = 0
                shoulder_1SAFO = np.where(
                    input_shoulderSAFO[min1000pSAFO:min2000pSAFO + 1] == max(input_shoulderSAFO[min1000pSAFO:min2000pSAFO + 1])
                )[0][-1] + min1000pSAFO
                fitxp1SAFO = max(0, shoulder_1SAFO - ofsetSAFO)
                maxs1 = min(shoulder_1SAFO + ofsetSAFO, bands - 1)
                fitx1SAFO = wavelengths[fitxp1SAFO:maxs1]
                fity1SAFO = input_shoulderSAFO[fitxp1SAFO:maxs1]
                fit_1SAFO = np.polyfit(fitx1SAFO, fity1SAFO, 2)
                polyval_1SAFO = np.polyval(fit_1SAFO, wavelengths[fitxp1SAFO:shoulder_1SAFO + ofsetSAFO + 1])
                max1pSAFO = np.where(polyval_1SAFO == max(polyval_1SAFO))[0][0]
                s1 = wavelengths[max1pSAFO + fitxp1SAFO]
                s2 = wavelengths[74]  # último valor
            except:
                pass

        return s0, s1, s2

    def process_block(r_start, r_end, c_start, c_end):
        sh0 = []
        sh1 = []
        sh2 = []
        for a in range(r_start, r_end):
            for b in range(c_start, c_end):
                s0, s1, s2 = process_pixel(a, b)
                sh0.append(s0)
                sh1.append(s1)
                sh2.append(s2)
        shape = (r_end - r_start, c_end - c_start)
        return (
            np.array(sh0, dtype=np.float32).reshape(shape),
            np.array(sh1, dtype=np.float32).reshape(shape),
            np.array(sh2, dtype=np.float32).reshape(shape)
        )

    blocks = []
    for r in range(0, y, block_size):
        for c in range(0, z, block_size):
            r_end = min(r + block_size, y)
            c_end = min(c + block_size, z)
            blocks.append((r, r_end, c, c_end))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_block)(r_start, r_end, c_start, c_end) for (r_start, r_end, c_start, c_end) in blocks
    )

    # Reconstrucción
    arr_sh0 = np.zeros((y, z), dtype=np.float32)
    arr_sh1 = np.zeros((y, z), dtype=np.float32)
    arr_sh2 = np.zeros((y, z), dtype=np.float32)

    idx = 0
    for r_start, r_end, c_start, c_end in blocks:
        sh0, sh1, sh2 = results[idx]
        arr_sh0[r_start:r_end, c_start:c_end] = sh0
        arr_sh1[r_start:r_end, c_start:c_end] = sh1
        arr_sh2[r_start:r_end, c_start:c_end] = sh2
        idx += 1

    shoulder0SAFO.data = arr_sh0
    shoulder1SAFO.data = arr_sh1
    shoulder2SAFO.data = arr_sh2

    return shoulder0SAFO, shoulder1SAFO, shoulder2SAFO
