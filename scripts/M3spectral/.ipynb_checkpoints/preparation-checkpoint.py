import numpy as np
import pysptools.spectro as spectro
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy as sp


#This function attach the wavelength to the cube
def attach_wavelen (cube_alone,wave):
    cube_alone2=cube_alone[2:85,:,:]
    cube_alone2.data[cube_alone2.data < -1000]= 0
    cube_alone2.data[cube_alone2.data > 1000]= 0
    cube_alone2.coords['wavelength'] = ('band', wave)
    cube_wave = cube_alone2.swap_dims({'band':'wavelength'})
    return cube_wave


#This function attach the wavelength to the cube, sets the no value data to 0 and deletes the two first malfunctioning bands
def crop_cube (initial_cube,crx1,crx2,cry1,cry2):
    cx1,cx2=crx1,crx2
    cy1,cy2=cry1,cry2
    M3_cubecrop=initial_cube[:,cy1:cy2,cx1:cx2]
    rect_crop=patches.Rectangle((cx1,cy1),(cx2-cx1),(cy2-cy1),linewidth=1, edgecolor='r', facecolor='none')

    plot0, ax=plt.subplots(1,2, figsize=(5,20))
    ax[0].imshow(initial_cube[5,:,:])
    ax[0].set_title('Full cube')
    ax[0].add_patch(rect_crop)
    ax[1].imshow(M3_cubecrop[5,:,:])
    ax[1].set_title('Cropped cube')
    return M3_cubecrop


#This function makes the convex hull
def convexhull_removal(fourier_cube, wavelengths_full,mid_point):
    
    hull_cube=fourier_cube[0:76,:,:].copy()  #Copying the raster and cropping
    wavelengths=wavelengths_full[0:76]
    stack_hull=[]  
    x_hull,y_hull,z_hull=hull_cube.shape
    
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            imput_fourier=fourier_cube.data[0:76,a,b]
            imput_midpoint=mid_point.data[a,b]
            
            add_point=np.where(wavelengths==imput_midpoint)[0]  #Adding the midpoint
            add_array2=np.vstack((wavelengths[add_point],  imput_fourier[add_point])).T
    
            points = np.c_[wavelengths, imput_fourier]  #Doing the convexhull
            wavelengths, imput_fourier = points.T
            augmented = np.concatenate([points, [(wavelengths[0], np.min(imput_fourier)-1), (wavelengths[-1], np.min(imput_fourier)-1)]], axis=0)
            hull = sp.spatial.ConvexHull(augmented)
            pre_continuum_points2 = points[np.sort([v for v in hull.vertices if v < len(points)])]
            pre_continuum_points2 = np.concatenate((pre_continuum_points2,add_array2), axis=0)
            pre_continuum_points2.sort(axis=0)
            continuum_points2=np.unique(pre_continuum_points2,axis=0)
            continuum_function2 = sp.interpolate.interp1d(*continuum_points2.T)
            fourier_cube_prime = imput_fourier / continuum_function2(wavelengths)
            fourier_cube_prime[fourier_cube_prime >= 1]= 1
            
            stack_hull.append(fourier_cube_prime)
            
    stack_hulla=np.array(stack_hull)
    hull_cube.data=stack_hulla.reshape(y_hull,z_hull,x_hull).transpose(2,0,1)
    
    return hull_cube


# Making rasters with the wavelength of minimum reflectance in 1000 um and 2000 um respectively
def find_minimums (hull_cube,midpoint,wavelengths2):
    min1000=hull_cube[0,:,:].copy()  #Saving the filtered data in a new cube, copied from the original to maintain the projection
    stack_min1000=[]
    min2000=hull_cube[0,:,:].copy()
    stack_min2000=[]
    ymin1000,zmin1000=hull_cube[0,:,:].shape
    ymin2000,zmin2000=hull_cube[0,:,:].shape
    wavelengths=wavelengths2[0:76]
    
    #Min for 1000 nm
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
        
            imput_hull=hull_cube.data[:,a,b]
            imput_midpoint=midpoint.data[a,b]
            pre_midpointp=np.where(wavelengths==imput_midpoint)[0]
            midpointp=int(pre_midpointp)
            
            minimum_1000=np.where(imput_hull[0:midpointp] == min(imput_hull[0:midpointp]))[0]  #Finds the minimum value of the reflectance in wavelengths, the limtis is defined by the midpoint  
            value_1000=wavelengths[minimum_1000]
            stack_min1000.append(value_1000)
            
            minimum_2000=np.where(imput_hull[midpointp:76] == min(imput_hull[midpointp:76]))[0]+midpointp
            value_2000=wavelengths[minimum_2000]
            stack_min2000.append(value_2000)
            
    stack_min1000a=np.array(stack_min1000)
    min1000.data=stack_min1000a.reshape(ymin1000,zmin1000)

    stack_min2000a=np.array(stack_min2000)
    min2000.data=stack_min2000a.reshape(ymin2000,zmin2000)
    return (min1000,min2000)


#Obtaining the shoulders, point of maximum reflectance beetween the minimum
def find_shoulders (hull_cube2,midpoint,min_1000,min_2000, wavelengths3):
    shoulder0=hull_cube2[0,:,:].copy()
    stack_shoulder0=[]
    shoulder1=hull_cube2[0,:,:].copy()
    stack_shoulder1=[]
    shoulder2=hull_cube2[0,:,:].copy()
    stack_shoulder2=[]
    shoulder3=hull_cube2[0,:,:].copy()
    stack_shoulder3=[]
    y5,z5=hull_cube2[0,:,:].shape
    wavelengths=wavelengths3[0:76]

    for a in range(hull_cube2.data.shape[1]):
        for b in range(hull_cube2.data.shape[2]):

            imput_hull_shoulder=hull_cube2.data[:,a,b]
            imput_midpoint_shoulder=midpoint.data[a,b]
            pre_midpointp=np.where(wavelengths==imput_midpoint_shoulder)[0]
            midpoint_shoulderp=int(pre_midpointp)
            imput_min1000=min_1000.data[a,b]
            pre_imput_min1000p=np.where(wavelengths==imput_min1000)[0]
            min1000p=int(pre_imput_min1000p)
            imput_min2000=min_2000.data[a,b]
            pre_imput_min2000p=np.where(wavelengths==imput_min2000)[0]
            min2000p=int(pre_imput_min2000p)
            

            shoulder_0=np.where(imput_hull_shoulder[0:min1000p] == max(imput_hull_shoulder[0:min1000p]))[0][-1]  # Works similar to the minimums, but the last argument ensures than only the last value is returned
            value_0=wavelengths[shoulder_0]
            stack_shoulder0.append(value_0)

            shoulder_1=np.where(imput_hull_shoulder[min1000p:midpoint_shoulderp] == max(imput_hull_shoulder[min1000p:midpoint_shoulderp]))[0][-1]+min1000p
            value_1=wavelengths[shoulder_1]
            stack_shoulder1.append(value_1)

            if midpoint_shoulderp-min2000p > 0:
                
                shoulder_2=np.where(imput_hull_shoulder[midpoint_shoulderp:min2000p] == max(imput_hull_shoulder[midpoint_shoulderp:min2000p]))[0][-1]+midpoint_shoulderp
                value_2=wavelengths[shoulder_2]
                stack_shoulder2.append(value_2)
                
            else:
                
                stack_shoulder2.append(wavelengths[midpoint_shoulderp])

            shoulder_3=np.where(imput_hull_shoulder[min2000p:76] == max(imput_hull_shoulder[min2000p:76]))[0][-1]+min2000p
            value_3=wavelengths[shoulder_3]
            stack_shoulder3.append(value_3)

    stack_shoulder0a=np.array(stack_shoulder0)
    shoulder0.data=stack_shoulder0a.reshape(y5,z5)

    stack_shoulder1a=np.array( stack_shoulder1)
    shoulder1.data=stack_shoulder1a.reshape(y5,z5)

    stack_shoulder2a=np.array(stack_shoulder2)
    shoulder2.data=stack_shoulder2a.reshape(y5,z5)

    stack_shoulder3a=np.array(stack_shoulder3)
    shoulder3.data=stack_shoulder3a.reshape(y5,z5)
    return (shoulder0, shoulder1, shoulder2, shoulder3)


#Function to find the midpoint to add to the convexhull
def midpoint(fourier_cube,wavelengths,peak_distance,peak_prominence):
    midpoint_cube=fourier_cube[0,:,:].copy()
    midpoint_stack=[]
    x_midpoint,y_midpoint,z_midpoint=fourier_cube.shape
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
        
            cube_fourier=fourier_cube[20:60,a,b]
            midpoint_y=np.array(cube_fourier[0])
            midpoint_y2=np.append(midpoint_y,cube_fourier[39])
            midpoint_x=np.array(wavelengths[20])
            midpoint_x2=np.append(midpoint_x,wavelengths[59])
            midpoint_fit=np.polyfit(midpoint_x2,midpoint_y2,1)
            midpoint_polival=np.polyval(midpoint_fit,wavelengths[20:60])  #Creating a linear fit in the desired range
        
            dif_cube=cube_fourier-midpoint_polival  #Diferrence beetwen the data and the linear fit
            peaks, _ =sp.signal.find_peaks(dif_cube,distance=peak_distance, prominence=peak_prominence)  #We use peak detection in the resulting curve to chose the value
        
            if len(peaks) == 0:
                midpoint_stack.append(wavelengths[42])  #If no peak is detected we select the arbitrary position of 1469.07 nm, obtained from literature (few percentage of pixels need this)
            else:
                midpoint_stack.append(wavelengths[peaks+20][-1])

    midpoint_stacka=np.array(midpoint_stack)
    midpoint_cube.data=midpoint_stacka.reshape(y_midpoint,z_midpoint)
    return (midpoint_cube)


#Continumm fit 1000
def continnum_1000 (filtered_cube,hull_cube,wavelengths,x_continum,y_continum):
    
            shoulder_0p=np.where(hull_cube[0:20,y_continum,x_continum] == max(hull_cube[0:20,y_continum,x_continum]))[0][-1]  #Finding the shoulders, as they will limit the fit
            shoulder_0y=filtered_cube[shoulder_0p,y_continum,x_continum]
            shoulder_0x=wavelengths[shoulder_0p]
            shoulder_1p=np.where(hull_cube[20:40,y_continum,x_continum] == max(hull_cube[20:40,y_continum,x_continum]))[0][-1]+20
            shoulder_1y=filtered_cube[shoulder_1p,y_continum,x_continum]
            shoulder_1x=wavelengths[shoulder_1p]
            interpolation_x=np.array(shoulder_0x)
            interpolation_x1=np.append(interpolation_x,shoulder_1x)  #Creating X and Y values to do the fit
            interpolation_y=np.array(shoulder_0y)
            interpolation_y1=np.append(interpolation_y,shoulder_1y)
            fit_1000=np.polyfit(interpolation_x1,interpolation_y1,1)  #Doing the fit, it is a linear fit
            return fit_1000


#Continumm fit 2000       
def continnum_2000 (filtered_cube,hull_cube,wavelengths,x_continum,y_continum):
            shoulder_2p=np.where(hull_cube[40:66,y_continum,x_continum] == max(hull_cube[40:66,y_continum,x_continum]))[0][-1]+40  #Finding the shoulders, as they will limit the fit
            shoulder_2y=filtered_cube[shoulder_2p,y_continum,x_continum]
            shoulder_2x=wavelengths[shoulder_2p]
            shoulder_3p=np.where(hull_cube[66:76,y_continum,x_continum] == max(hull_cube[66:76,y_continum,x_continum]))[0][-1]+66
            shoulder_3y=filtered_cube[shoulder_3p,y_continum,x_continum]
            shoulder_3x=wavelengths[shoulder_3p]
            interpolation_x2=np.array(shoulder_2x)
            interpolation_x3=np.append(interpolation_x2,shoulder_3x)  #Creating X and Y values to do the fit
            interpolation_y2=np.array(shoulder_2y)
            interpolation_y3=np.append(interpolation_y2,shoulder_3y)
            fit_2000=np.polyfit(interpolation_x3,interpolation_y3,1)  #Doing the fit, it is a linear fit
            return fit_2000