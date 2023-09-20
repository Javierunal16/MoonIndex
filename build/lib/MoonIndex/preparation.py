import numpy as np
import pysptools.spectro as spectro
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy as sp


#This function attach the wavelength to the cube, sets the no value data to 0 and deletes the two first malfunctioning bandsdef attach_wavelength
def attach_wavelen (cube_alone,wave):
    cube_alone2=cube_alone[2:85,:,:]
    cube_alone2.data[cube_alone2.data < -1000]= 0
    cube_alone2.data[cube_alone2.data > 1000]= 0
    cube_alone2.coords['wavelength'] = ('band', wave)
    cube_wave = cube_alone2.swap_dims({'band':'wavelength'})
    return cube_wave


#This function crop the cube using the array indexes
def crop_cube_size (initial_cube,crx1,crx2,cry1,cry2):
    cx1,cx2=crx1,crx2
    cy1,cy2=cry1,cry2
    M3_cubecrop=initial_cube[:,cy1:cy2,cx1:cx2]
    rect_crop=patches.Rectangle((cx1,cy1),(cx2-cx1),(cy2-cy1),linewidth=1, edgecolor='r', facecolor='none')

    plot0, ax=plt.subplots(1,2, figsize=(5,20))
    ax[0].imshow(initial_cube[0,:,:])
    ax[0].set_title('Full cube')
    ax[0].add_patch(rect_crop)
    ax[1].imshow(M3_cubecrop[0,:,:])
    ax[1].set_title('Cropped cube')
    return M3_cubecrop

#This function crop the cube using the coordinates
def crop_cube (initial_cube,minnx,minny,maxxx,maxxy):
    croped_cube=initial_cube.rio.clip_box(minx=minnx,miny=minny,maxx=maxxx,maxy=maxxy)
    rect_crop=patches.Rectangle((minnx,minny),(maxxx-minnx),(maxxy-minny),linewidth=1, edgecolor='r', facecolor='none')

    plot0, (axs, axs1)=plt.subplots(1,2, figsize=(5,20))

    initial_cube[0:3,:,:].plot.imshow(ax=axs,robust=True,add_labels=False)
    axs.add_patch(rect_crop)
    axs.set_aspect(1)
    axs.set_title('Full cube')

    croped_cube[0:3,:,:].plot.imshow(ax=axs1,robust=True,add_labels=False)
    axs1.set_aspect(1)
    axs1.set_title('Cropped cube')
    plt.tight_layout() 
    return croped_cube

##CONVEX HULL METHOD
#This function makes the convex hull
def convexhull_removal(fourier_cube, wavelengths_full,mid_point):
    
    hull_cube=fourier_cube[0:74,:,:].copy()  #Copying the raster and cropping
    wavelengths=wavelengths_full[0:74]
    stack_hull=[]  
    x_hull,y_hull,z_hull=hull_cube.shape
    
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            input_fourier=fourier_cube.data[0:74,a,b]
            input_midpoint=mid_point.data[a,b]
            
            if input_fourier[39] == 0: 
                stack_hull.append(np.zeros(74))
            else:
            
                add_point=np.where(wavelengths==input_midpoint)[0]  #Adding the midpoint
                add_array2=np.vstack((wavelengths[add_point],  input_fourier[add_point])).T
    
                points = np.c_[wavelengths, input_fourier]  #Doing the convexhull
                wavelengths, input_fourier = points.T
                augmented = np.concatenate([points, [(wavelengths[0], np.min(input_fourier)-1), (wavelengths[-1], np.min(input_fourier)-1)]], axis=0)
                hull = sp.spatial.ConvexHull(augmented)
                pre_continuum_points2 = points[np.sort([v for v in hull.vertices if v < len(points)])]
                pre_continuum_points2 = np.concatenate((pre_continuum_points2,add_array2), axis=0)
                pre_continuum_points2.sort(axis=0)
                continuum_points2=np.unique(pre_continuum_points2,axis=0)
                continuum_function2 = sp.interpolate.interp1d(*continuum_points2.T)
                fourier_cube_prime = input_fourier / continuum_function2(wavelengths)
                fourier_cube_prime[fourier_cube_prime >= 1]= 1
                fourier_cube_prime=np.nan_to_num(fourier_cube_prime)
            
                stack_hull.append(fourier_cube_prime)
            
    stack_hulla=np.array(stack_hull)
    stack_hullb=stack_hulla.astype(np.float32)
    hull_cube.data=stack_hullb.reshape(y_hull,z_hull,x_hull).transpose(2,0,1)
    
    return hull_cube

# Making rasters with the wavelength of minimum reflectance in 1000 um and 2000 um respectively 
def find_minimums_ch (hull_cube,midpoint,wavelengths2):
    min1000=hull_cube[0,:,:].copy()  #Saving the filtered data in a new cube, copied from the original to maintain the projection
    stack_min1000=[]
    min2000=hull_cube[0,:,:].copy()
    stack_min2000=[]
    ymin1000,zmin1000=hull_cube[0,:,:].shape
    ymin2000,zmin2000=hull_cube[0,:,:].shape
    wavelengths=wavelengths2[0:74]
    
    
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
        
            input_hull=hull_cube.data[:,a,b]
            
            if input_hull[39] == 0:
                
                stack_min1000.append(0)
                stack_min2000.append(0)
                
            else:
                input_midpoint=midpoint.data[a,b]
                pre_midpointp=np.where(wavelengths==input_midpoint)[0]
                midpointp=int(pre_midpointp)
                minimum_1000=np.argmin(input_hull[0:midpointp])  #Finds the minimum value of the reflectance in wavelengths, the limtis is defined by the midpoint  
                ofset=5
                fitxp=minimum_1000-ofset  #This creates a window around the minimum in the convex hull to do a posteriro fit
                fitxp=max(0,fitxp)  #If the value is minor to 0, it converts it to 0
                fitxp2=np.array(minimum_1000+ofset) 
                fitxp2[fitxp2 > midpointp]= midpointp-1  #The -1 is to avoid that the resulting array is at the midpoint
                fitx=wavelengths[int(fitxp):int(fitxp2)]
                fity=input_hull[int(fitxp):int(fitxp2)]
                fit_1000=np.polyfit(fitx,fity,2)  #Creates a second order fit aroud the minimum
                polyval_1000=np.polyval(fit_1000,wavelengths[int(fitxp):int(fitxp2)])
                min1000p=np.where(polyval_1000== min(polyval_1000))[0]  #Finds the minimum in the fit, this reduce the noise of the final data
                final_1000=wavelengths[min1000p+fitxp]
                stack_min1000.append(final_1000[0])
            
                minimum_2000=np.argmin(input_hull[midpointp:74])+midpointp  #This one does not need specific defintion of indexes vecause there is no risk of ocerlap with the midpoint
                fit_2000=np.polyfit(wavelengths[int(minimum_2000-ofset):int(minimum_2000+ofset)],input_hull[int(minimum_2000-ofset):int(minimum_2000+ofset)],2)
                polyval_2000=np.polyval(fit_2000,wavelengths[int(minimum_2000-ofset):int(minimum_2000+ofset+1)])
                min2000p=np.where(polyval_2000== min(polyval_2000))[0]
                final_2000=wavelengths[min2000p+minimum_2000-ofset]
                stack_min2000.append(final_2000[0])
    
    
    stack_min1000a=np.array(stack_min1000)
    stack_min1000a[stack_min1000a ==  wavelengths[0]]= wavelengths[18]
    stack_min1000b=stack_min1000a.astype(np.float32)
    min1000.data=stack_min1000b.reshape(ymin1000,zmin1000)

    stack_min2000a=np.array(stack_min2000)
    stack_min2000b=stack_min2000a.astype(np.float32)
    min2000.data=stack_min2000b.reshape(ymin2000,zmin2000)
    
    return (min1000,min2000)


#Obtaining the shoulders, point of maximum reflectance beetween the minimum
def find_shoulders_ch (hull_cube2,midpoint,min_1000,min_2000, wavelengths3):
    shoulder0=hull_cube2[0,:,:].copy()
    stack_shoulder0=[]
    shoulder1=hull_cube2[0,:,:].copy()
    stack_shoulder1=[]
    shoulder2=hull_cube2[0,:,:].copy()
    stack_shoulder2=[]
    shoulder3=hull_cube2[0,:,:].copy()
    stack_shoulder3=[]
    y5,z5=hull_cube2[0,:,:].shape
    wavelengths=wavelengths3[0:74]

    for a in range(hull_cube2.data.shape[1]):
        for b in range(hull_cube2.data.shape[2]):

            input_hull_shoulder=hull_cube2.data[:,a,b]
            
            if input_hull_shoulder[39] == 0:
                
                stack_shoulder0.append(0)
                stack_shoulder1.append(0)
                stack_shoulder2.append(0)
                stack_shoulder3.append(0)
            else:
                input_midpoint_shoulder=midpoint.data[a,b]
                pre_midpointp=np.where(wavelengths==input_midpoint_shoulder)[0]
                midpoint_shoulderp=int(pre_midpointp)
                input_min1000=min_1000.data[a,b]
                pre_input_min1000p=np.where(wavelengths==input_min1000)[0]
                min1000p=int(pre_input_min1000p)
                input_min2000=min_2000.data[a,b]
                pre_input_min2000p=np.where(wavelengths==input_min2000)[0]
                min2000p=int(pre_input_min2000p)
            
                shoulder_0=np.where(input_hull_shoulder[0:min1000p] == max(input_hull_shoulder[0:min1000p]))[0][-1]  # Works similar to the maximums, but the last argument ensures than only the last value is returned
                ofset=3
                fitxp0=shoulder_0-ofset  #This creates a window around the maximum in the convex hull to do a posterior fit
                fitxp0=max(0, fitxp0) #If the value is minor to 0, it converts it to 0
                fitx0=wavelengths[int(fitxp0):int(shoulder_0+ofset)]
                fity0=input_hull_shoulder[int(fitxp0):int(shoulder_0+ofset)]
                fit_0=np.polyfit(fitx0,fity0,2)  #Creates a second order fit aroud the maxima
                polyval_0=np.polyval(fit_0,wavelengths[int(fitxp0):int(shoulder_0+ofset+1)])
                max0p=np.where(polyval_0== max(polyval_0))[0]  #FInds the minimum in the fit, this reduce the noise of the final data
                final_0=wavelengths[max0p+fitxp0]
                stack_shoulder0.append(final_0[0])

                shoulder_1=np.where(input_hull_shoulder[min1000p:midpoint_shoulderp] == max(input_hull_shoulder[min1000p:midpoint_shoulderp]))[0][-1]+min1000p
                fitxp1=shoulder_1-ofset
                fitxp1=max(0,fitxp1)
                fit_1=np.polyfit(wavelengths[int(fitxp1):int(shoulder_1+ofset)],input_hull_shoulder[int(fitxp1):int(shoulder_1+ofset)],2)
                polyval_1=np.polyval(fit_1,wavelengths[int(fitxp1):int(shoulder_1+ofset+1)])
                max1p=np.where(polyval_1== max(polyval_1))[0]
                final_1=wavelengths[max1p+fitxp1]
                stack_shoulder1.append(final_1[0])

                if midpoint_shoulderp-min2000p < 0:  #To avoid errors where the aborsoption feature is weak, if the value is too low it assing the midpoint
                
                    shoulder_2=np.where(input_hull_shoulder[midpoint_shoulderp:min2000p] == max(input_hull_shoulder[midpoint_shoulderp:min2000p]))[0][-1]+midpoint_shoulderp
                    value_2=wavelengths[shoulder_2]
                    stack_shoulder2.append(value_2)
                
                else:
                
                    stack_shoulder2.append(wavelengths[midpoint_shoulderp])

                shoulder_3=np.where(input_hull_shoulder[min2000p:74] == max(input_hull_shoulder[min2000p:74]))[0][-1]+min2000p
                value_3=wavelengths[shoulder_3]
                stack_shoulder3.append(value_3)

    stack_shoulder0a=np.array(stack_shoulder0)
    stack_shoulder0b=stack_shoulder0a.astype(np.float32)
    shoulder0.data=stack_shoulder0b.reshape(y5,z5)

    stack_shoulder1a=np.array( stack_shoulder1)
    stack_shoulder1b=stack_shoulder1a.astype(np.float32)
    shoulder1.data=stack_shoulder1b.reshape(y5,z5)

    stack_shoulder2a=np.array(stack_shoulder2)
    stack_shoulder2b=stack_shoulder2a.astype(np.float32)
    shoulder2.data=stack_shoulder2b.reshape(y5,z5)

    stack_shoulder3a=np.array(stack_shoulder3)
    stack_shoulder3b=stack_shoulder3a.astype(np.float32)
    shoulder3.data=stack_shoulder3b.reshape(y5,z5)
    
    return (shoulder0, shoulder1, shoulder2, shoulder3)

#Function to find the midpoint to add to the convexhull
def midpoint(fourier_cube,wavelengths,peak_distance,peak_prominence):
    midpoint_cube=fourier_cube[0,:,:].copy()
    midpoint_stack=[]
    x_midpoint,y_midpoint,z_midpoint=fourier_cube.shape
    
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):
            
            cube_fourier=fourier_cube[20:60,a,b]
            if cube_fourier[39] == 0: 
                midpoint_stack.append(0)
            else:
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
    midpoint_stackb=midpoint_stacka.astype(np.float32)
    midpoint_cube.data=midpoint_stackb.reshape(y_midpoint,z_midpoint)
    return (midpoint_cube)


#LINEAR FIT METHOD

#Continuum removal with the linear fit method
def continuum_removal_lf (gauss_cube,wavelengths2):
    lf=gauss_cube[0:74,:,:].copy()
    stack_lf=[]
    x,y,z=lf[:,:,:].shape
    wavelengths=wavelengths2[0:74]
    
    for a in range(gauss_cube.data.shape[1]):
        for b in range(gauss_cube.data.shape[2]):
    
            lf_cube=gauss_cube.data[0:74,a,b]  #Second order fit for 1000 nm, it used a range for the two shoudlers around the 1000 nm absorption
            
            if lf_cube[39] == 0: 
                stack_lf.append(np.zeros(74))
            else:
                
                fitx10001=wavelengths[1:7]
                fitx10002=wavelengths[39:42]
                fitx1000=np.hstack((fitx10001,fitx10002))
                fity10001=lf_cube[1:7]
                fity10002=lf_cube[39:42]
                fity1000=np.hstack((fity10001,fity10002))
                fit1000=np.polyfit(fitx1000,fity1000,2)
                polival1000=np.polyval(fit1000,wavelengths[0:42])

                fitx2000=np.hstack((fitx10002,wavelengths[73])) #Fit for 2000 nm, linear
                fity2000=np.hstack((fity10002,lf_cube[73]))
                fit2000=np.polyfit(fitx2000,fity2000,1)
                polival2000=np.polyval(fit2000,wavelengths[42:74])

                continuum=np.hstack((polival1000,polival2000))  #Continuum removal by dividing
                continuum_removed=lf_cube/continuum
                continuum_removed[continuum_removed > 1]= 1
                stack_lf.append(continuum_removed)
            
    stack_lfa=np.array(stack_lf)
    lf.data=stack_lfa.reshape(y,z,x).transpose(2,0,1)
    
    return(lf)

#Finding the minimuums with the linear fit method
def find_minimuumslf (lf_cube,wavelengths):
    min_1000lf=lf_cube[0,:,:].copy()  #Saving the filtered data in a new cube, copied from the original to maintain the projection
    stack_min_1000lf=[]
    min_2000lf=lf_cube[0,:,:].copy()
    stack_min_2000lf=[]
    y,z=lf_cube[0,:,:].shape

    for a in range(lf_cube.data.shape[1]):
        for b in range(lf_cube.data.shape[2]):
        
            min_lf=lf_cube.data[:,a,b]
            
            if min_lf[39] == 0:
                
                stack_min_1000lf.append(0)
                stack_min_2000lf.append(0)
                
            else:
                minimum_1000lf=np.argmin(min_lf[7:39])+7  #Finds the minimum value of the reflectance in wavelengths, the limtis is defined by the midpoint  
                ofsetlf=5
                fitxplf=minimum_1000lf-ofsetlf  #This creates a window around the minimum in the convex hull to do a posteriro fit
                fitxp2lf=np.array(minimum_1000lf+ofsetlf)
                fitxp2lf[fitxp2lf > 39]= 38
                fitxlf=wavelengths[int(fitxplf):int(fitxp2lf)]
                fitylf=min_lf[int(fitxplf):int(fitxp2lf)]
                fit_1000lf=np.polyfit(fitxlf,fitylf,2)  #Creates a second order fit aroud the minimum
                polyval_1000lf=np.polyval(fit_1000lf,wavelengths[int(fitxplf):int(fitxp2lf)])
                min1000plf=np.argmin(polyval_1000lf)  #Finds the minimum in the fit, this reduce the noise of the final data
                final_1000lf=wavelengths[min1000plf+fitxplf]
                stack_min_1000lf.append(final_1000lf)

                minimum_2000lf=np.argmin(min_lf[39:74])+39
                min2000=minimum_2000lf+ofsetlf
                if min2000 > 73: min2000=73
                fit_2000lf=np.polyfit(wavelengths[int(minimum_2000lf-ofsetlf):int(min2000)],min_lf[int(minimum_2000lf-ofsetlf):int(min2000)],2)
                polyval_2000lf=np.polyval(fit_2000lf,wavelengths[int(minimum_2000lf-ofsetlf):int(minimum_2000lf+ofsetlf+1)])
                min2000plf=np.argmin(polyval_2000lf)
                wave_index2000=min2000plf+minimum_2000lf-ofsetlf
                if wave_index2000 > 73: wave_index2000=73
                if wave_index2000 < 39: wave_index2000=39
                final_2000lf=wavelengths[wave_index2000]
                stack_min_2000lf.append(final_2000lf)
    
    
    stack_min1000lfa=np.array(stack_min_1000lf)
    stack_min1000lfa[stack_min1000lfa ==  wavelengths[0]]= wavelengths[18]
    min_1000lf.data=stack_min1000lfa.reshape(y,z)

    stack_min2000lfa=np.array(stack_min_2000lf)
    min_2000lf.data=stack_min2000lfa.reshape(y,z)
    
    return (min_1000lf,min_2000lf)

#Finding the shoulders with the lienar fit method

def find_shoulders_lf (lf_cube,min_1000lf,min_2000lf, wavelengths):
    shoulder0lf=lf_cube[0,:,:].copy()
    stack_shoulder0lf=[]
    shoulder1lf=lf_cube[0,:,:].copy()
    stack_shoulder1lf=[]
    shoulder2lf=lf_cube[0,:,:].copy()
    stack_shoulder2lf=[]
    y,z=lf_cube[0,:,:].shape

    for a in range(lf_cube.data.shape[1]):
        for b in range(lf_cube.data.shape[2]):

            input_shoulderlf=lf_cube.data[:,a,b]
            if input_shoulderlf[39] == 0:
                
                stack_shoulder0lf.append(0)
                stack_shoulder1lf.append(0)
                stack_shoulder2lf.append(0)
                
            else:
                input_min1000lf=min_1000lf.data[a,b]
                pre_input_min1000plf=np.where(wavelengths==input_min1000lf)[0]
                min1000plf=int(pre_input_min1000plf)
                input_min2000lf=min_2000lf.data[a,b]
                pre_input_min2000plf=np.where(wavelengths==input_min2000lf)[0]
                min2000plf=int(pre_input_min2000plf)

                shoulder_0lf=np.where(input_shoulderlf[0:min1000plf] == max(input_shoulderlf[0:min1000plf]))[0][-1]  # but the last argument ensures than only the last value is returned
                ofsetlf=3
                fitxp0lf=shoulder_0lf-ofsetlf  #This creates a window around the maximum in the convex hull to do a posterior fit
                fitxp0lf=max(0, fitxp0lf) #If the value is minor to 0, it converts it to 0
                fitx0lf=wavelengths[int(fitxp0lf):int(shoulder_0lf+ofsetlf)]
                fity0lf=input_shoulderlf[int(fitxp0lf):int(shoulder_0lf+ofsetlf)]
                fit_0lf=np.polyfit(fitx0lf,fity0lf,2)  #Creates a second order fit aroud the maxima
                polyval_0lf=np.polyval(fit_0lf,wavelengths[int(fitxp0lf):int(shoulder_0lf+ofsetlf+1)])
                max0plf=np.where(polyval_0lf== max(polyval_0lf))[0]  #Finds the maximuum in the fit, this reduce the noise of the final data
                final_0lf=wavelengths[max0plf+fitxp0lf]
                stack_shoulder0lf.append(final_0lf[0])

                shoulder_1lf=np.where(input_shoulderlf[min1000plf:min2000plf+1] == max(input_shoulderlf[min1000plf:min2000plf+1]))[0][-1]+min1000plf
                maxs1=shoulder_1lf+ofsetlf
                if maxs1 > 74: maxs1=74
                fitxp1lf=shoulder_1lf-ofsetlf
                fitxp1lf=max(0,fitxp1lf)
                fit_1lf=np.polyfit(wavelengths[int(fitxp1lf):int(maxs1)],input_shoulderlf[int(fitxp1lf):int(maxs1)],2)
                polyval_1lf=np.polyval(fit_1lf,wavelengths[int(fitxp1lf):int(shoulder_1lf+ofsetlf+1)])
                max1plf=np.where(polyval_1lf== max(polyval_1lf))[0]
                final_1lf=wavelengths[max1plf+fitxp1lf]
                stack_shoulder1lf.append(final_1lf[0])

                value_2=wavelengths[74]
                stack_shoulder2lf.append(value_2)

    stack_shoulder0lfa=np.array(stack_shoulder0lf)
    shoulder0lf.data=stack_shoulder0lfa.reshape(y,z)

    stack_shoulder1lfa=np.array( stack_shoulder1lf)
    shoulder1lf.data=stack_shoulder1lfa.reshape(y,z)

    stack_shoulder2lfa=np.array(stack_shoulder2lf)
    shoulder2lf.data=stack_shoulder2lfa.reshape(y,z)

    return (shoulder0lf, shoulder1lf, shoulder2lf)

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