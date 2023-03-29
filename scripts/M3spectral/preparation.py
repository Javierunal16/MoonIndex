import numpy as np
import pysptools.spectro as spectro

#This function attach the wavelength to the cube
def attach_wavelen (cube_alone,wave):
    cube_alone.coords['wavelength'] = ('band', wave)
    cube_wave = cube_alone.swap_dims({'band':'wavelength'})
    return cube_wave

#This function makes the convex hull
def convex_hull (fourier_cube,wavelengths):
    hull_cube=fourier_cube.copy()
    stack_hull=[]  
    x3,y3,z3=fourier_cube.shape
    for a in range(fourier_cube.data.shape[1]):
        for b in range(fourier_cube.data.shape[2]):  #This iterates for every pixel
            imput_imgh=fourier_cube.data[:,a,b]
            RasterHull=spectro.convex_hull_removal(imput_imgh,wavelengths)  #Convex hull
            RasterHull1=np.array(RasterHull[0])
            stack_hull.append(RasterHull1)
        
    stack_hulla=np.array(stack_hull)
    hull_cube.data=stack_hulla.reshape(y3,z3,x3).transpose(2,0,1)
    return hull_cube

#Finding the minimums
def find_minimums (hull_cube,wavelengths):
    min1000=hull_cube[0,:,:].copy()  #Saving the filtered data in a new cube, copied from the original to maintain the projection
    stack_min1000=[]
    min2000=hull_cube[0,:,:].copy()
    stack_min2000=[]
    y4,z4=hull_cube[0,:,:].shape
    
    for a in range(hull_cube.data.shape[1]):
        for b in range(hull_cube.data.shape[2]):
        
            imput_hull_1000=hull_cube.data[:,a,b]
            
            minimum_1000=np.where(imput_hull_1000[0:29] == min(imput_hull_1000[0:29]))[0]  #Finds the minimum value of the reflectance in wavelengths, the limtis were set after a manual iteration  
            value_1000=wavelengths[minimum_1000]
            stack_min1000.append(value_1000)
            
            minimum_2000=np.where(imput_hull_1000[35:75] == min(imput_hull_1000[35:75]))[0]+35
            value_2000=wavelengths[minimum_2000]
            stack_min2000.append(value_2000)
            
    stack_min1000a=np.array(stack_min1000)
    min1000.data=stack_min1000a.reshape(y4,z4)

    stack_min2000a=np.array(stack_min2000)
    min2000.data=stack_min2000a.reshape(y4,z4)
    return (min1000,min2000)

#Finding the shoulders, highest poitns beetwen the minimums
def find_shoulders (hull_cube2, wavelengths):
    shoulder0=hull_cube2[0,:,:].copy()
    stack_shoulder0=[]
    shoulder1=hull_cube2[0,:,:].copy()
    stack_shoulder1=[]
    shoulder2=hull_cube2[0,:,:].copy()
    stack_shoulder2=[]
    shoulder3=hull_cube2[0,:,:].copy()
    stack_shoulder3=[]
    y5,z5=hull_cube2[0,:,:].shape
    
    for a in range(hull_cube2.data.shape[1]):
        for b in range(hull_cube2.data.shape[2]):
        
            imput_hull_shoulder=hull_cube2.data[:,a,b]
        
            shoulder_0=np.where(imput_hull_shoulder[0:20] == max(imput_hull_shoulder[0:20]))[0][-1]  # Works similar to the minimums, but the last argument ensures than only the last value is returned
            value_0=wavelengths[shoulder_0]
            stack_shoulder0.append(value_0)
        
            shoulder_1=np.where(imput_hull_shoulder[20:40] == max(imput_hull_shoulder[20:40]))[0][-1]+20
            value_1=wavelengths[shoulder_1]
            stack_shoulder1.append(value_1)
        
            shoulder_2=np.where(imput_hull_shoulder[40:66] == max(imput_hull_shoulder[40:66]))[0][-1]+40
            value_2=wavelengths[shoulder_2]
            stack_shoulder2.append(value_2)
        
            shoulder_3=np.where(imput_hull_shoulder[66:76] == max(imput_hull_shoulder[66:76]))[0][-1]+66
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

#Continumm fits
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