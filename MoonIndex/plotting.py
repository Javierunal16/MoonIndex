import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler


def cube_plot(cube_plot,size,title):
    '''Plots a cube or RGB composite and normalizes the values to 0-255. 
    Inputs:
    cube_plot = cube,
    size = size of the plot,
    title = title.
    
    Outputs:
    The plot of the cube.'''
    #Copying the cube
    cube_plot2=cube_plot[:,:,:].copy()
    cube_plot2.data=np.nan_to_num(cube_plot.data)
    x,y,z=cube_plot[:,:,:].shape
    #Scale values to 0-255
    scaled=[]
    scaler=MinMaxScaler(feature_range=(0, 255))
    for band in range(cube_plot.data.shape[0]):
    
        scaledRGB=scaler.fit_transform(cube_plot2.data[band,:,:])
        scaled.append(scaledRGB)
            
    scaleda=np.array(scaled)
    cube_plot.data=scaleda.reshape(x,y,z)
    #Plotting the cube
    plot=cube_plot.plot.imshow(aspect=cube_plot.shape[2]/cube_plot.shape[1], size=size,add_labels=False, robust=True)
    plt.title(title)
    return (plot)


def image_plot(image_input,size2,title):
    '''Plots a single band image with an "Spectral" colormap. 
    
    Inputs:
    image_input = single band image,
    size = the size fo the plot, 
    title = title.
    
    Outputs:
    Plot of the image.'''
    
    plot2=image_input.plot.imshow(aspect=image_input.shape[1]/image_input.shape[0], size=size2, robust=True,add_labels=False, cmap='Spectral')
    plt.title(title)
    return (plot2)


def plot_comparison (cube_plot1, cube_plot2, band1,band2, title1, title2):
    '''Plots two selected band of a cube or cubes, to compare between them. 
    
    Inputs: 
    cube_plot1 = the first cube, 
    cube_plot2 = the second cube,
    band1 = the band of the first cube,
    band2 = the band of the second cube,
    title1 = the title of the first cube,
    title2 = the title of the second cube.
    
    Outputs:
    The comparing plot of both bands.'''
    
    plot1, ax=plt.subplots(1,2)
    ax[0].imshow(cube_plot1[band1,:,:])
    ax[0].set_title(title1)
    ax[1].imshow(cube_plot2[band2,:,:])
    ax[1].set_title(title2)
    return(plt.show(plot1))


def fourier_plot (initial_cube,band,percentage_width, percentage_height):
    '''Plot the steps of the Fourier filtering to check the results. This function is only for viewing, filter the cube use the homonimous function under filtration. 
    
    Inputs: 
    initial_cube = the cube, 
    band = the band to check, 
    percentage_width = the width of the filter in percentange,
    percentage_height = and height of the filter in percentage.
    
    Otput:
    Image to check the Fourier fitlering.'''
    
    fouraster2=cv2.dft(initial_cube.data[band,:,:], flags=cv2.DFT_COMPLEX_OUTPUT) 
    fouraster2_shift=np.fft.fftshift(fouraster2)
    #For plotting
    magnitude_spectrum = 20 * np.log((cv2.magnitude(fouraster2_shift[:, :, 0], fouraster2_shift[:, :, 1])))  
    y,z=initial_cube.data[band,:,:].shape
    
    rows, cols =initial_cube[band,:,:].shape
    mask = np.ones((rows, cols, 2), np.uint8)
    
    filter_width2=int((z*((100-percentage_width)/100))/2)
    filter_height2=int((y*(percentage_height/100))/2)

    #Mask
    cv2.rectangle(mask, (0,((y//2)-filter_height2)), (((z//2)-filter_width2),((y//2)+filter_height2)), 0, -1)
    cv2.rectangle(mask, (((z//2)+filter_width2),((y//2)-filter_height2)), (z,((y//2)+filter_height2)), 0, -1)

    #Mask applciation and reverse-fourier
    mfouraster2=fouraster2_shift*mask
    mshift_mask_mag = 20 * np.log(cv2.magnitude(mfouraster2[:, :, 0], mfouraster2[:, :, 1]))  #For plotting
    m_ishift2 = np.fft.ifftshift(mfouraster2)


    return_raster2=cv2.idft(m_ishift2)/(y*z)
    return_raster2= cv2.magnitude(return_raster2[:, :, 0], return_raster2[:, :, 1])
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(initial_cube.data[band,:,:], cmap="gray")
    ax1.title.set_text('Original Data')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum, cmap="gray")
    ax2.title.set_text('Fourier of image')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(mshift_mask_mag, cmap="gray")
    ax3.title.set_text('Fourier + Mask')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(return_raster2.data, cmap="gray")
    ax4.title.set_text('After inverse Fourier')
    

def profiles_comparison(wavelengths,first_cube, second_cube,title1,title2, in_x, in_y,roi):
    '''Plot two spectral signatures to compare, the spectra is averaged in a window around the selected pixel. 
    Inputs:
    wavelengths =  the wavlengths, 
    first_cube = the first cube,
    second_cube = the second cube, 
    title1 = the title of the first cube,
    title2 = the title of the second cube, 
    in_x = the x position of the pixel to plot, 
    in_y = the y position of the pixel to plot
    roi = the size of the window in number of pixels (3 recommended).
    
    Outputs:
    The profiles comparison.'''
    
    #Defining the window
    stack_averaw=[]  
    roi_plus=int((roi/2)+0.5)  
    roi_minus=int((roi/2)-0.5)
    for band in range(first_cube.data.shape[0]):
        
        raw_cube2=first_cube[band,:,:]
        #Slicing the data to the ROI
        neighbourhood = raw_cube2[in_y-roi_minus:in_y+roi_plus, in_x-roi_minus:in_x+roi_plus] 
        #Average of the pixels
        average=np.mean(neighbourhood) 
        
        stack_averaw.append(average)
    first_average=np.array(stack_averaw)
    
    stack_avegauss=[]
    #Same for the filtered profile
    for band in range(second_cube.data.shape[0]):  
        
        gauss_cube2=second_cube[band,:,:]  
        neighbourhood2 = gauss_cube2[in_y-roi_minus:in_y+roi_plus, in_x-roi_minus:in_x+roi_plus]
        average2=np.mean(neighbourhood2)
        
        stack_avegauss.append(average2)
    second_average=np.array(stack_avegauss)
    
    plt.plot(wavelengths[0:len(first_average)], first_average, label=title1)
    plt.plot(wavelengths[0:len(second_average)],second_average, label=title2)
    plt.xlabel("Wavelengths (um)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.title("Profile comparison")
    return plt


def profile_plot (wavelengths,profile_singlecube,title_singleprofile, pixelx,pixely,roi):
    '''Plot the spectral signature of a cube, the spectra is averaged in a window around the selected pixel. 
    
    Inputs:
    wavelengths =  the wavlengths, 
    profile_singlecube = the first cube,
    title_singleprofile = the title of the first cube, 
    pixelx = the x position of the pixel to plot, 
    pixely = the y position of the pixel to plot
    roi = the size of the window in number of pixels (3 recommended).
    
    Outputs:
    The spectral profile.'''
    stack_ave=[]  
    roi_plus=int((roi/2)+0.5)  #Definign the window
    roi_minus=int((roi/2)-0.5)
    
    for band in range(profile_singlecube.data.shape[0]):
        
        neighbourhood = profile_singlecube[pixelx-roi_minus:pixelx+roi_plus, pixely-roi_minus:pixely+roi_plus]  #Slicing the data to the ROI
        average=np.mean(neighbourhood)  #Average of the pixels
        
        stack_ave.append(average)
    average2=np.array(stack_ave)
    
    fig=plt.plot(wavelenghts, average2)
    plt.title(title_singleprofile)
    return plt.show(fig)


def filter_comparison (cube_1,cube_2,title1,title2,band):
    '''Plots a comparison between the cubes before and after the filtration. It also plots the ratio between the cubes, and an iamge showing the pixels that changed more than 2% in black. 
    
    Inputs: 
    cube1 =  the cube before, 
    cube2 = the cube after,
    title1 = the title of te first one, 
    title2 = the title of the second one, 
    band = the band to ceck.
    
    Outputs:
    Plots of to check the effect of the filtering.'''

    #Ratio of the two images
    ratio_cubes=cube_1/cube_2  
    #Calculating which pixels changed more than 2%
    cube_plus=cube_1+(cube_1*0.02)  
    cube_minus=cube_1-(cube_1*0.02)
    ratio_plus=cube_plus-cube_2
    ratio_minus=cube_2-cube_minus
    change_ratio=ratio_plus*ratio_minus
    change_ratio.data[change_ratio.data < 0]= 0
    #Plotting
    fig4, axs = plt.subplots(ncols=2,nrows=2, figsize=(10,10))  
    cube_1[band,:,:].plot.imshow(ax=axs[0,0],add_labels=False)
    axs[0,0].title.set_text(title1)
    axs[0,0].set_aspect(1)
    cube_2[band,:,:].plot.imshow(ax=axs[0,1],add_labels=False)
    axs[0,1].title.set_text(title2)
    axs[0,1].set_aspect(1)
    ratio_cubes[band,:,:].plot.imshow(ax=axs[1,0],robust=True,add_labels=False)
    axs[1,0].title.set_text('Ratio')
    axs[1,0].set_aspect(1)
    change_ratio[band,:,:].plot.imshow(ax=axs[1,1],robust=True,add_labels=False)
    axs[1,1].title.set_text('Change over 2%')
    axs[1,1].set_aspect(1)
    return 


def convexhull_plot(filtered_cube, wavelengths_full,mid_point,y_hull,x_hull):
    '''Plots the reuslt of the convex hull continuum-removal method for a pixel. This function is only for viewing, to change the removal use the homonimous function under Preparation. 
    Inptus:
    filtered_cube = filtered cube,
    wavelengths_full = the wavelengths, 
    mid_point = the tie-point cube, 
    y_hull = the y position of the pixel, 
    x_hull = the x position of the pixel.
    
    Outputs:
    The plot of the continuum-removed spectrum (CH).'''
    
    wavelengths=wavelengths_full[0:76]
    average=filtered_cube[0:76,x_hull,y_hull]
    
    add_point=np.where(wavelengths==mid_point[x_hull,y_hull])[0]
    add_array=np.vstack((wavelengths[add_point], filtered_cube[add_point,x_hull,y_hull])).T
    #Temporarly removes the egdes of the spectrum
    points = np.c_[wavelengths, average]
    wavelengths, average = points.T
    augmented = np.concatenate([points, [(wavelengths[0], np.min(average)-1), (wavelengths[-1], np.min(average)-1)]], axis=0)
    #Performs the convex hull
    hull = sp.spatial.ConvexHull(augmented)
    pre_continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
    pre_continuum_points2 = np.concatenate((pre_continuum_points,add_array), axis=0)
    pre_continuum_points2.sort(axis=0)
    continuum_points=np.unique(pre_continuum_points2,axis=0)
    continuum_function = sp.interpolate.interp1d(*continuum_points.T)
    #Removes the continuum
    average_prime = average / continuum_function(wavelengths)
    average_prime[average_prime >= 1]= 1
    #Plot the image
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(wavelengths, average, label='Data')
    axes[0].plot(*continuum_points.T, label='Continuum')
    axes[0].legend()
    axes[1].plot(wavelengths, average_prime, label='Data / Continuum')
    plt.xlabel("Wavelengths (um)")
    plt.ylabel("Reflectance")
    axes[1].legend()
    return plt


#second-and-first-order fit fit method
def safofit_plot(filtered_cube, removed_cube, wavelengths,y_plot,x_plot):    
    '''Plots the reuslt of the second-and-first-order fit continuum-removal method for a pixel. The limits for the fits are manually defined using values established in the literature. This function is only for viewing, to perform the removal use the homonimous function under Preparation. 
    
    Inptus:
    filtered_cube = the filtered cube,
    removed_cube = the continuum-removed cube,
    y_plot = the y position of the pixel,
    x_plot = the x position of the pixel.
    
    Outputs:
    Plot of the continuum-removed spectrum (SAFO).'''

    #Second order fit for 1000 nm, it used a range for the two shoudlers around the 1000 nm absorption
    SAFO_cube=filtered_cube.data[0:74,x_plot,y_plot]  
    fitx10001=wavelengths[1:7]
    fitx10002=wavelengths[39:42]
    fitx1000=np.hstack((fitx10001,fitx10002))
    fity10001=SAFO_cube[1:7]
    fity10002=SAFO_cube[39:42]
    fity1000=np.hstack((fity10001,fity10002))
    fit1000=np.polyfit(fitx1000,fity1000,2)
    polival1000=np.polyval(fit1000,wavelengths[0:42])
    #Fit for 2000 nm, linear
    fitx2000=np.hstack((fitx10002,wavelengths[73])) 
    fity2000=np.hstack((fity10002,SAFO_cube[73]))
    fit2000=np.polyfit(fitx2000,fity2000,1)
    polival2000=np.polyval(fit2000,wavelengths[42:74])
    #Continuum removal by dividing
    continuum=np.hstack((polival1000,polival2000))  
    
    continum_removed=removed_cube[:,x_plot,y_plot]
    
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(wavelengths[0:74], SAFO_cube, label='Data')
    axes[0].plot(wavelengths[0:74],continuum, label='Continuum')
    axes[0].legend()
    axes[1].plot(wavelengths[0:74], continum_removed[0:74], label='Data / Continuum')
    plt.xlabel("Wavelengths (um)")
    plt.ylabel("Reflectance")
    axes[1].legend()
    return plt