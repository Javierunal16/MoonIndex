import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy as sp


#Viewing a cube
def cube_plot(cube_plot,band,size,title):
    plot=cube_plot[band,:,:].plot.imshow(aspect=cube_plot.shape[2]/cube_plot.shape[1], size=size, robust=True,add_labels=False)
    plt.title(title)
    return (plot)


#Viewing an image
def image_plot(image_imput,size2,title):
    plot2=image_imput.plot.imshow(aspect=image_imput.shape[1]/image_imput.shape[0], size=size2, robust=True,add_labels=False)
    plt.title(title)
    return (plot2)


#Comparing images
def plot_comparison (cube_plot1, cube_plot2, title1, title2, band):
    plot1, ax=plt.subplots(1,2)
    ax[0].imshow(cube_plot1[band,:,:])
    ax[0].set_title(title1)
    ax[1].imshow(cube_plot2[band,:,:])
    ax[1].set_title(title2)
    return(plt.show(plot1))


#Fourier fitler images, it does the filter for one band to check visually
def fourier_plot (gauss_filter2,fourie_cube2,band,percentage_width, percentage_high):
    fouraster2=cv2.dft(gauss_filter2.data[band,:,:], flags=cv2.DFT_COMPLEX_OUTPUT) 
    fouraster2_shift=np.fft.fftshift(fouraster2)
    magnitude_spectrum = 20 * np.log((cv2.magnitude(fouraster2_shift[:, :, 0], fouraster2_shift[:, :, 1])))  #For plotting
    y3,z3=gauss_filter2.data[band,:,:].shape
    
    rows, cols =gauss_filter2[band,:,:].shape
    mask = np.ones((rows, cols, 2), np.uint8)
    
    filter_width2=int((z3*((100-percentage_width)/100))/2)
    filter_high2=int((y3*(percentage_high/100))/2)

    #Valentine Mask
    cv2.rectangle(mask, (0,((y3//2)-filter_high2)), (((z3//2)-filter_width2),((y3//2)+filter_high2)), 0, -1)
    cv2.rectangle(mask, (((z3//2)+filter_width2),((y3//2)-filter_high2)), (z3,((y3//2)+filter_high2)), 0, -1)

    #Mask applciation and reverse-fourier
    mfouraster2=fouraster2_shift*mask
    mshift_mask_mag = 20 * np.log(cv2.magnitude(mfouraster2[:, :, 0], mfouraster2[:, :, 1]))  #For plotting
    m_ishift2 = np.fft.ifftshift(mfouraster2)


    return_raster2=cv2.idft(m_ishift2)/(y3*z3)
    return_raster2= cv2.magnitude(return_raster2[:, :, 0], return_raster2[:, :, 1])
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(gauss_filter2.data[band,:,:])
    ax1.title.set_text('M3_gauss')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(magnitude_spectrum)
    ax2.title.set_text('Fourier of image')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(mshift_mask_mag)
    ax3.title.set_text('Fourier + Mask')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(fourie_cube2.data[band,:,:])
    ax4.title.set_text('After inverse Fourier')
    

#Profile comparisons
def profiles_comparison(wavelenghts,raw_cube, gauss_cube,tittle1,tittle2, in_x, in_y,roi):

    stack_averaw=[]  
    roi_plus=int((roi/2)+0.5)  #Definign the window
    roi_minus=int((roi/2)-0.5)
    for band in range(raw_cube.data.shape[0]):
        
        raw_cube2=raw_cube[band,:,:]
        neighbourhood = raw_cube2[in_y-roi_minus:in_y+roi_plus, in_x-roi_minus:in_x+roi_plus]  #Slicing the data to the ROI
        average=np.mean(neighbourhood)  #Average of the pixels
        
        stack_averaw.append(average)
    raw_average=np.array(stack_averaw)
    
    stack_avegauss=[]
    for band in range(gauss_cube.data.shape[0]):  #Same for the filtered profile
        
        gauss_cube2=gauss_cube[band,:,:]  
        neighbourhood2 = gauss_cube2[in_y-roi_minus:in_y+roi_plus, in_x-roi_minus:in_x+roi_plus]
        average2=np.mean(neighbourhood2)
        
        stack_avegauss.append(average2)
    gauss_average=np.array(stack_avegauss)
    
    fig3, (ax22, ax23)=plt.subplots(1,2,figsize=(10,3))  #Plotting
    ax22.plot(wavelenghts, raw_average)
    ax22.title.set_text(tittle1)
    ax23.plot(wavelenghts, gauss_average, color='red')
    ax23.title.set_text(tittle2)
    
    return plt.show(fig3)


#Single profile
def profile_plot (wavelengths,profile_singlecube,title_singleprofile, pixelx2,pixely2,roi):
    stack_ave=[]  
    roi_plus=int((roi/2)+0.5)  #Definign the window
    roi_minus=int((roi/2)-0.5)
    for band in range(profile_singlecube.data.shape[0]):
        
        profile_singlecube2=profile_singlecube[band,:,:]
        neighbourhood3 = profile_singlecube2[pixelx2-roi_minus:pixelx2+roi_plus, pixely2-roi_minus:pixely2+roi_plus]  #Slicing the data to the ROI
        average3=np.mean(neighbourhood3)  #Average of the pixels
        
        stack_ave.append(average3)
    average4=np.array(stack_ave)
    
    fig4=plt.plot(wavelenghts, average4)
    plt.title(tittle1)
    return plt.show(fig4)

#Plotting the continums
def plot_continnums (fourier_cube,fitp_1000,fitp_2000,wavelengths,x_coord,y_coord):
    
    plt.plot(wavelengths,fourier_cube[:,y_coord,x_coord],label='Spectrum')
    plt.plot(wavelengths,np.polyval(fitp_1000,wavelengths), label='Fit 1000')
    plt.plot(wavelengths,np.polyval(fitp_2000,wavelengths), label='Fit 2000')    
    plt.legend()
    return 

#Compares the cubes before and after the filtering

def filter_comaprison (cube_1,cube_2,title1,title2,band):

    ratio_cubes=cube_1/cube_2  #Ratio of the two images

    cube_plus=cube_1+(cube_1*0.02)  #Calculating which pixels changed more than 2%
    cube_minus=cube_1-(cube_1*0.02)
    ratio_plus=cube_plus-cube_2
    ratio_minus=cube_2-cube_minus
    change_ratio=ratio_plus*ratio_minus
    change_ratio.data[change_ratio.data < 0]= 0

    fig4, axs = plt.subplots(ncols=2,nrows=2,figsize=(cube_1.shape[2]/cube_1.shape[1]*15,15))  #Plotting
    cube_1[band,:,:].plot.imshow(ax=axs[0,0],add_labels=False)
    axs[0,0].title.set_text(title1)
    cube_2[band,:,:].plot.imshow(ax=axs[0,1],add_labels=False)
    axs[0,1].title.set_text(title2)
    ratio_cubes[band,:,:].plot.imshow(ax=axs[1,0],robust=True,add_labels=False)
    axs[1,0].title.set_text('Ratio')
    change_ratio[band,:,:].plot.imshow(ax=axs[1,1],robust=True,add_labels=False)
    axs[1,1].title.set_text('Change over 2%')
    
    return 


#Convex hull plotting
def convexhull_plot(fourier_cube, wavelengths_full,mid_point,y_hull,x_hull):

    wavelengths=wavelengths_full[0:76]
    average4=fourier_cube[0:76,x_hull,y_hull]
    
    add_point=np.where(wavelengths==mid_point[x_hull,y_hull])[0]
    add_array2=np.vstack((wavelengths[add_point], fourier_cube[add_point,x_hull,y_hull])).T
    
    points = np.c_[wavelengths, average4]
    wavelengths, average4 = points.T
    augmented = np.concatenate([points, [(wavelengths[0], np.min(average4)-1), (wavelengths[-1], np.min(average4)-1)]], axis=0)
    hull = sp.spatial.ConvexHull(augmented)
    pre_continuum_points2 = points[np.sort([v for v in hull.vertices if v < len(points)])]
    pre_continuum_points22 = np.concatenate((pre_continuum_points2,add_array2), axis=0)
    pre_continuum_points22.sort(axis=0)
    continuum_points2=np.unique(pre_continuum_points22,axis=0)
    continuum_function2 = sp.interpolate.interp1d(*continuum_points2.T)
    average4_prime = average4 / continuum_function2(wavelengths)
    average4_prime[average4_prime >= 1]= 1
    
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(wavelengths, average4, label='Data')
    axes[0].plot(*continuum_points2.T, label='Continuum')
    axes[0].legend()
    axes[1].plot(wavelengths, average4_prime, label='Data / Continuum')
    axes[1].legend()

    return plt.show()