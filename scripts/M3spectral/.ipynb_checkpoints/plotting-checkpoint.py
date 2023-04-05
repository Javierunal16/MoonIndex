import matplotlib.pyplot as plt
import cv2
import numpy as np


#Viewing a cube
def cube_plot(cube_plot,band,size):
    plot=cube_plot[band,:,:].plot.imshow(aspect=cube_plot.shape[2]/cube_plot.shape[1], size=size, robust=True)
    return (plot)


#Viewing an image
def image_plot(image_imput,size2):
    plot2=image_imput.plot.imshow(aspect=image_imput.shape[1]/image_imput.shape[0], size=size2, robust=True)
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
def fourier_plot (gauss_filter2,fourie_cube2,band,filter_width2, filter_high2):
    fouraster2=cv2.dft(gauss_filter2.data[band,:,:], flags=cv2.DFT_COMPLEX_OUTPUT) 
    fouraster2_shift=np.fft.fftshift(fouraster2)
    magnitude_spectrum = 20 * np.log((cv2.magnitude(fouraster2_shift[:, :, 0], fouraster2_shift[:, :, 1])))  #For plotting
    y3,z3=gauss_filter2.data[band,:,:].shape
    
    rows, cols =gauss_filter2[band,:,:].shape
    mask = np.ones((rows, cols, 2), np.uint8)

    #Valentine Mask
    cv2.rectangle(mask, (0,((y3//2)-filter_high2)), (((z3//2)-filter_width2),((y3//2)+filter_high2)), 0, -1)
    cv2.rectangle(mask, (((z3//2)+filter_width2),((y3//2)-filter_high2)), (z3,((y3//2)+filter_high2)), 0, -1)

    #Mask applciation and reverse-fourier
    mfouraster2=fouraster2_shift*mask
    mshift_mask_mag = 20 * np.log(cv2.magnitude(mfouraster2[:, :, 0], mfouraster2[:, :, 1]))  #For plotting
    m_ishift2 = np.fft.ifftshift(mfouraster2)


    return_raster2=cv2.idft(m_ishift2)/(y3*z3)
    return_raster2= cv2.magnitude(return_raster2[:, :, 0], return_raster2[:, :, 1])
    
    fig = plt.figure(figsize=(20, 20))
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
def profile_comparison (cube_profile1, cube_profile2,wavelengths, title_profile1, title_profile2, pixelx,pixely):
    fig2, (ax22, ax23)=plt.subplots(1,2,figsize=(10,3))
    ax22.plot(wavelengths, cube_profile1[:,pixely,pixelx])
    ax22.title.set_text(title_profile1)
    ax23.plot(wavelengths, cube_profile2[:,pixely,pixelx], color='red')
    ax23.title.set_text(title_profile2)
    return plt.show(fig2)


#Single profile
def profile_plot (profile_singlecube, wavelengths,title_singleprofile, pixelx2,pixely2):
    plt.figure(figsize=(10,3))
    plt.plot(wavelengths, profile_singlecube[:,pixely2,pixelx2])
    plt.title(title_singleprofile)
    return (plt.plot)


#Plotting the continums
def plot_continnums (fourier_cube,fitp_1000,fitp_2000,wavelengths,x_coord,y_coord):
    
    plt.plot(wavelengths,fourier_cube[:,y_coord,x_coord],label='Spectrum')
    plt.plot(wavelengths,np.polyval(fitp_1000,wavelengths), label='Fit 1000')
    plt.plot(wavelengths,np.polyval(fitp_2000,wavelengths), label='Fit 2000')    
    plt.legend()
    return 