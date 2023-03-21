import matplotlib.pyplot as plt

#Viewing the image
def M3_plot(cube_plot,band,size):
    plot=cube_plot[band,:,:].plot.imshow(aspect=cube_plot.shape[2]/cube_plot.shape[1], size=size, robust=True)
    return (plot)

#Comparing images
def plot_comparison (cube_plot1, cube_plot2, title1, title2, band):
    plot1, ax=plt.subplots(1,2)
    ax[0].imshow(cube_plot1[band,:,:])
    ax[0].set_title(title1)
    ax[1].imshow(cube_plot2[band,:,:])
    ax[1].set_title(title2)
    return(plt.show(plot1))