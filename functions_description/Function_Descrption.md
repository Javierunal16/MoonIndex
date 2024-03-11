# Description of MoonIndex functions
The functions of the package are divided in four classes: Preparation, Filtration, Plotting, and Indexes. Here we explain their use.  


## Preparation

**MoonIndex.preparation.attach_wave (cube_alone,wave):** This function eliminates the two first empty bands, turn all anomalous values to nodata and attach the wavelengths. **Inputs** the M<sup>3</sup> cube, and wavelengths file.

**MoonIndex.preparation.crop_cube (initial_cube,minnx,minny,maxx,maxy):** Crop the prepared cube to a desired location, it can be done using the unit of the coordinate system or the number of lines and columns of the file. **Inputs** are the cube, minx, miny, maxx, maxy.

**MoonIndex.preparation.midpoint(fourier_cube,wavelengths,peak_distance,peak_prominence):** Find the midpoint to limit the two absorption bands. **Inputs** are the filtered cube, the wavelengths, the minimum distance between peaks (6 is recommended), and the minimum prominence of the peaks (0.002 is recommended).

**MoonIndex.preparation.convexhull_removal(fourier_cube, wavelengths_full,mid_point):** Remove the continuum of the spectra using the convex-hull method. **Inputs** are the filtered cube, the wavelengths, and the midpoint.

**MoonIndex.preparation.find_minimums_ch (hull_cube,midpoint,wavelengths2):** This function finds the minimums around the 1 $\mu$m and 2 $\mu$m bands for the convex hull method. **Inputs** are the continuum-removed cube, the midpoint, and the wavelegnths.

**MoonIndex.preparation.find_shoulders_ch (hull_cube2,midpoint,min_1000,min_2000, wavelengths3):** Find the shoulders around the 1 $\mu$m and 2 $\mu$m bands minimums for the convex hull method. **Inputs** are the continuum-removed cube, the midpoint,the minimuum at 1 $\mu$m, the minimuum at 2 $\mu$m, and the wavelegnths.

**MoonIndex.preparation.continuum_removal_lf (gauss_cube,wavelengths2,order1,order2):** Remove the continuum of the spectra using the convex-hull method. **Inputs** are the filtered cube, the wavelegths, the polynomial order for the band at 1 $\mu$m, and the polynomial order for the band at 2 $\mu$m.

**MoonIndex.preparation.find_minimuumslf (lf_cube,wavelengths):** This function finds the minimums around the 1 $\mu$m and 2 $\mu$m bands for the linear fit method.**Inputs** are the continuum-removed cube and the wavelegnths.

**MoonIndex.preparation.find_shoulders_lf (lf_cube,min_1000lf,min_2000lf, wavelengths):** Find the shoulders arounf the the 1 $\mu$m and 2 $\mu$m bands minimums for the linear fit method. **Inputs** are the continuum-removed cube,the minimuum at 1 $\mu$m, the minimuum at 2 $\mu$m, and the wavelegnths.

**MoonIndex.preparation.continuum_1000 (filtered_cube,hull_cube,wavelengths,x_continum,y_continum):** Function to remove the continuum of an specific pixel at 1 $\mu$m (not intended for use of the user). **Inputs** are the filtered cube, the continuum-removed cube, the wavelengths, the x position of the pixel, the y position of the pixel.

**MoonIndex.preparation.continuum_2000 (filtered_cube,hull_cube,wavelengths,x_continum,y_continum):** Function to remove the continuum of an specific pixel at 2 $\mu$m (not intended for use of the user). **Inputs** are the filtered cube, the continuum-removed cube, the wavelengths, the x position of the pixel, the y position of the pixel.

## Filtration

**MoonIndex.filtration.fourier_filter(original_cube,percentage_width,percentage_high):** Performs the fourier filtation of the cube in the spatial domain. Inputs are the prepared cube, width of the filter in percentange, and high of the filter in percentage.

**MoonIndex.filtation.gauss_filter (cube_filter1,wavelen):** Performs the gaussian filter in the spectral domain. **Inputs** are the fourier-filtered cube, and the wavelengths file.

## Plotting

**MoonIndx.plotting.cube_plot(cube_plot,size,title):** Plots a cube or RGB composite and normalizes the values to 0-255. **Inputs** are the cube the size of te plot, and the title.

**MoonIndex.plotting.image_plot(image_input,size2,title):** Plots a single band image with an "Spectral" colormap. **Inputs** the image, the size fo the graph, and the title.

**MoonIndex.plotting.plot_comparison (cube_plot1, cube_plot2, band1, band2, title1, title2):** Plots two selected band of a cube or cubes, to compare between them. **Inputs** the first cube, the second cube,the band of the first cube, the band of the second cube, the title of the first cube, and the title of the second cube.

**MoonIndex.plotting.fourier_plot (gauss_filter2,band,percentage_width, percentage_high):** Plots the steps of the Fourier filter to check the results. **Improtant** this function is only for viewing, to change the filtering of the cube use the homonimous function under __filtration__. **Inputs** are the cube, the band to check, the width of the filter in percentange, and high of the filter in percentage.

**MoonIndex.plotting.profile_plot (wavelengths,profile_singlecube,title_singleprofile, pixelx2,pixely2,roi):** Plots a single spectrum from a pixel,the spectra is averaged in a 2x2 window around the selected pixel. **Inputs** are the wavlengths, the cube, the title of the cube, the x position of the pixel to plot, and the y position of the pixel to plot. 

**MoonIndex.plotting.profiles_comparison(wavelengths,first_cube, second_cube,tittle1,tittle2, in_x, in_y,roi):** Plot two spectral signatures to compare, the spectra is averaged in a 2x2 window around the selected pixel. **Inputs** are the wavlengths, the first cube, the second cube, the title of the first cube, the title of the second cube, the x position of the pixel to plot, and the y position of the pixel to plot.

**MoonIndex.plotting.filter_comparison (cube_1,cube_2,title1,title2,band):** Plots a comparison between the cubes before and after the filtration. It also plots the ratio between the cubes, and an iamge showing the pixels that changed more than 2% in black. **Inputs** are the cube before, the cube after, the title of te first one, the title of the second one, and the band to ceck.

**MoonIndex.plotting.convexhull_plot(fourier_cube, wavelengths_full,mid_point,y_hull,x_hull):** Plots the result of the convex hull continuum-removal method for a pixel. **Improtant** this function is only for viewing, to change the removal use the homonimous function under __Preparation__. **Inptus** are the filtered cube, the wavelengths, the midpoint, the y position of te pixel, and the x position of the pixel.

**MoonIndex.plotting.linearfit_plot(gauss_cube, removed_cube, wavelengths,y_plot,x_plot):** Plots the result of the linear fit continuum-removal method for a pixel. **Improtant** this function is only for viewing, to change the removal use the homonimous function under __Preparation__. **Inptus** are the filtered cube, the continuum-removed cube, the y position of te pixel, and the x position of the pixel.

## Indexes

**MoonIndex.indexes.indexes_total_CH(M3_cube,wavelengths):** This function peforms the full procces of indexes creation usign the convex-hull removal method, from the filtering to the indexes generation. The **attach_wavelen (cube_alone,wave)** function must still be runned beforehand, but the user can inputs the full cube after that (will take a long time), or crop it with rop_cube **(initial_cube,minnx,minny,maxx,maxy):** to save time. **Inputs** The cube, wavelengths.

**MoonIndex.indexes.indexes_total_LF(M3_cube,wavelengths,order1,order2):** This function peforms the full procces of indexes creation usign the linear fit removal method, from the filtering to the indexes generation. The **attach_wavelen (cube_alone,wave)** function must still be runned beforehand, but the user can inputs the full cube after that (will take a long time), or crop it with rop_cube **(initial_cube,minnx,minny,maxx,maxy):** to save time. **Inputs** The cube, wavelengths, te polynomial roder for the 1 $\mu$m band, and the polynomial order for the 2 $\mu$m band.

**MoonIndex.indexes.R540(gauss_cube):** Creates the reflectance at 540 nm. **Input** the filtered cube.

**MoonIndex.indexes.R1580(gauss_cube):** Creates the reflectance at 1580 nm. **Input** the filtered cube.

**MoonIndex.indexes.olivine(gauss_cube):** Creates the olivine index. **Input** the filtered cube.

**MoonIndex.indexes.spinel(gauss_cube):** Creates the spinel index. **Input** the filtered cube.

**MoonIndex.indexes.chromite(gauss_cube):** Creates the chromite index. **Input** the filtered cube.

**MoonIndex.indexes.iron(gauss_cube):** Creates the iron index. **Input** the filtered cube.

**MoonIndex.indexes.titanium(gauss_cube):** Creates the titanium index. **Input** the filtered cube.

**MoonIndex.indexes.clementine(gauss_cube):** Creates the clementine-like index. The output is a RGB composite. **Input** the filtered cube.

**MoonIndex.indexes.spanpx(gauss_cube):** Creates the spanpx index. The output is a RGB composite **Input** the filtered cube.

**MoonIndex.indexes.band_center(minimum):** Creates the band minimuum, it works for both absorption bands by changing the corresponding inputs. **Input** the minimum image.

**MoonIndex.indexes.band_depth(hull_cube,minimum,wavelengths):** Creates the band depth index, it works for both absorption bands by changing the corresponding inputs. **Inputs** the continuum-removed cube, the minimum image, and the wavelengths.

**MoonIndex.indexes.BA (hull_cube,wavelengths,shoulder0,shoulder1):** Creates the band area index, it works for both absorption bands by changing the corresponding inputs. **Inputs** the continuum-removed cube, the wavelengths,the left shoudler of the band, and the right shoudler of the band.

**MoonIndex.indexes.ASY (hull_cube,wavelengths,shoulder0,shoulder1,min1000):** Creates the band asymmetry index, it works for both absorption bands by changing the corresponding inputs. **Inputs** the continuum-removed cube, the wavelengths,the left shoudler of the band, the right shoudler of the band, and the minimum.

**MoonIndex.indexes.SSI(gauss_cube,shoulder1, wavelengths):** Creates the sprectral slope index. This is done between the 540 nm band and the left shoudler of the 1 $\mu$m band **Inputs** the filtered cube, the right shoudler of the 1 $\mu$m band, and the wavelengths.

**MoonIndex.indexes.IBDI(hull_cube):** Calculated the integrated band depth arounf the 1 $\mu$m band. Internal process. **Input** the continuum-removed cube.

**MoonIndex.indexes.IBDII(hull_cube):** Calculated the integrated band depth arounf the 2 $\mu$m band. Internal process. **Input** the continuum-removed cube.

**MoonIndex.indexes.RGB1(fourier_cube,SSI_cube,BDI_cube,BDII_cube):** Creates the RGB1 index. The output is a RGB composite. **Inputs** are the filtered cube, the spectral slope index, the band depth at 1 $\mu$m, and the band depth at 2 $\mu$m.

**MoonIndex.indexes.RGB2 (gauss_cube,SSI_cube, R540_cube, BCII_cube):** Creates the RGB2 index. The output is a RGB composite. **Inputs** are the filtered cube, the spectral slope index, the R540 index, and the band center at 2 $\mu$m.

**MoonIndex.indexes.RGB3 (gauss_cube,SSI_cube,R540_cube,BCI_cube):** Creates the RGB3 index. The output is a RGB composite. **Inputs** are the filtered cube, the spectral slope index, the R540 index, and the band center at 1 $\mu$m.

**MoonIndex.indexes.RGB4(fourier_cube,wavelengths,shoulder0,shoulder1,minimum_1000,minimum_2000):** Creaed the RB4 index. The output is a RGB composite. **Inputs** are the filtered cube, the wavelengths, the left shoudler of the 1 $\mu$m band, the right shoudler of the 1 $\mu$m, the minimum at 1 $\mu$m, and the minimum at 2 $\mu$m.

**MoonIndex.indexes.RGB5 (fourier_cube,wavelengths,shoulder0,shoulder1,min1000,min2000):** Creaed the RB5 index. The output is a RGB composite. **Inputs** are the filtered cube, the wavelengths, the left shoudler of the 1 $\mu$m band, the right shoudler of the 1 $\mu$m, the minimum at 1 $\mu$m, and the minimum at 2 $\mu$m.

**MoonIndex.indexes.RGB6 (hull_cube):** Creates the RGB6 index. THe output is a RGB composite. **Input** the continuum-removed cube.

**MoonIndex.indexes.RGB7 (gauss_cube,R1580,IBD1000,IBD2000):** Creates the RGB7 index. The output is a RGB composite. **Inputs** are the filtered cube, the R1580 index, the integrated band depth at 1 $\mu$m, and the integrated band depth at 2 $\mu$m.

**MoonIndex.indexes.RGB8 (fourier_cube,hull_cube):** Creates the RGB8 index. The output is a RGB composite. **Inputs** are the filtered cube and the continuum-removed cube.