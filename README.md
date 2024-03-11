

[![DOI](https://zenodo.org/badge/614282836.svg)](https://zenodo.org/doi/10.5281/zenodo.10036998)


# MoonIndex
MoonIndex is a python library to create spectral indexes from the Moon Mineralogy Mapper (M<sup>3</sup>). The majority of indexes were collected from the litrature, some were formualted in this work. The tool uses map-projected hyperspectral cubes and common python libraries to achieve the indexes. The general procces of the tool consist of: preparation, filtration, continuum removal and creation of the indexes (Tested on Python 3.11).

![alt text](https://github.com/Javierunal16/Index/blob/main/README_files/Figure%204.png)

## Requirements
To use the package you first need map-projected (M<sup>3</sup>) cubes. This can be achieved using the USGS Integrated Software for Imagers and Spectrometers (ISIS), see https://github.com/DOI-USGS/ISIS3. The file Wavelengths.txt is also needed during the procces. Moonindex requires python 3.12.

## Instalation
The instalation can be done via PyPI using pip:

`pip install MoonIndex`

Or after downloading the MoonIndex-1.0.tar.gz file under dist:

`pip install MoonIndex-1.0.tar.gz`

## Example
The notebook called M3_Indexes.ipynb under scripts details the workflow followed to obtain the indexes. We recommend slicing the cubes before creating the indexes, since the process is intensive. The sample cube used in this notebook can be found at: https://zenodo.org/records/10014564

## Straighforwad proccesing

First import the cube and the Wavelegnths.txt file using rioxarray and numpy:

`input_cube=rioxarray.open_rasterio('input_cube.tiff')`  
`wavelengths=numpy.loadtxt('Wavelength.txt', delimiter=",")`
`wavelengths=(wavelengths).astype(numpy.float32)`

Then a function to prepare the data:

`M3_cube=MoonIndex.preparation.attach_wavelen(input_cube,wavelengths)`

If you desire to crop the cube, you can use:

`M3_cube=MoonIndex.preparation.crop_cube(M3_cube,x1,y1,x2,y2)` Where x and y are the coordinates in the reference system of your cube.

And then, to create the indexes:

`M3_full_CH=MoonIndex.indexes.indexes_total_CH(M3_cube,wavelengths)` With the convex hull continuum-removal method.

Or:

`M3_full_LF=MoonIndex.indexes.indexes_total_LF(M3_cube,wavelengths,2,1)` With the linear-fit continuum removal method (the last two variables are the polynomial order for the fit around the 1000 and 2000 nm bands).

## List of indexes
| Index Name                          | Abrev. Name | Product type    | Source                     |
| ----------------------------------- | ----------- | ------------- | -------------------------- |
| Reflectance at 540 nm               | R540        | Parameter     | Adams and McCord (1971)    |
| Band center at 1 µm                 | BCI         | Parameter     | Adams (1974)               |
| Band center at 2 µm                 | BCII        | Parameter     | Adams (1974)               |
| Band depth at 1 µm                  | BDI         | Parameter     | Adams (1974)               |
| Band deepth at 2 µm                 | BDII        | Parameter     | Adams (1974)               |
| Spectral slope at 1 µm              | SS          | Parameter     | Hazen et al. (1978)        |
| Clementine-like red channel         | Clem RED    | Parameter     | Lucey et al. (2000)        |
| Clementine-like green channel       | Clem GREEN  | Parameter     | Lucey et al. (2000)        |
| Clementine-like blue channel        | Clem BLUE   | Parameter     | Lucey et al. (2000)        |
| Band depth at 1.9 µm                | BD1900      | Parameter     | Bretzfelder et al. (2020)  |
| Integrated band depth at 1 µm       | IBDI        | Parameter     | Bretzfelder et al. (2020)  |
| Integrated band depth at 2 µm       | IBDII       | Parameter     | Bretzfelder et al. (2020)  |
| Band area at 1 µm                   | BAI         | Parameter     | Cloutis et al. (1986)      |
| Band area at 2 µm                   | BAII        | Parameter     | This papper                |
| Band asymmetry at 1 µm              | ASYI        | Parameter     | Cloutis et al. (1986)      |
| Band asymmetry at 2 µm              | ASYII       | Parameter     | This papper                |
| Olivine parameter                   | Ol          | Parameter     | Corley et al. (2018)       |
| Spinel ratio                        | Sp1         | Parameter     | Pieters et al. (2014)      |
| Spinel ratio                        | Sp2         | Parameter     | Moriarty III et al. (2022) |
| Pyroxene ratio                      | Px          | Parameter     | Pieters et al. (2014)      |
| Pure anorthosite ratio              | An          | Parameter     | Pieters et al. (2014)      |
| Band depth at 950 nm                | BD950       | Parameter     | Besse et al. (2011)        |
| Band depth at 1.05 µm               | BD1050      | Parameter     | Besse et al. (2011)        |
| Band depth at 1.25 µm               | BD1250      | Parameter     | Besse et al. (2011)        |
| Reflectnace at 1.58 µm              | R1580       | Parameter     | Besse et al. (2011)        |
| Iron oxide parameter                | Fe          | Parameter     | Wu et al. (2012)           |
| Titanium parameter                  | Ti          | Parameter     | Wu et al. (2012)           |
| Chromite parameter                  | Cr          | Parameter     | This paper                 |
| RGB Clementine-like color composite | Clem        | RGB composite | Lucey et al,( 2000)        |
| Color composite 1                   | RGB 1       | RGB composite | Zambon et al. (2020)       |
| Color composite 2                   | RGB 2       | RGB composite | Zambon et al. (2020)       |
| Color composite 2                   | RGB 3       | RGB composite | Zambon et al. (2020)       |
| Color composite of band area        | RGB4        | RGB composite | Horgan et al. (2014)       |
| Color composite of band asymmetry    | RGB5        | RGB composite | Horgan et al. (2014)       |
| Color composite 6                   | RGB6        | RGB composite | Besse et al. (2011)        |
| Color composite 7                   | RGB7        | RGB composite | Besse et al. (2011)        |
| Near infrarred color composite      | RGB8        | RGB composite | Bretzfelder et al. (2020)  |
| Color composite of spinel           | Spanpx      | RGB composite | Moriarty III et al. (2022) |
