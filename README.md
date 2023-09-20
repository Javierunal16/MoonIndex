# MoonIndex
MoonIndex is a python library to create spectral indexes from the Moon Mineralogy Mapper (M<sup>3</sup>). The majority of indexes were collected from the litrature, some were formualted in this work. The tool uses map-projected hyperspectral cubes and common python libraries to achieve the indexes. The general procces of the tool consist of: preparation, filtration, continuum removal and creation of the indexes.

## Instalation
The instalation can be done via PyPI using pip:

`pip install MoonIndex`

Or after downloading the MoonIndex-1.0.tar.gz file under dist:

`pip install MoonIndex-1.0.tar.gz`

## Requirements
To use the package you first need map-projected (M<sup>3</sup>) cubes. This can be achieved using the USGS Integrated Software for Imagers and Spectrometers (ISIS), see https://github.com/DOI-USGS/ISIS3. The file Wavelengths.txt is also needed during the procces. 

## Example
The notebook called M3_Indexes.ipynb under scripts details the workflow followed to obtain the indexes.

After defining the paths to the cubes and the Wavelegnths.txt file, the whole procces can be made with two functions:

`M3_cube=MoonIndex.preparation.attach_wavelen(input_cube,wavelengths)`

And then:

`M3_full_CH=MoonIndex.indexes.indexes_total_CH(M3_cube,wavelengths)` With the convex hull continuum-removal method.

Or:

`M3_full_LF=MoonIndex.indexes.indexes_total_LF(M3_cube,wavelengths)` With the linear-fit continuum removal method.
