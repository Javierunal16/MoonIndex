from setuptools import setup, find_packages

setup(
    name='MoonIndex',
    version='1.0',
    packages=find_packages(),
    author='Javier Suarez',
    author_email='jsuarezvalencia@constructor.university',
    license='GPL',
    install_requires=['astropy==5.3.4'
    , 'matplotlib==3.8.0'
    , 'matplotlib-inline==0.1.6'
    , 'numpy==1.26.1'
    , 'opencv-python==4.8.1.78'
    , 'plotly>=5.17.0'
    , 'pyproj>=3.6.1'
    , 'pysptools==0.15.0'
    , 'python-dateutil==2.8.2'
    , 'rasterio==1.3.9'
    , 'rioxarray==0.15.0'
    , 'scikit-learn==1.3.2'
    , 'scipy==1.11.3'
    , 'specutils==1.12.0'
    , 'xarray==2023.10.1'
                     ],
    python_requires='==3.12.0'
)