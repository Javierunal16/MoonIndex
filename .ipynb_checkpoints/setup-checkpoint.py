from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description =f.read()

setup(
    name='MoonIndex',
    version='2.0.1',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Javier Suarez',
    author_email='jsuarezvalencia@constructor.university',
    license='GPL',
    install_requires=['astropy>=5.3.4'
    , 'jmespath>=1.0.1'                  
    , 'matplotlib>=3.8.0'
    , 'numpy>=1.26.1'
    , 'opencv-python>=4.8.1.78'
    , 'pysptools>=0.15.0'
    , 'python-dateutil>=2.8.2'
    , 'rasterio>=1.3.9'
    , 'rioxarray>=0.15.0'
    , 'scikit-learn>=1.3.2'
    , 'scipy>=1.11.3'
    , 'specutils>=1.12.0'
    , 'xarray>=2023.10.1'
                     ],
    python_requires='>=3.10'
)