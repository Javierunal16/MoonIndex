#This function attach the wavelength to the cube
def attach_wavelen (cube_alone,wave):
    cube_alone.coords['wavelength'] = ('band', wave)
    cube_wave = cube_alone.swap_dims({'band':'wavelength'})
    return cube_wave
