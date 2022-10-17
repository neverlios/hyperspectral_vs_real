import pandas as pd
import numpy as np
import scipy


waves = pd.read_csv("../../awb/awb/spectral_data/sensitivities/iitp/canon600d.csv")
illums = pd.read_csv("./answers_v2.csv")


def double_discr(wavelength):
    wavelength_mid = np.concatenate((np.zeros(1),wavelength))
    a=(wavelength_mid+np.concatenate((wavelength,np.zeros(1))))/2
    a=np.delete(a,(0,-1))
    last_wavelength = []
    for i in range(wavelength.shape[0]-1):
        last_wavelength.append(wavelength[i])
        last_wavelength.append(a[i])
    last_wavelength.append(wavelength[-1])
    last_wavelength = np.array(last_wavelength)
    return last_wavelength

#double digits
wavelength = double_discr(np.array(waves['wavelength']))
red = double_discr(np.array(waves['red']))
green = double_discr(np.array(waves['green']))
blue = double_discr(np.array(waves['blue']))
#count sencetivity matrix [3 x n]
rgb = np.vstack((red,green,blue))

# d50 = rgb @ illums['D50'][400:721:5]
# d65 = rgb @ illums['D65'][400:721:5]
# d75 = rgb @ illums['D75'][400:721:5]

def clear_rgb():
    return rgb

def spawn():
    return np.vstack((d50, d65, d75))



