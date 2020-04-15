import numpy as np
from matplotlib import pyplot as plt

x_stag = 5 # m
a = 3 # m

def velocity_field(x_i, z_i, U_inf, a, x_stag):



    m = np.pi*U_inf/a * (a**2 - x_stag**2)

    v_x = U_inf + m/(2*np.pi) * ( (x_i + a)/((x_i + a)**2 + z_i**2) - (x_i - a)/((x_i - a)**2 + z_i**2) )
    v_z = (m*z_i)/(2*np.pi) * (1/((x_i + a)**2 + z_i**2) - 1/((x_i - a)**2 + z_i**2))

    return v_x, v_z