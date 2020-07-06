"""
Regen power simulation in cylindrical/oval wind field


Author: Midas Gossye


"""

import numpy as np
from matplotlib import pyplot as plt
import csv
from matplotlib import ticker, cm


# Define general hill parameters

R_ridge = 50.0 # [m] Radius of the hill/ridge
U_inf = -25.0 # [m/s] Mean wind speed (negative to have flow flow right to left)
flow_type = 'oval' # Defines if hill is cylindrical or oval, options: 'oval' and 'circle'

# Oval shaped hill parameters
a = 45 # m, defines loci x-position
x_stag = 67 # m, defines a-axis of standard oval
b = np.sqrt(x_stag**2 - a**2) # calculates b parameter of oval

surf_rough = 0.5 # m, surface roughness

oval_pointsx = np.array([])
oval_pointsz = np.array([])

def wind_comp(x_i, z_i, U):
    ## Wind field calculation function

    # Description: Given an x and z coordinate and the mean wind speed it calculates the wind field velocity components
    #              using potential flow theory and a logarithmic boundary layer

    if flow_type == 'oval':

        m = np.pi * U_inf / a * (x_stag ** 2 - a **2)
        v_x = U_inf + m / (2 * np.pi) * ((x_i + a) / ((x_i+0.001 + a) ** 2 + z_i ** 2) - (x_i+0.001 - a) / ((x_i+0.001 - a)**2 + z_i**2))
        v_z = -(m * z_i) / (2 * np.pi) * (1 / ((x_i+0.001 + a)**2 + z_i**2) - 1 / ((x_i+0.001 - a)**2 + z_i**2))

        ellipse_eq = (x_i ** 2) / (x_stag ** 2) + (z_i ** 2) / (b ** 2)
        if ellipse_eq < 1:
            return 0.0, 0.0

        if x_i < x_stag:

            z_ellipse = -b / x_stag * np.sqrt(x_stag ** 2 - x_i ** 2)

            mult_factor = (np.log(-(z_i - z_ellipse) / surf_rough)) / (np.log(-(-70 - z_ellipse) / surf_rough))
            if mult_factor < 0:
                print("z_ab =", -(z_i - z_ellipse), "mult =", mult_factor)
            if not (np.isnan(mult_factor)):
                v_x *= mult_factor
                v_z *= mult_factor

        else:
            if z_i < 0:
                z_ellipse = -0.01
                mult_factor = (np.log(-(z_i - z_ellipse) / surf_rough)) / (np.log(-(-70 - z_ellipse) / surf_rough))
                v_x *= mult_factor
                v_z *= mult_factor
            else:
                return 0.0, 0.0
        return v_x, v_z

    elif flow_type == 'circle':
        theta = np.arctan2(-z_i, x_i)
        r = np.sqrt(x_i**2 + z_i**2)

        global R_ridge
        u_r = (1-((R_ridge**2)/(r**2)))*U*np.cos(theta)
        u_th = -(1+((R_ridge**2)/(r**2)))*U*np.sin(theta)

        if R_ridge**2 - x_i**2 > 0:
            z_circle = -np.sqrt(R_ridge**2 - x_i**2)

        else:
            z_circle = -0.01
        mult_factor = (np.log(-(z_i - z_circle) / surf_rough)) / (np.log(-(-70 - z_circle) / surf_rough))
        u_x = np.cos(theta)*u_r - np.sin(theta)*u_th*mult_factor
        u_z = (np.sin(theta)*u_r + np.cos(theta)*u_th)*mult_factor


        if r<R_ridge:
            return 0.0,0.0

        return u_x, u_z

vwind_comp = np.vectorize(wind_comp) # Vectorizes the wind comp function so arrays can be parsed and returned by the new
                                     # vwind_comp function

def calc_jacobian(i, j, dx, dz, u_xs, u_zs):
    # calculates the jacobian of the wind field using the central difference formula for the
    # first spatial derivatives

    J = np.array([[(u_xs[j, i+1]-u_xs[j, i-1])/2*dx, (u_xs[j+1, i]-u_xs[j-1, i])/2*dz],
                  [(u_zs[j, i+1]-u_zs[j, i-1])/2*dx, (u_zs[j+1, i]-u_zs[j-1, i])/2*dz]])
    return J

def x_to_i(x_coor, max_i, x_coords):
    # converts x coordinate in meters to closest discrete i index of grid

    i = int(np.around(x_coor*(1/dx)) + x_coords[-1])
    if i>max_i:
        return max_i-2
    elif i < 0:
        return 0
    else:
        return i

def z_to_j(z_coor, max_j, z_coords):
    # converts z coordinate in meters to closest discrete j index of grid
    j = int(np.around(z_coor*(1/dz)) + z_coords[-1])
    if j>max_j:
        return max_j-2
    elif j < 0:
        return 0
    else:
        return j



x_vels = np.array([])
z_vels = np.array([])

## Specify 2D grid resolution [in meters]
dx = 0.5
dz = -0.5
## ======================================

## Create x and z coordinate array
x_coords = np.arange(-5, 100, dx) # specifies range of x coords
z_coords = np.arange(0, -100, dz)  # specifies range of z coords
## ======================================

xs, zs = np.meshgrid(x_coords, z_coords) # create meshgrid to construct wind velocity field

Winds_u, Winds_v = vwind_comp(xs, zs, U_inf) # generates 2D matrices with horizontal and vertical wind velocity components

Total_wind = np.sqrt(np.power(Winds_u, 2) + np.power(Winds_v, 2)) # generates 2D matrix with total wind velocities

jacobians = np.zeros((len(x_coords), len(z_coords), 2,2)) # Initialize empty jacobian matrix

for i in range(1, x_coords.size-1):
    for j in range(1, z_coords.size-1):
        #print("i:", i)
        #print("j:", j)
        jacobians[i, j] = calc_jacobian(i, j, dx, dz, Winds_u, Winds_v)



if flow_type == 'circle':
    # generates cartesian coordinates where edge of hill is present (for circular shaped hill)
    thetas = np.arange(0, 180, 0.01)
    xs_hill = R_ridge*np.cos(thetas)
    zs_hill = -R_ridge*np.sin(thetas)

else:
    # generates cartesian coordinates where edge of hill is present (for oval shaped hill)
    ts = np.arange(0, 2 * np.pi, 0.01)
    u = np.tan(ts / 2)
    xs_hill = x_stag * (1 - u ** 2) / (u ** 2 + 1)
    zs_hill = -(2 * b * u) / (u ** 2 + 1)


fig, ax = plt.subplots(1,1) # initialize pyplot figures (2 rows, 3 columns)
#ax[0][0].invert_yaxis() # invert y-axis on plot (represent z-axis)
ax.invert_yaxis()
skip=(slice(None,None,10),slice(None,None,10))
#ax[0][0].quiver(xs[skip], zs[skip], Winds_u[skip], Winds_v[skip], Total_wind[skip], cmap='Reds') # plots vector field
qv1 = ax.quiver(xs[skip], zs[skip], Winds_u[skip], Winds_v[skip], Total_wind[skip], cmap='cool') # plots vector field
#ax[0][0].streamplot(xs, zs, Winds_u, -Winds_v, density=1) # creates streamlines on plot
ax.streamplot(xs, zs, Winds_u, -Winds_v, density=1) # creates streamlines on plot
#ax[0][0].plot(xs_hill, zs_hill, '-r')
ax.plot(xs_hill, zs_hill, '-r')


## initial conditions for simulated aircraft in wind field
m = 4.0 # kg

S = 1 # m^2
rho = 1.225 # kg/m^3
g = 9.80665 # m/s^2
CL_alpha = 5.7 # 1/rad
alpha_0L = np.deg2rad(-4)
AR = 6
e = 0.8
CD_0 = 0.1
W = m*g
V_a = 25 # initial airspeed for simulated aircraft m/s
gamma_a = -10*(np.pi/180) # deg/rad
dgamma_dt = 0 # rad/s
dt = 0.01 # s
t = 0 #s
x_i = 17
z_i = -5
D_prop = 0.3 # m

Cdi_coef = 1/(np.pi*AR*e) # 1/pi*A*e

P_maxs = (16/27)*(rho/2)* Total_wind**3 * (np.pi*D_prop**2)/4

S_turb_spec = 0.14 # 0.006 for smallest point at gridsize of 0.1


def calc_eq():
    C_L_req = W/(0.5*rho*np.power(Total_wind, 2)*S) * (np.abs(Winds_u)/Total_wind)
    C_D_req = W / (0.5 * rho * np.power(Total_wind, 2) * S) * (np.abs(Winds_v) / Total_wind)

    C_D_min_ach = CD_0+np.power(C_L_req, 2)*Cdi_coef
    C_D_max_ach = C_D_min_ach + 1*(2/9)*S_turb_spec

    C_D_turb = C_D_req - C_D_min_ach

    D_turb = 0.5*rho*np.power(Total_wind, 2)*S*C_D_turb

    alpha = C_L_req / CL_alpha + alpha_0L

    stall = (alpha > np.deg2rad(15)) & (alpha < np.deg2rad(-10))
    print("stall condition:", stall[stall == True])
    C_L_req[stall] = np.nan
    C_D_req[stall] = np.nan
    C_D_min_ach[stall] = np.nan
    C_D_max_ach[stall] = np.nan
    alpha[stall] = np.nan

    eq_points = (C_D_req > C_D_min_ach) & (C_D_req < C_D_max_ach)
    P_turb = 0.5*rho*S*np.power(Total_wind, 3)*C_D_turb
    P_turb[eq_points == False] = np.nan
    alpha[eq_points == False] = np.nan
    D_turb[eq_points == False] = np.nan
    return P_turb, np.rad2deg(alpha), D_turb

# vars to store
ts = np.array([])
V_as = np.array([])
x_is = np.array([])
z_is = np.array([])
W_xs_s = np.array([])
W_zs_s = np.array([])
Ps_turb = np.array([])
V_airs = np.array([])
#print(wind_comp(-11, -12, U_inf))

C_L_opt = np.sqrt(3*np.pi*AR*e*CD_0)
C_D_opt = CD_0 + C_L_opt**2/(np.pi*AR*e)

P_turbs, alphas_eq, D_turb = calc_eq()


while t < 1:

    if np.isnan(x_i):
        print("BIG ERROR")
    i = x_to_i(x_i, len(x_coords)-1, x_coords)
    j = z_to_j(z_i, len(z_coords)-1, z_coords)

    U_x, U_z = wind_comp(x_i, z_i, U_inf)
    V_air = np.sqrt(U_x**2 + U_z**2)
    V_airs = np.append(V_airs, V_air)
    U_z = -1*U_z
    x_dot_i = V_a * np.cos(gamma_a) + U_x
    z_dot_i = -V_a * np.sin(gamma_a) + U_z
    W_xs_s = np.append(W_xs_s, U_x)
    W_zs_s = np.append(W_zs_s, U_z)
    #print(i, j)
    gamma_test = np.arctan2(z_dot_i-U_z, x_dot_i -  U_x)
    #print(gamma_test*180/np.pi)


    M1 = np.array([np.sin(gamma_a), np.cos(gamma_a)])
    M3 = np.array([x_dot_i, z_dot_i])
    M3 = np.reshape(M3, (2,1))
    temp_M = np.matmul(M1, jacobians[i,j])
    #print("temp_M: ", temp_M)
    #print("M3: ", M3)
    MUL = np.matmul(temp_M, M3)

    L_m = g*np.cos(gamma_a) + V_a*dgamma_dt - MUL

    C_L = L_m*m / (0.5*rho*V_a**2*S)
    C_D = CD_0 + (C_L**2)/(np.pi*AR*e)

    Drag_m_turbine = (1/9*m) * rho * V_a**2 * np.pi * D_prop**2
    P_turbine = Drag_m_turbine*m
    Ps_turb = np.append(Ps_turb, P_turbine)


    D_m = (1/(2*m)) * rho * V_a**2 * S * CD_0 + ((m*(L_m)**2)) / (0.5*rho*V_a**2 * S * np.pi * AR * e) + Drag_m_turbine
    tan_gamma = (Drag_m_turbine + D_m)/(L_m)
    gamma_goal = np.arctan2(Drag_m_turbine + D_m, L_m)

    # if gamma_goal-gamma_test > 2*(np.pi/180):
    #     dgamma_dt = -15*(np.pi/180)
    # elif gamma_goal-gamma_test < -2*(np.pi/180):
    #     dgamma_dt = 15 * (np.pi / 180)
    # else:
    #     dgamma_dt = 0

    #print(gamma_goal*(180/np.pi))
    N1 = np.array([np.cos(gamma_a), -np.sin(gamma_a)])
    N3 = np.array([x_dot_i, z_dot_i])
    N3 = np.reshape(N3, (2, 1))
    temp_N = np.matmul(N1, jacobians[i,j])
    #print("temp_N: ", temp_N)
    #print("N3: ", N3)
    NUL = np.matmul(temp_N, N3)
    dVa_dt = -g*np.sin(gamma_a) - D_m - NUL

    ts = np.append(ts, t)
    V_as = np.append(V_as, V_a)
    x_is = np.append(x_is, x_i)
    z_is = np.append(z_is, z_i)

    V_a += dVa_dt*dt
    x_i += x_dot_i*dt
    z_i += z_dot_i*dt
    gamma_a += dgamma_dt*dt
    t += dt

def get_local_min_h_dot(V_loc):
    min_h_dot = (rho * S * CD_0) / (2 * m * g) * V_loc ** 3 + (2 * m * g * Cdi_coef) / (rho * S * V_loc)
    return min_h_dot

#ax[0][0].plot(x_is, z_is)
#ax.plot(x_is, z_is)
#min_h_dot = np.sqrt(W/(0.5*rho*S))*C_D_opt/(C_L_opt**(1.5))
print(P_maxs)
P_turbs = np.ma.masked_where(Winds_v <= get_local_min_h_dot(Total_wind), P_turbs) # changed to P_maxs
P_turb_max = np.nanmax(P_turbs)
print("max P_turb:", P_turb_max)
#5 * round(P_turb_max/5)+0.1
D_turbs = np.ma.masked_where(Winds_v <= get_local_min_h_dot(Total_wind), D_turb)

colormap_levels = np.arange(0, 5 * round(P_turb_max/5), 0.1)
colormap_levels2 = np.arange(0, 1 * round(np.nanmax(D_turbs)/1), 0.1)





#colormap_levels = 50
#cp = ax[0][0].contourf(xs, zs, P_turbs , colormap_levels,cmap='Reds')#locator=ticker.LogLocator(subs=0.5), cmap='Reds')
#cp = ax.contourf(xs, zs, P_turbs, colormap_levels, cmap='Reds')#locator=ticker.LogLocator(subs=0.5), cmap='Reds')
cp3 = ax.contourf(xs, zs, D_turbs, colormap_levels2, cmap='Reds')

xs_eq = np.ma.masked_where(np.isnan(P_turbs), xs)
zs_eq = np.ma.masked_where(np.isnan(P_turbs), zs)
#ax.plot(xs_eq, zs_eq, '*r')
#cbar = plt.colorbar(cp)
cbar3 = plt.colorbar(cp3)
#cbar.set_label("Max regen power [W]")
cbar3.set_label("Max Turbine Drag [N]")

cbar2 = plt.colorbar(qv1)
cbar2.set_label("Total wind speed [m/s]")
#ax[0][0].set_xlabel('x location [m]')
ax.set_xlabel('x location [m]')
#ax[0][0].set_ylabel('z location [m]')
ax.set_ylabel('z location [m]')
#str(np.around(S_turb_spec*100, 2)) + ' % of S'
ax.set_title('Regen simulation for oval hill section, surf roughness: ' + str(surf_rough) + ' [V_inf = ' + str(U_inf) + ']')
# ax[0][1].plot(x_is, V_as)
# ax[0][1].set_ylabel('Airspeed [m/s]')
# ax[1][0].plot(x_is, W_xs_s)
# ax[1][0].set_ylabel('Wx [m/s]')
# ax[1][0].set_xlabel('x location [m]')
# ax[1][1].plot(x_is, W_zs_s)
# ax[1][1].set_ylabel('Wz [m/s]')
# ax[1][1].set_xlabel('x location [m]')
# ax[0][2].plot(x_is, Ps_turb)
# ax[0][2].set_ylabel('P_turb [W]')
# ax[0][2].set_xlabel('x location [m]')
#ax.plot(oval_pointsx, oval_pointsz, '*b')
ax.set_xlim([0,100])
ax.set_ylim([0,-100])
plt.show()

np.savetxt("drag_vals.csv", D_turbs, delimiter=",")
np.savetxt("total_winds.csv", Total_wind, delimiter=",")
np.savetxt("winds_u.csv", Winds_u, delimiter=",")
np.savetxt("winds_v.csv", Winds_v, delimiter=",")
