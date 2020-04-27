import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from matplotlib import ticker, cm

R_ridge = 10.0 # m
U_inf = -10.0 # m/s
a = 10 # m, defines loci x-position
x_stag = 14 # m, defines a-axis of standard oval
b = np.sqrt(x_stag**2 - a**2)
flow_type = 'oval'


# def find_h_oval():
#     m = np.pi * U_inf / a * (x_stag ** 2 - a ** 2)
#     f = lambda x : 4*np.pi**2 * np.tan(x)*U_inf**2*a**2 + 4*np.pi*U_inf*a*m*x - np.tan(x)*x**2*m**2
#     h_int = np.array([np.pi/8, -np.pi/8, np.pi, -np.pi, np.pi*10])
#     plt.plot(np.arange(0.0, 2*np.pi, 0.001), f(np.arange(0.0, 2*np.pi, 0.001)))
#     plt.show()
#     solutions = fsolve(f, h_int)
#     pos_hs = (solutions*m)/(2*np.pi*U_inf)
#     print(pos_hs)


def wind_comp(x_i, z_i, U):


    if flow_type == 'oval':

        m = np.pi * U_inf / a * (x_stag ** 2 - a **2)
        v_x = U_inf + m / (2 * np.pi) * ((x_i + a) / ((x_i+0.001 + a) ** 2 + z_i ** 2) - (x_i+0.001 - a) / ((x_i+0.001 - a)**2 + z_i**2))
        v_z = -(m * z_i) / (2 * np.pi) * (1 / ((x_i+0.001 + a)**2 + z_i**2) - 1 / ((x_i+0.001 - a)**2 + z_i**2))

        ellipse_eq = (x_i ** 2) / (x_stag ** 2) + (z_i ** 2) / (b ** 2)
        if ellipse_eq < 1:
            return 0,0

        if x_i < x_stag:

            z_ellipse = -b / x_stag * np.sqrt(x_stag ** 2 - x_i ** 2)


            #v_x *= ((z_i - z_ellipse)/-20)**(1/7)
            #v_z *= ((z_i - z_ellipse) / -20) ** (1 / 7)

            mult_factor = (np.log(-(z_i - z_ellipse))/0.05)/(np.log(-(-20 - z_ellipse))/0.05)
            print("z_ab =", (z_i - z_ellipse), "mult =", mult_factor)
            if not(np.isnan(mult_factor)):

                v_x *= mult_factor
                v_z *= mult_factor

        else:
            if z_i < 0:
                z_ellipse = -0.01
                mult_factor = (np.log(-(z_i - z_ellipse)) / 0.05) / (np.log(-(-20 - z_ellipse)) / 0.05)
                v_x *= mult_factor
                v_z *= mult_factor
            else:
                return 0,0
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
        mult_factor = (np.log(-(z_i - z_circle)) / 0.05) / (np.log(-(-20 - z_circle)) / 0.05)
        u_x = np.cos(theta)*u_r - np.sin(theta)*u_th*mult_factor
        u_z = (np.sin(theta)*u_r + np.cos(theta)*u_th)*mult_factor


        if r<R_ridge:
            return 0,0

        return u_x, u_z

vwind_comp = np.vectorize(wind_comp)

def calc_jacobian(i, j, dx, dz, u_xs, u_zs):

    J = np.array([[(u_xs[i+1, j]-u_xs[i-1, j])/2*dx, (u_xs[i, j+1]-u_xs[i, j-1])/2*dz],
                  [(u_zs[i+1, j]-u_zs[i-1, j])/2*dx, (u_zs[i, j+1]-u_zs[i, j-1])/2*dz]])
    return J

def x_to_i(x_coor, max_i, x_coords):
    i = int(np.around(x_coor*(1/dx)) + x_coords[-1])
    if i>max_i:
        return max_i-2
    elif i < 0:
        return 0
    else:
        return i

def z_to_j(z_coor, max_j, z_coords):
    j = int(np.around(z_coor*(1/dz)) + z_coords[-1])
    if j>max_j:
        return max_j-2
    elif j < 0:
        return 0
    else:
        return j



x_vels = np.array([])
z_vels = np.array([])

dx = 0.5
dz = -0.5
x_coords = np.arange(-40, 40, dx)
z_coords = np.arange(0, -40, dz)

xs, zs = np.meshgrid(x_coords, z_coords)

Winds_u, Winds_v = vwind_comp(xs, zs, U_inf)

Total_wind = np.sqrt(Winds_u**2 + Winds_v**2)


W_xs = np.zeros((len(x_coords), len(z_coords)))
W_zs = np.zeros((len(x_coords), len(z_coords)))

jacobians = np.zeros((len(x_coords), len(z_coords), 2,2))
i = 0

max_V = 0.0
max_coords = (0,0)

for x in x_coords:
    j = 0
    for z in z_coords:
        r = np.sqrt(x**2 + z**2)
        if r > R_ridge:
            u_x, u_z = wind_comp(x, z, U_inf)
            W_xs[i, j] = u_x
            W_zs[i, j] = u_z
            V = np.sqrt(u_x**2 + u_z**2)
            if u_z > max_V and r > (1.5*R_ridge):
                max_V = u_z
                max_coords = (x, z)

        j += 1
    i += 1

for i in range(1, len(x_coords)-1):
    for j in range(1, len(z_coords)-1):
        jacobians[i, j] = calc_jacobian(i, j, dx, dz, W_xs, W_zs)



if flow_type == 'circle':
    thetas = np.arange(0, 180, 0.01)
    xs_hill = R_ridge*np.cos(thetas)
    zs_hill = -R_ridge*np.sin(thetas)

else:
    ts = np.arange(0, 2 * np.pi, 0.01)
    u = np.tan(ts / 2)
    xs_hill = x_stag * (1 - u ** 2) / (u ** 2 + 1)
    zs_hill = -(2 * b * u) / (u ** 2 + 1)


fig, ax = plt.subplots(2,3)
ax[0][0].invert_yaxis()
ax[0][0].quiver(xs, zs, Winds_u, Winds_v)
ax[0][0].streamplot(xs, zs, Winds_u, -Winds_v, density=1)




ax[0][0].plot(xs_hill, zs_hill, '-r')

#plt.show()


## initial conditions
m = 1.75 # kg

S = 1 # m^2
rho = 1.225 # kg/m^3
g = 9.80665 # m/s^2
CL_alpha = 5.7 # 1/rad
alpha_0L = np.deg2rad(-4)
AR = 3
e = 0.8
CD_0 = 0.05
W = m*g
V_a = 25 #m/s
gamma_a = -10*(np.pi/180) # deg/rad
dgamma_dt = 0 # rad/s
dt = 0.01 # s
t = 0 #s
x_i = 7
z_i = -15
D_prop = 0.3 # m

Cdi_coef = 1/(np.pi*AR*e) # 1/pi*A*e

P_maxs = (16/27)*(rho/2)* Total_wind**3 * (np.pi*D_prop**2)/4


def calc_eq():
    C_L_req = W/(0.5*rho*np.power(Total_wind, 2)*S) * (np.abs(Winds_u)/Total_wind)
    C_D_req = W / (0.5 * rho * np.power(Total_wind, 2) * S) * (np.abs(Winds_v) / Total_wind)

    C_D_min_ach = CD_0+np.power(C_L_req, 2)*Cdi_coef
    C_D_max_ach = C_D_min_ach + 1*(2/9)*0.1

    alpha = C_L_req / CL_alpha + alpha_0L

    stall = (alpha > np.deg2rad(15)) & (alpha < np.deg2rad(-10))
    C_L_req[stall] = np.nan
    C_D_req[stall] = np.nan
    C_D_min_ach[stall] = np.nan
    C_D_max_ach[stall] = np.nan

    eq_points = (C_D_req > C_D_min_ach) & (C_D_req < C_D_max_ach)

    return eq_points

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

eq_positions = calc_eq()

while t < 0.1:

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

ax[0][0].plot(x_is, z_is)
min_h_dot = np.sqrt(W/(0.5*rho*S))*C_D_opt/(C_L_opt**(1.5))
print(P_maxs)
P_maxs = np.ma.masked_where(Winds_v <= get_local_min_h_dot(Total_wind), P_maxs)
P_maxs = np.ma.masked_where(eq_positions == False, P_maxs)


cp = ax[0][0].contourf(xs, zs, P_maxs , 50,cmap='Reds')#locator=ticker.LogLocator(subs=0.5), cmap='Reds')
fig.colorbar(cp)
ax[0][0].set_xlabel('x location [m]')
ax[0][0].set_ylabel('z location [m]')
ax[0][1].plot(x_is, V_as)
ax[0][1].set_ylabel('Airspeed [m/s]')
ax[1][0].plot(x_is, W_xs_s)
ax[1][0].set_ylabel('Wx [m/s]')
ax[1][0].set_xlabel('x location [m]')
ax[1][1].plot(x_is, W_zs_s)
ax[1][1].set_ylabel('Wz [m/s]')
ax[1][1].set_xlabel('x location [m]')
ax[0][2].plot(x_is, Ps_turb)
ax[0][2].set_ylabel('P_turb [W]')
ax[0][2].set_xlabel('x location [m]')
plt.show()


