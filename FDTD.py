import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
# think of elements in the E_z and H_y arrays as being offset from each other by half a spatial step
# but they will be accessed using integer index


# parameters
size = 400
# electric permittivity and magnetic permeability of free space
eps_vac = const.epsilon_0
mu_vac = const.mu_0
eta = np.sqrt(mu_vac/eps_vac)
c = const.c # speed of light
E_z = np.zeros(size)
H_y = np.zeros(size)
J_z = np.zeros(size)


# change Courant number
dx = 10e-9
dt = dx / (2*c)
S_c = c * (dt/dx) # Courant number
q_time = 0
tot_time = 20000
mm = 0
mn = mm+1


# Drude material (silver) parameters
thin_film_thickness = 200e-9 # 200 nm
j_Ag_start = int(size / 2)
j_Ag_end = int(thin_film_thickness/dx + j_Ag_start - 1)
omega_p = 1.38e16 # plasma frequency rad/s
gamma = 2.73e13  # damping constant rad/s
eps_inf = 3.7 # high frequency limit


# ADE constants
C_jj = (1 - ((gamma * dt) / 2)) / (1 + ((gamma * dt) / 2))
C_je = (eps_vac * omega_p**2 * dt) / (1 + (gamma * dt) / 2) * (1/2)


# coefficients for electric field update in Drude region
denom = (eps_inf * eps_vac / dt) + (C_je / 2)
E_coeff_1 = ((eps_inf * eps_vac / dt) - C_je / 2) / denom
E_coeff_2 = 1 / denom


# set up wavelengths and recording arrays
n_freq = 1000
wavelengths = np.linspace(300e-9, 800e-9, n_freq)
omegas = 2 * np.pi * c / wavelengths


E_inc  = np.zeros(n_freq, dtype=complex)
E_refl = np.zeros(n_freq, dtype=complex)
E_tran = np.zeros(n_freq, dtype=complex)




plt.ion() # interactive


fig, ax = plt.subplots()
line, = ax.plot(E_z)
ax.set_ylim(-1.5, 1.5)
ax.axvspan(j_Ag_start, j_Ag_end, alpha=0.3, color='gray', label='Ag slab')
ax.legend()


# close event to stop simulation
running = True
def on_close(event):
   global running
   running = False
fig.canvas.mpl_connect('close_event', on_close)


# pulse parameters
t_0 = 200 # pulse time delay
width = 1000
pulse = np.exp(-(q_time - t_0)**2 / width)


# do time stepping
for q_time in range(tot_time):
   if not running:
       break


   # save boundary values before updates
   E_z_left_prev = E_z[1]
   E_z_right_prev = E_z[size-2]


   # update magnetic field
   for mm in range(size-1):
       H_y[mm] += (dt / (mu_vac*dx)) * (E_z[mm+1] - E_z[mm])


   # correction for H_y adjacent to TFSF interface
   H_y[49] -= (dt / (mu_vac*dx)) * np.exp(-(q_time - t_0)**2 / width)


   # copy electric field to temporary variable (needed for J update)
   E_z_temp = E_z.copy()


   # update electric field
   for mn in range(1, size):
       if j_Ag_start <= mn < j_Ag_end:
           curl_H_y = (H_y[mn] - H_y[mn-1]) / dx
           E_z[mn] = (E_coeff_1 * E_z[mn]) + (E_coeff_2 * (curl_H_y - 0.5 * (1 + C_jj) * J_z[mn]))
           J_z[mn] = C_jj * J_z[mn] + C_je * (E_z[mn] + E_z_temp[mn])
       else:
           E_z[mn] += (dt / (eps_vac*dx)) * (H_y[mn] - H_y[mn-1])




   # correction for E_z adjacent to the TFSF interface
   E_z[50] += (dt / (eps_vac*dx)) * np.exp(-(q_time + 0.5 - (-0.5) - t_0)**2 / width) / eta


   # ABC
   E_z[0] = E_z_left_prev  + (c*dt - dx) / (c*dt + dx) * (E_z[1] - E_z_left_prev)
   E_z[size-1] = E_z_right_prev + (c*dt - dx) / (c*dt + dx) * (E_z[size-2] - E_z_right_prev)
   # Fourier transform electric fields
   pulse = np.exp(-(q_time - t_0)**2 / width)
   E_inc  += pulse * np.exp(1j * omegas * q_time * dt)
   E_refl += E_z[25] * np.exp(1j * omegas * q_time * dt)
   E_tran += E_z[350] * np.exp(1j * omegas * q_time * dt)
   # update plot
   line.set_ydata(E_z)
   ax.set_title(f"Time step: {q_time}")
   plt.pause(0.01)


R = np.abs(E_refl / E_inc)**2 * 100
fig2, ax2 = plt.subplots()
ax2.plot(wavelengths*1e9, R)
# ax2.plot(wavelengths, T, label='Transmittance')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Reflectance (%)')
# ax2.legend()


plt.ioff()
plt.show()
