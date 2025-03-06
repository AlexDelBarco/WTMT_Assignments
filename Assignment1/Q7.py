#%% IMPORTS

import numpy as np

#%% CONSTANTS

WS = [4, 8, 12]
u_cal1 = 0.06
u_cal2 = 0.01/np.sqrt(3)
k_c = 0.8
u_mount = 0.01

#%% FUNCTIONS

def compute_Uv(k_c, V, u_cal1, u_cal2, u_mount):

    u_ope = k_c/(100) * 1/(np.sqrt(3)) * (0.5*V+5)

    u_cal = np.sqrt(u_cal1**2 + u_cal2**2)

    U_v = np.sqrt(u_cal**2 + u_mount**2 + u_ope**2)

    return U_v

#%% MAIN



results = {}

for W in WS:
    U_v = compute_Uv(k_c, W, u_cal1, u_cal2, u_mount)
    results[W] = U_v

