# %% IMPORTS
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import functions as fn

# IMPORT DATA

RD = 52 #m
HH = 44 #m
co_WS = 25 #m/s

#data_path = os.path.join(os.path.dirname(__file__), 'data', '2023-Jan-July-Power-data_AS03.csv')
#data = pd.read_csv(data_path, index_col=0, parse_dates=True)
# QUESTION 1 (226 to 4)

#Distance between WT and WME
L = 124 #m

#Maximum heights for each radius sector to be considered no obstacle 
h_2L = 1/3*(HH-0.5*RD) #m
h_2L4L = 2/3*(HH-0.5*RD) #m
h_4L8L = HH-0.5*RD #m
h_8L16L = 4/3*(HH-0.5*RD) #m

# %% kurry

Ih_RB = 10 #m
Iw_RB = 0.22*1000 #m
Le_RB = 0.06*1000 #m

VS_i_RB = 351 # deg Visual segments
VS_ii_RB = 68 # deg Visual segments
VS_m_RB = VS_ii_RB - (360-VS_i_RB+VS_ii_RB)/2 # deg Middle of the segment

De_RB = fn.ERD(Ih_RB, Iw_RB)
al_RB = fn.alpha(De_RB, Le_RB)

RD_RB = Le_RB/De_RB #m relative distance
print(f'Risø buildings; alpha: {al_RB}, relative distance:{RD_RB}')
print(f'Risø buildings; remove {al_RB} degrees, centerd on {VS_m_RB}, then from {VS_m_RB - al_RB/2} to {VS_m_RB + al_RB/2}')
print('OUT')

#TURBINE

# %% 2L

h_2L = 1/3*(HH-0.5*RD) #m

#Risø buildings
Ih_RB = 10 #m
Iw_RB = 200 #m
Le_RB = 80 #m

VS_i_RB = 351 # deg Visual segments
VS_ii_RB = 68 # deg Visual segments
VS_m_RB = VS_ii_RB - (360-VS_i_RB+VS_ii_RB)/2 # deg Middle of the segment

De_RB = fn.ERD(Ih_RB, Iw_RB)
al_RB = fn.alpha(De_RB, Le_RB)

RD_RB = Le_RB/De_RB #m relative distance
print(f'Risø buildings nad trees; alpha: {al_RB}, relative distance:{RD_RB}')
print(f'Risø buildings nad trees; remove {al_RB} degrees, centerd on {VS_m_RB}, then from {VS_m_RB - al_RB/2} to {VS_m_RB + al_RB/2}')

#Trees
Ih_T = 12 #m
Iw_T = 350 #m
Le_T = 100 #m

VS_i_T = 50 # deg Visual segments
VS_ii_T = 155 # deg Visual segments
VS_m_T = VS_i_T + (VS_ii_T - VS_i_T)/2 # deg Middle of the segment

De_T = fn.ERD(Ih_T, Iw_T)
al_T = fn.alpha(De_T, Le_T)

RD_RB = Le_T/De_T #m relative distance
print(f'Trees nad trees; alpha: {al_T}, relative distance:{RD_T}')
print(f'Trees nad trees; remove {al_T} degrees, centerd on {VS_m_T}, then from {VS_m_T - al_T/2} to {VS_m_T + al_T/2}')



#Turbine V27
Ih_V27 = (31.5 + 27/2) #m
Iw_V27 = 27 #m
Le_V27 = 90 #m

VS_m_V27 = 197 # deg Middle of the segment

De_V27 = 27 
al_V27 = fn.alpha(De_V27, Le_V27)

RD_V27 = Le_V27/De_V27 #m relative distance
print(f'Turbine V27; alpha: {al_V27}, relative distance:{RD_V27}')
print(f'Turbine V27; remove {al_V27} degrees, centerd on {VS_m_V27}, then from {VS_m_V27 - al_V27/2} to {VS_m_V27 + al_V27/2}')



# %% 4L

h_2L4L = 2/3*(HH-0.5*RD) #m

#House and trees
Ih_HT = 12 #m
Iw_HT = 130 #m
Le_HT = 230 #m

VS_i_HT = 154 # deg Visual segments
VS_ii_HT = 188 # deg Visual segments
VS_m_HT = VS_i_HT + (VS_ii_HT - VS_i_HT)/2 # deg Middle of the segment

De_HT = fn.ERD(Ih_HT, Iw_HT)
al_HT = fn.alpha(De_HT, Le_HT)

RD_HT = Le_HT/De_HT #m relative distance
print(f'Houese and trees; alpha: {al_HT}, relative distance:{RD_HT}')
print(f'Houese and trees; remove {al_HT} degrees, centerd on {VS_m_HT}, then from {VS_m_HT - al_HT/2} to {VS_m_V27 + al_HT/2}')


#MAST

# %%
