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
print(f'Risø buildings; alpha: {al_RB}, relative distance:{RD_RB}')
print(f'Risø buildings; remove {al_RB} degrees, centerd on {VS_m_RB}, then from {VS_m_RB - al_RB/2} to {VS_m_RB + al_RB/2}')

#Trees
Ih_T = 12 #m
Iw_T = 350 #m
Le_T = 100 #m

VS_i_T = 50 # deg Visual segments
VS_ii_T = 155 # deg Visual segments
VS_m_T = VS_i_T + (VS_ii_T - VS_i_T)/2 # deg Middle of the segment

De_T = fn.ERD(Ih_T, Iw_T)
al_T = fn.alpha(De_T, Le_T)

RD_T = Le_T/De_T #m relative distance
print(f'Trees; alpha: {al_T}, relative distance:{RD_T}')
print(f'Trees; remove {al_T} degrees, centerd on {VS_m_T}, then from {VS_m_T - al_T/2} to {VS_m_T + al_T/2}')



#Turbine V272
Ih_V27 = (31.5 + 27/2) #m
Iw_V27 = 27 #m
Le_V27 = 90 #m

VS_m_V27 = 197 # deg Middle of the segment

De_V27 = 27 
al_V27 = fn.alpha(De_V27, Le_V27)

RD_V27 = Le_V27/De_V27 #m relative distance
print(f'Turbine V27; alpha: {al_V27}, relative distance:{RD_V27}')
print(f'Turbine V27; remove {al_V27} degrees, centerd on {VS_m_V27}, then from {VS_m_V27 - al_V27/2} to {VS_m_V27 + al_V27/2}')



# 4L

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
print(f'Houese and trees; remove {al_HT} degrees, centerd on {VS_m_HT}, then from {VS_m_HT - al_HT/2} to {VS_m_HT + al_HT/2}')
print(f'WT FROM {VS_m_RB - al_RB/2} TO {VS_m_V27 + al_V27/2}')

fn.plot_obstructed_sectors([(VS_m_RB - al_RB/2, VS_m_RB + al_RB/2), (VS_m_T - al_T/2, VS_m_T + al_T/2), 
                            (VS_m_V27 - al_V27/2, VS_m_V27 + al_V27/2), (VS_m_HT - al_HT/2, VS_m_HT + al_HT/2)], ['Risø buildings', 'Trees', 'Turbine V27', 'House and trees'], 'Turbine V52')

#MAST

# %% 2L

#Risø buildings
Ih_RB2 = 10 #m
Iw_RB2 = 160 #m
Le_RB2 = 170 #m

VS_i_RB2 = 41 # deg Visual segments
VS_ii_RB2 = 92 # deg Visual segments
VS_m_RB2 = VS_i_RB2 + (VS_ii_RB2 - VS_i_RB2)/2 # deg Middle of the segment

De_RB2 = fn.ERD(Ih_RB2, Iw_RB2)
al_RB2 = fn.alpha(De_RB2, Le_RB2)

RD_RB2 = Le_RB2/De_RB2 #m relative distance
print(f'Risø buildings; alpha: {al_RB2}, relative distance:{RD_RB2}')
print(f'Risø buildings; remove {al_RB2} degrees, centerd on {VS_m_RB2}, then from {VS_m_RB2 - al_RB2/2} to {VS_m_RB2 + al_RB2/2}')


# V52
Ih_V522 = 70 #m
Iw_V522 = 52 #m
Le_V522 = 124 #m

VS_i_V522 = 100 # deg Visual segments
VS_ii_V522 = 126 # deg Visual segments
VS_m_V522 = VS_i_V522 + (VS_ii_V522 - VS_i_V522)/2 # deg Middle of the segment

De_V522 = fn.ERD(Ih_V522, Iw_V522)
al_V522 = fn.alpha(De_V522, Le_V522)

RD_V522 = Le_V522/De_V522 #m relative distance
print(f'V52; alpha: {al_V522}, relative distance:{RD_V522}')
print(f'V52; remove {al_V522} degrees, centerd on {VS_m_V522}, then from {VS_m_V522 - al_V522/2} to {VS_m_V522 + al_V522/2}')



# V27
Ih_V272 = (31.5 + 27/2) #m
Iw_V272 = 27 #m
Le_V272 = 160 #m

VS_i_V272 = 140 # deg Visual segments
VS_ii_V272 = 151 # deg Visual segments
VS_m_V272 = VS_i_V272 + (VS_ii_V272 - VS_i_V272)/2 # deg Middle of the segment

De_V272 = fn.ERD(Ih_V272, Iw_V272)
al_V272 = fn.alpha(De_V272, Le_V272)

RD_V272 = Le_V272/De_V272 #m relative distance
print(f'V27; alpha: {al_V272}, relative distance:{RD_V272}')
print(f'V27; remove {al_V272} degrees, centerd on {VS_m_V272}, then from {VS_m_V272 - al_V272/2} to {VS_m_V272 + al_V272/2}')

# 4L

# Trees and house

Ih_TH2 = 12 #m
Iw_TH2 = 60 #m
Le_TH2 = 300 #m

VS_i_TH2 = 90 # deg Visual segments
VS_ii_TH2 = 100 # deg Visual segments
VS_m_TH2 = VS_i_TH2 + (VS_ii_TH2 - VS_i_TH2)/2 # deg Middle of the segment

De_TH2 = fn.ERD(Ih_TH2, Iw_TH2)
al_TH2 = fn.alpha(De_TH2, Le_TH2)

RD_TH2 = Le_TH2/De_TH2 #m relative distance
print(f'Trees and house; alpha: {al_TH2}, relative distance:{RD_TH2}')
print(f'Trees and house; remove {al_TH2} degrees, centerd on {VS_m_TH2}, then from {VS_m_TH2 - al_TH2/2} to {VS_m_TH2 + al_TH2/2}')


# Trees

#Trees
Ih_T2 = 12 #m
Iw_T2 = 100 #m
Le_T2 = 340 #m

VS_i_T2 = 126 # deg Visual segments
VS_ii_T2 = 140 # deg Visual segments
VS_m_T2 = VS_i_T2 + (VS_ii_T2 - VS_i_T2)/2 # deg Middle of the segment

De_T2 = fn.ERD(Ih_T2, Iw_T2)
al_T2 = fn.alpha(De_T2, Le_T2)

RD_T2 = Le_T2/De_T2 #m relative distance
print(f'Trees; alpha: {al_T2}, relative distance:{RD_T2}')
print(f'Trees; remove {al_T2} degrees, centerd on {VS_m_T2}, then from {VS_m_T2 - al_T2/2} to {VS_m_T2 + al_T2/2}')
print(f'WME FROM {VS_m_RB2 - al_RB2/2} TO {VS_m_V272 + al_V272/2}')

fn.plot_obstructed_sectors([(VS_m_RB2 - al_RB2/2, VS_m_RB2 + al_RB2/2), (VS_m_V522 - al_V522/2, VS_m_V522 + al_V522/2), 
                            (VS_m_V272 - al_V272/2, VS_m_V272 + al_V272/2), (VS_m_TH2 - al_TH2/2, VS_m_TH2 + al_TH2/2),
                            (VS_m_T2 - al_T2/2, VS_m_T2 + al_T2/2)],
                            ['Risø buildings', 'Turbine V52', 'Turbine V27', 'House and trees', 'Trees'], 'Met Mast')



