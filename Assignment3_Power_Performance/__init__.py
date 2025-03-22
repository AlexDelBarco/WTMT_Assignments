import numpy as np


def ERD(Ih, Iw):

    De = (2*Ih*Iw)/(Ih+Iw)

    return De


def alpha():

    al = 1.3*np.arctan(2.5*((De)/(Le)+0.15)+10)

    return al
