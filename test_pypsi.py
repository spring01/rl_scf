
import numpy as np
from pypsi_interface import PyPsiInterface

cart = np.array([[8, 0.0,  0.000000,  0.110200],
                 [1, 0.0,  0.711600, -0.440800],
                 [1, 0.0, -0.711600, -0.440800]])

info = {'cart': cart,
        'basis': '6-31g',
        'charge': 1,
        'mult': 4}

intf = PyPsiInterface(info)

print intf.GuessDensity()
#~ print intf.GuessOccMO()
#~ print intf.FockEnergy(guessDensity)
