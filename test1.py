
import numpy as np
from g09interface import G09Interface

cart = np.array([[8, 0.0,  0.000000,  0.110200],
                 [1, 0.0,  0.711600, -0.440800],
                 [1, 0.0, -0.711600, -0.440800]])

info = {'cart': cart,
        'basis': '6-31g',
        'charge': 0,
        'multiplicity': 1}

intf = G09Interface(info)

guessDensity = intf.GuessDensity()
print intf.GuessDensity()
print intf.FockEnergy(guessDensity)
