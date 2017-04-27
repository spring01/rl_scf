
import numpy as np
from g09interface import G09Interface

cart = np.array([[8, 0.0,  0.000000,  0.110200],
                 [1, 0.0,  0.711600, -0.440800],
                 [1, 0.0, -0.711600, -0.440800]])

info = {'cart': cart,
        'basis': 'sto-3g',
        'charge': 0,
        'mult': 3}

intf = G09Interface(info)

np.set_printoptions(precision=3, linewidth=100)
guessDensity = intf.GuessDensity()
print guessDensity
print intf.FockEnergy(guessDensity)

