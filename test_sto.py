
import numpy as np
from interface import PyPsiInterface

#~ cart = np.array([[24, 0.0,  0.000000,  0.000000],
                 #~ [ 6, 0.0,  0.000000,  2.000000]])

cart = np.array([[ 8, 0.0,  0.000000,  0.110200],
                 [ 1, 0.0,  0.711600, -0.440800],
                 [ 1, 0.0, -0.711600, -0.440800]])

info = {'cart': cart,
        'basis': '6-31g*',
        'charge': 0,
        'mult': 1,}
info['dft'] = 'b3lyp'
info['hfExcMix'] = 0.2

intf = PyPsiInterface(info)

#~ # Harris guess
#~ def _Guess(fock, numElecAB, overlap):
    #~ # Get transformation to orthogonal MO space 'toOrtho'
    #~ (eigVal, eigVec) = np.linalg.eigh(overlap)
    #~ keep = eigVal > 1.0e-6
    #~ toOrtho = eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]

    #~ # Solve oneElecHam to get coreGuessMO
    #~ orFock = toOrtho.T.dot(fock).dot(toOrtho)
    #~ orbEigVal, orOrb = np.linalg.eigh(orFock)
    #~ argsort = np.argsort(orbEigVal)
    #~ coreGuessMO = toOrtho.dot(orOrb[:, argsort])

    #~ # Compute core guess density
    #~ if numElecAB[0] == numElecAB[1]:
        #~ numElecTup = numElecAB[0:1]
    #~ else:
        #~ numElecTup = numElecAB
    #~ guessOccMO = (coreGuessMO[:, :ne] for ne in numElecTup)
    #~ return tuple(mo.dot(mo.T) for mo in guessOccMO)
#~ intf._pypsi.SCF_SetGuessType('SAD')
#~ sadDensity = intf._pypsi.SCF_GuessDensity()
#~ harris = intf.FockEnergy((sadDensity,))[0][0]
#~ guessDensity = _Guess(harris, intf.numElecAB, intf.overlap)




np.set_printoptions(precision=3, linewidth=100)
guessDensity = intf.GuessDensity()
#~ import cPickle as pickle
#~ with open('crc_harris.p', 'rb') as pic:
    #~ guessDensity = pickle.load(pic)


def SolveFock(fock):
    # Get transformation to orthogonal MO space 'toOrtho'
    (eigVal, eigVec) = np.linalg.eigh(intf.overlap)
    keep = eigVal > 1.0e-6
    toOrtho = eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]

    # Solve oneElecHam to get coreGuessMO
    orFock = toOrtho.T.dot(fock).dot(toOrtho)
    orbEigVal, orOrb = np.linalg.eigh(orFock)
    argsort = np.argsort(orbEigVal)
    coreGuessMO = toOrtho.dot(orOrb[:, argsort])

    # Compute core guess density
    if intf.numElecAB[0] == intf.numElecAB[1]:
        numElecTup = intf.numElecAB[0:1]
    else:
        numElecTup = intf.numElecAB
    guessOccMO = (coreGuessMO[:, :ne] for ne in numElecTup)
    return tuple(mo.dot(mo.T) for mo in guessOccMO)

#~ dens = guessDensity
#~ for _ in range(100):
    #~ fock, energy = intf.FockEnergy(dens)
    #~ print energy
    #~ dens = SolveFock(fock[0])


#~ dens = guessDensity
#~ for i in range(100):
    #~ fock, energy = intf.FockEnergy(dens)
    #~ print i, energy
    #~ densNew = SolveFock(fock[0])
    #~ wtNew = 0.25
    #~ dens = (1.0 - wtNew) * dens[0] + wtNew * densNew[0],

from lciis import LCIIS
(eigVal, eigVec) = np.linalg.eigh(intf.overlap)
keep = eigVal > 1.0e-6
toOrtho = eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]
lciis = LCIIS(intf.overlap, toOrtho)
dens = guessDensity
for _ in range(100):
    fock, energy = intf.FockEnergy(dens)
    fock = lciis.NewFock(fock, dens)
    print energy
    dens = SolveFock(fock[0])

