
import numpy as np
from interface import *
from acceleration import *

cart = np.array([[24, 0.0,  0.000000,  0.000000],
                 [ 6, 0.0,  0.000000,  2.000000]])

#~ cart = np.array([[ 8, 0.0,  0.000000,  0.110200],
                 #~ [ 1, 0.0,  0.711600, -0.440800],
                 #~ [ 1, 0.0, -0.711600, -0.440800]])

info = {'cart': cart,
        'basis': '6-31g',
        'charge': 0,
        'mult': 1,}
info['dft'] = 'b3lyp'
#~ info['hfExcMix'] = 0.2

intf = G09Interface(info)

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


''' Frank-Wolfe with stepsize 1 '''
#~ dens = guessDensity
#~ for _ in range(100):
    #~ fock, energy = intf.FockEnergy(dens)
    #~ print energy
    #~ dens = SolveFock(fock[0])


''' Frank-Wolfe with stepsize ? '''
#~ stepsize = 0.25
#~ dens = guessDensity
#~ for i in range(100):
    #~ fockTup, energy = intf.FockEnergy(dens)
    #~ fock = fockTup[0]
    #~ densNew = SolveFock(fock)
    #~ dens = (1.0 - stepsize) * dens[0] + stepsize * densNew[0],
    #~ print i, energy


''' Stochastic Frank-Wolfe with stepsize ? '''
#~ nbf = intf.overlap.shape[0]
#~ print nbf
#~ print 'stochastic'
#~ dens = guessDensity
#~ for i in range(200):
    #~ if not i % 20:
        #~ fock, energy = intf.FockEnergy(dens)
        #~ dens = SolveFock(fock[0])
    #~ sel = np.random.choice(range(nbf), size=10, replace=False)
    #~ stoDens = np.zeros(dens[0].shape)
    #~ stoDens[sel, :] = dens[0][sel, :]
    #~ stoDens[:, sel] = dens[0][:, sel]
    #~ fockTup, _ = intf.FockEnergy((stoDens,))
    #~ _, energy = intf.FockEnergy(dens)
    #~ fock = fockTup[0]
    #~ stoFock = fock - intf.oneElecHam
    #~ stoFock += intf.oneElecHam * len(sel) / nbf
    #~ densNew = SolveFock(stoFock)
    #~ wtNew = 0.1
    #~ dens = (1.0 - wtNew) * dens[0] + wtNew * densNew[0],
    #~ print i, energy

#~ print 'non-stochastic'
#~ stepize = 0.25
#~ for i in range(100):
    #~ fockTup, energy = intf.FockEnergy(dens)
    #~ fock = fockTup[0]
    #~ densNew = SolveFock(fock)
    #~ dens = (1.0 - stepize) * dens[0] + stepize * densNew[0],
    #~ print i, energy


''' Accelerated stochastic Frank-Wolfe with stepsize ? '''
#~ (eigVal, eigVec) = np.linalg.eigh(intf.overlap)
#~ keep = eigVal > 1.0e-6
#~ toOrtho = eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]
#~ lciis = LCIIS(intf.overlap, toOrtho)
#~ nbf = intf.overlap.shape[0]
#~ print nbf
#~ print 'stochastic'
#~ dens = guessDensity
#~ for i in range(100):
    #~ sel = np.random.choice(range(nbf), size=5, replace=False)
    #~ stoDens = np.zeros(dens[0].shape)
    #~ stoDens[sel, :] = dens[0][sel, :]
    #~ stoDens[:, sel] = dens[0][:, sel]
    #~ fockTup, _ = intf.FockEnergy((stoDens,))
    #~ _, energy = intf.FockEnergy(dens)
    #~ fock = fockTup[0]
    #~ stoFock = fock - intf.oneElecHam
    #~ stoFock += intf.oneElecHam * len(sel) / nbf
    #~ stoFockNew = lciis.NewFock((stoFock,), (stoDens,))
    #~ densNew = SolveFock(stoFockNew[0])
    #~ wtNew = 0.25
    #~ dens = (1.0 - wtNew) * dens[0] + wtNew * densNew[0],
    #~ print i, energy



''' Accelerated Frank-Wolfe with stepsize ? '''
#~ stepsize = 0.1
#~ from lciis import LCIIS
#~ (eigVal, eigVec) = np.linalg.eigh(intf.overlap)
#~ keep = eigVal > 1.0e-6
#~ toOrtho = eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]
#~ lciis = LCIIS(intf.overlap, toOrtho)
#~ dens = guessDensity
#~ for i in range(100):
    #~ fock, energy = intf.FockEnergy(dens)
    #~ fock = lciis.NewFock(fock, dens)
    #~ print i, energy
    #~ densNew = SolveFock(fock[0])
    #~ dens = (1.0 - stepsize) * dens[0] + stepsize * densNew[0],



''' Accelerated Frank-Wolfe with stepsize ? '''
stepsize = 0.25
cdiis = CDIIS(intf.overlap)
dens = guessDensity
for i in range(100):
    numDrop = 1 if i < 50 else 0
    fock, energy = intf.FockEnergy(dens)
    fock = cdiis.NewFock(fock, dens, numDrop)
    print i, energy
    densNew = SolveFock(fock[0])
    dens = (1.0 - stepsize) * dens[0] + stepsize * densNew[0],



