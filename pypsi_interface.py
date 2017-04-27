"""
An interface needs properties:

    numElecAB
    overlap
    oneElecHam

Needs methods:

    GuessDensity()
    FockEnergy(densTup)

"""

import numpy as np
from PyPsi import PyPsi

# Works in conjunction with PyPsi
class PyPsiInterface(object):

    def __init__(self, info):
        # Construct a PyPsi object
        cart, basis = info['cart'], info['basis']
        charge, mult = info['charge'], info['mult']
        pypsi = PyPsi(cart, basis, charge, mult)
        self._pypsi = pypsi

        # Calculate self.numElecAB from numElecTotal and mult
        numElecTotal = pypsi.Molecule_NumElectrons()
        self.numElecAB = _NumElecAB(numElecTotal, mult)

        # Compute integrals
        self.overlap = pypsi.Integrals_Overlap()
        self.oneElecHam = pypsi.Integrals_Kinetic()
        self.oneElecHam += pypsi.Integrals_Potential()
        self._coreGuessDensity = self._CoreGuess()

    def GuessDensity(self):
        return self._coreGuessDensity

    def _CoreGuess(self):
        # Get transformation to orthogonal MO space 'toOrtho'
        (eigVal, eigVec) = np.linalg.eigh(self.overlap)
        keep = eigVal > 1.0e-6
        toOrtho = eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]

        # Solve oneElecHam to get coreGuessMO
        orFock = toOrtho.T.dot(self.oneElecHam).dot(toOrtho)
        orbEigVal, orOrb = np.linalg.eigh(orFock)
        argsort = np.argsort(orbEigVal)
        coreGuessMO = toOrtho.dot(orOrb[:, argsort])

        # Compute core guess density from
        guessOccMO = (coreGuessMO[:, :ne] for ne in self.numElecAB)
        return tuple(mo.dot(mo.T) for mo in guessOccMO)


def _NumElecAB(numElecTotal, mult):
    numElecA = (numElecTotal + mult - 1) / 2.0
    if numElecA % 1.0 != 0.0:
        raise Exception('numElecTotal and multiplicity do not agree')
    return int(numElecA), int(numElecTotal - numElecA)



