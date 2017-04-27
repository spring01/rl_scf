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

        if 'dft' in info:
            self._dft = True
            pypsi.DFT_Initialize(info['dft'])
            self._hfExcMix = info['hfExcMix'] if 'hfExcMix' in info else 0.0
        else:
            self._dft = False
            self._hfExcMix = 1.0

        # Calculate self.numElecAB from numElecTotal and mult
        numElecTotal = pypsi.Molecule_NumElectrons()
        self.numElecAB = _NumElecAB(numElecTotal, mult)

        # Nuclei Coulomb repulsive energy
        self._nucRepEnergy = pypsi.Molecule_NucRepEnergy()

        # Compute integrals
        self.overlap = pypsi.Integrals_Overlap()
        self.oneElecHam = pypsi.Integrals_Kinetic()
        self.oneElecHam += pypsi.Integrals_Potential()
        self._coreGuessDensity = self._CoreGuess()

    def GuessDensity(self):
        return self._coreGuessDensity

    def FockEnergy(self, densTup):
        # A new list has to be constructed as it will be altered in PyPsi
        self._pypsi.JK_CalcAllFromDens(list(densTup))
        dens = list(densTup)
        cou = self._pypsi.JK_GetJ()
        coreCou = self.oneElecHam + sum(cou) * (2.0 / len(cou))
        if self._hfExcMix:
            exc = self._pypsi.JK_GetK()
            fock = [coreCou - self._hfExcMix * x for x in exc]
        else:
            fock = [coreCou for _ in dens]

        energy = 0.0
        for f, d in zip(fock, dens):
            energy += (self.oneElecHam + f).ravel().dot(d.ravel())
        energy /= len(dens)
        energy += self._nucRepEnergy

        if self._dft:
            dftV = self._pypsi.DFT_DensToV(dens)
            fock = [f + v for v in zip(fock, dftV)]
            energy += self._pypsi.DFT_EnergyXC()

        return tuple(fock), energy

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



