
import numpy as np
from collections import deque

class CDIIS(object):

    def __init__(self, overlap, verbose=False, maxNumVec=20):
        self._overlap = overlap
        self._fockList = deque(maxlen=maxNumVec)
        self._commList = deque(maxlen=maxNumVec)
        self._hess = np.zeros((maxNumVec, maxNumVec))

    # Enqueue Fock list and density list, then extrapolate Fock
    # Note: this Fock list should be built from this density list
    #   fockTup: (fock,) for rhf/rks; (fockA, fockB) for uhf/uks
    #   densTup: (dens,) for rhf/rks; (densA, densB) for uhf/uks
    #   return: (newFock,) for rhf/rks; (newFockA, newFockB) for uhf/uks
    def NewFock(self, fockTup, densTup, numDrop=0):
        # enqueue the most recent Fock matrix and the corresponding commutator
        self._fockList.append(fockTup)
        numUse = len(self._fockList)
        newComm = self._CommWithSpin(fockTup, densTup)
        self._commList.append(newComm)
        numUse -= numDrop
        if numUse <= 1:
            return fockTup

        # update the CDIIS Hessian
        newHess = np.array(self._commList)[-numUse:].dot(newComm)
        self._hess[:, :-1] = self._hess[:, 1:]
        self._hess[:-1, :] = self._hess[1:, :]
        self._hess[-numUse:, -1] = newHess
        self._hess[-1, -numUse:] = newHess

        # compute CDIIS extrapolation coefficients
        ones = np.ones((numUse, 1))
        hessUse = self._hess[:, -numUse:]
        hessUse = hessUse[-numUse:, :]
        hessLag = np.bmat([[hessUse,   ones   ],
                           [ones.T,    [[0.0]]]])
        gradLag = np.concatenate((np.zeros(numUse), [1.0]))
        coeffUse = np.linalg.solve(hessLag, gradLag)[:-1]

        # compute the extrapolated Fock matrix
        newFockList = []
        for s in range(len(fockTup)):
            fockVecs = np.array([ft[s].ravel() for ft in self._fockList])
            newFockVec = coeffUse.dot(fockVecs[-numUse:])
            newFockList.append(newFockVec.reshape(fockTup[0].shape))
        return tuple(newFockList)


    def _CommWithSpin(self, fockTup, densTup):
        return np.concatenate([self._Comm(*fd) for fd in zip(fockTup, densTup)])

    def _Comm(self, fock, dens):
        fds = fock.dot(dens).dot(self._overlap)
        comm = fds - fds.T
        return comm[np.triu_indices_from(comm, 1)]



