
import numpy as np
from collections import deque

class LCIIS(object):

    def _init_(self, overlap, toOr, verbose=False, maxNumFock=20):
        self._verbose = verbose
        self._fList = deque(maxlen=maxNumFock)
        self._trFList = deque(maxlen=maxNumFock)
        self._trDList = deque(maxlen=maxNumFock)
        self._toOr = toOr
        self._orTo = overlap.dot(toOr)
        # self._comm[i * numFock + j] = commutator [Fi, Dj]
        self._comm = [None] * maxNumFock**2
        # self._bigMat[i * numFock + j, k * numFock + l] = T(i, j, k, l)
        self._bigMat = np.zeros((maxNumFock**2, maxNumFock**2))
        self._maxIterNewton = 200
        self._gradNormThres = 1e-12

    # Enqueue Fock list and density list, then extrapolate Fock
    # Note: this Fock list should be built from this density list
    #   fockTup: (fock,) for rhf/rks; (fockA, fockB) for uhf/uks
    #   densTup: (dens,) for rhf/rks; (densA, densB) for uhf/uks
    #   return: [newFock] for rhf/rks; [newFockA, newFockB] for uhf/uks
    def NewFock(self, fockTup, densTup):
        # enqueue fock & density and update commutators & tensor
        if len(self._fList) == self._fList.maxlen:
            self._PreUpdateFull()
        else:
            self._PreUpdateNotFull()
        self._fList.append(fockTup)
        shape = (self._toOr.shape[0], -1)
        trFList = [self._toOr.T.dot(fock.reshape(shape)) for fock in fockTup]
        trDList = [dens.reshape(shape).dot(self._orTo) for dens in densTup]
        self._trFList.append(trFList)
        self._trDList.append(trDList)
        self._UpdateCommBigMat()

        # coeff always has length numFock with 0.0 filled at the front in need
        numFock = len(self._fList)
        shapeTensor = (numFock, numFock, numFock, numFock)
        tensor = self._bigMat[:numFock**2, :numFock**2].reshape(shapeTensor)
        coeff = np.zeros(numFock)
        for numUse in range(len(tensor), 0, -1):
            tensorUse = tensor[-numUse:, -numUse:, -numUse:, -numUse:]
            (success, iniCoeffUse) = self._InitialCoeffUse(tensorUse)
            if not success:
                print('_InitialCoeffUse failed; reducing tensor size')
                continue
            (success, coeffUse) = self._NewtonSolver(tensorUse, iniCoeffUse)
            if not success:
                print('_NewtonSolver failed; reducing tensor size')
                continue
            else:
                coeff[-len(coeffUse):] = coeffUse
                break
        # end for
        if self._verbose:
            print('  lciis coeff:')
            print('  ' + str(coeff).replace('\n', '\n  '))
        shape = self._fList[0][0].shape
        return [coeff.dot([fList[spin].ravel()
                           for fList in self._fList]).reshape(shape)
                for spin in range(len(self._fList[0]))]

    def _PreUpdateFull(self):
        maxNumFock = self._fList.maxlen
        for ind in range(1, maxNumFock):
            sourceFrom = ind * maxNumFock + 1
            sourceTo = (ind + 1) * maxNumFock
            shiftBy = -(maxNumFock + 1)
            source = slice(sourceFrom, sourceTo)
            target = slice(sourceFrom + shiftBy, sourceTo + shiftBy)
            self._comm[target] = self._comm[source]
            self._bigMat[target, :] = self._bigMat[source, :]
            self._bigMat[:, target] = self._bigMat[:, source]

    def _PreUpdateNotFull(self):
        numFock = len(self._fList) + 1
        for ind in range(numFock - 1, 1, -1):
            sourceFrom = (ind - 1) * (numFock - 1)
            sourceTo = ind * (numFock - 1)
            shiftBy = ind - 1
            source = slice(sourceFrom, sourceTo)
            target = slice(sourceFrom + shiftBy, sourceTo + shiftBy)
            self._comm[target] = self._comm[source]
            self._bigMat[target, :] = self._bigMat[source, :]
            self._bigMat[:, target] = self._bigMat[:, source]

    def _UpdateCommBigMat(self):
        numFock = len(self._fList)
        # update self._comm
        update1 = range(numFock - 1, numFock**2 - 1, numFock)
        update2 = range(numFock**2 - numFock, numFock**2)
        for indComm in update1:
            self._comm[indComm] = self._Comm((indComm + 1) // numFock - 1, -1)
        for indComm in update2:
            self._comm[indComm] = self._Comm(-1, indComm - update2[0])
        # update self._bigMat
        update = list(update1) + list(update2)
        full = slice(0, numFock**2)
        commFull = np.array(self._comm[full])
        self._bigMat[update, full] = commFull[update, :].dot(commFull.T)
        self._bigMat[full, update] = self._bigMat[update, full].T

    def _Comm(self, indFock, indDens):
        def Comm(trF, trD):
            trFdotOrD = trF.dot(trD)
            comm = trFdotOrD - trFdotOrD.T
            return comm[np.triu_indices_from(comm, 1)]
        zipList = zip(self._trFList[indFock], self._trDList[indDens])
        return np.concatenate([Comm(trF, trD) for (trF, trD) in zipList])

    # return (success, cdiis_coefficients)
    def _InitialCoeffUse(self, tensorUse):
        numUse = len(tensorUse)
        ones = np.ones((numUse, 1))
        hess = np.zeros((numUse, numUse))
        # hess[i, i] = tensorUse[i, i, j, j]
        for ind in range(numUse):
            hess[ind, :] = np.diag(tensorUse[ind, ind, :, :])
        hessLag = np.bmat([[hess,   ones   ],
                           [ones.T, [[0.0]]]])
        gradLag = np.concatenate((np.zeros(numUse), [1.0]))
        iniCoeffUse = np.linalg.solve(hessLag, gradLag)[0:-1]
        return (not np.isnan(sum(iniCoeffUse)), iniCoeffUse)

    # return (success, lciis_coefficients)
    def _NewtonSolver(self, tensorUse, coeffUse):
        tensorGrad = tensorUse + tensorUse.transpose(1, 0, 2, 3)
        tensorHess = tensorGrad + tensorUse.transpose(0, 2, 1, 3)
        tensorHess += tensorUse.transpose(3, 0, 1, 2)
        tensorHess += tensorUse.transpose(0, 3, 1, 2)
        tensorHess += tensorUse.transpose(1, 3, 0, 2)
        ones = np.ones((len(coeffUse), 1))
        value = np.inf
        for _ in range(self._maxIterNewton):
            (grad, hess) = self._GradHess(tensorGrad, tensorHess, coeffUse)
            oldValue = value
            value = self._Value(tensorUse, coeffUse)
            gradLag = np.concatenate((grad, [0.0]))
            hessLag = np.bmat([[hess,      ones],
                               [ones.T, [[0.0]]]])
            step = np.linalg.solve(hessLag, gradLag)
            if np.isnan(sum(step)):
                print('Inversion failed')
                return (False, coeffUse)
            coeffUse -= step[0:-1]
            if np.abs(value - oldValue) < self._gradNormThres:
                return (True, coeffUse)
        # end for
        print('Newton did not converge')
        return (False, coeffUse)

    def _GradHess(self, tensorGrad, tensorHess, coeffUse):
        numUse = len(coeffUse)
        grad = tensorGrad.reshape(-1, numUse).dot(coeffUse)
        grad = grad.reshape(-1, numUse).dot(coeffUse)
        grad = grad.reshape(-1, numUse).dot(coeffUse)
        hess = tensorHess.reshape(-1, numUse).dot(coeffUse)
        hess = hess.reshape(-1, numUse).dot(coeffUse)
        hess = hess.reshape(-1, numUse)
        return (grad, hess)

    def _Value(self, tensorUse, coeffUse):
        numUse = len(coeffUse)
        value = tensorUse.reshape(-1, numUse).dot(coeffUse)
        value = value.reshape(-1, numUse).dot(coeffUse)
        value = value.reshape(-1, numUse).dot(coeffUse)
        value = value.dot(coeffUse)
        return value

