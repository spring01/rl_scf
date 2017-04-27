"""
An interface needs properties:

    numElecAB
    overlap
    oneElecHam

Needs methods:

    GuessDensity()
    FockEnergy(densTup)

"""

import subprocess
import re
import struct
import tempfile
import numpy as np


""" Works in conjunction with g09d01_backend """
class G09Interface(object):

    _intBytes = 4
    _floatBytes = 8
    _logLastLineLength = 200

    def __init__(self, info):
        self._workPath = tempfile.mkdtemp()
        self._workGjf = self._workPath + '/run.gjf'
        self._workLog = self._workPath + '/run.log'
        self._workDat = self._workPath + '/run.dat'
        self._BuildGjf(info)
        self._RunG09('iop(4/199=1) guess=harris')
        with open(self._workLog, 'r') as fLog:
            for line in fLog:
                if 'alpha electrons' in line:
                    numElecABStr = re.findall(r'[\d]+', line)
                    break
        self.numElecAB = tuple(int(num) for num in numElecABStr)
        with open(self._workDat, 'rb') as fDat:
            matrixBytes = struct.unpack('i', fDat.read(self._intBytes))[0]
            nbf = int(np.sqrt(matrixBytes / float(self._floatBytes)))
            nbfSq = nbf**2
            shape = nbf, nbf
            self.overlap = np.fromfile(fDat, float, nbfSq).reshape(shape)
            fDat.read(self._intBytes)
            fDat.read(self._intBytes)
            self._harrisMO = np.fromfile(fDat, float, nbfSq).reshape(shape).T
            fDat.read(self._intBytes)
            fDat.read(self._intBytes)
            self.oneElecHam = np.fromfile(fDat, float, nbfSq).reshape(shape)
            fDat.read(self._intBytes)

    def __del__(self):
        subprocess.call(['rm', '-r' , self._workPath])

    def _BuildGjf(self, info):
        numCPUCore = info['numcores'] if 'numcores' in info else 1
        memory = info['memory'] if 'memory' in info else '2gb'
        method = info['dft'] if 'dft' in info else 'hf'
        header = ['%nprocshared={:d}'.format(numCPUCore)]
        header.append('%mem={:s}'.format(memory))
        cmdLine = ' '.join(['#', method, info['basis'],
                            'iop(5/13=1,5/18=-2) scf(maxcycle=1,vshift=-1) '])
        header.append(cmdLine)
        body = ['', '', 'dummy title', '']
        body.append('{:3d} {:3d}'.format(info['charge'], info['mult']))
        cartEntryFormat = '{:3d}' + ' {:15.10f}' * 3
        for line in info['cart']:
            atomNum = int(line[0])
            coord = tuple(line[1:])
            body.append(cartEntryFormat.format(int(line[0]), *tuple(line[1:])))
        for _ in range(3):
            body.append('')
        self._gjf_header = '\n'.join(header)
        self._gjf_body = '\n'.join(body)

    # Initial guess density matrix
    def GuessDensity(self):
        numElecAB = tuple(np.unique(self.numElecAB)[::-1])
        guessOccMO = (self._harrisMO[:, :ne] for ne in numElecAB)
        return tuple(gmo.dot(gmo.T) for gmo in guessOccMO)

    # Construct a list of Fock matrix and calculate energy
    def FockEnergy(self, densTup):
        nbf = self.overlap.shape[0]
        nbfSq = nbf**2
        packedMatrixBytes = struct.pack('i', self._floatBytes * nbfSq)
        with open(self._workDat, 'wb') as fDat:
            for dens in densTup:
                fDat.write(packedMatrixBytes)
                dens.tofile(fDat)
                fDat.write(packedMatrixBytes)
        self._RunG09('iop(5/199=1) guess=core')
        fockList = []
        with open(self._workDat, 'rb') as fDat:
            fDat.read(self._intBytes)
            energy = struct.unpack('d', fDat.read(self._floatBytes))[0]
            fDat.read(self._intBytes)
            for _ in range(len(set(self.numElecAB))):
                fDat.read(self._intBytes)
                fock = np.fromfile(fDat, float, nbfSq).reshape(nbf, nbf)
                fDat.read(self._intBytes)
                fockList.append(fock)
        return tuple(fockList), energy

    def _RunG09(self, keyword):
        gjf = ''.join([self._gjf_header, keyword, self._gjf_body])
        with open(self._workGjf, 'w') as fGjf:
            fGjf.write(gjf)
        cmd = ['g09binfile=' + self._workDat, 'g09',
               self._workGjf, self._workLog]
        try:
            subprocess.call([' '.join(cmd)], shell=True)
        except:
            raise Exception('g09 did not terminate')
        with open(self._workLog, 'r') as fLog:
            fLog.seek(-self._logLastLineLength, 2) # go to the last line
            lastLine = fLog.readlines()[-1].decode()
        if 'Normal termination' not in lastLine:
            raise Exception('g09 terminated but failed')



