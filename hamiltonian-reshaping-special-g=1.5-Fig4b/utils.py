import numpy as np
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy, expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, thermal_dm)
from matrix_pencil import mp_est

'''
Hamiltonian can be represented as weighted summation of Pauli strings and can be stored into python dictionary.
Example: {'XIIIII':-0.5,'IXIIII':-0.5,...}
'''

def localPauliToQobj(localPauli):
    if localPauli=='I':
        return qeye(2)
    elif localPauli=='X':
        return sigmax()
    elif localPauli=='Y':
        return sigmay()
    elif localPauli=='Z':
        return sigmaz()
    else:
        print('Local Pauli to Qobj error.')
        quit(1)

def qutipHamiltonian(hamiltonian:dict):
    '''
    Return the Hamiltonian which can be passed into qutip's mesolve function.

    Parameters
    ---------
    hamiltonian: a dictionary which include the information of Hamiltonian.

    Returns
    ---------
    `Qobj` of correspond Hamiltonian.
    '''
    ham=0

    def pauliToQobj(pauliString):
        n=len(pauliString)
        qobjResult=localPauliToQobj(pauliString[0])
        
        for i in range(n-1):
            qobjResult=tensor(qobjResult,localPauliToQobj(pauliString[i+1]))
        
        return qobjResult

    for key in hamiltonian.keys():
        ham+=hamiltonian[key]*pauliToQobj(key)

    return ham

def inttobin(number,n):
    '''
    Return length n binary string.
    '''
    form='0'+str(n)+'b'
    return format(number,form)

def loadState(quantumState,n)->Qobj:
    '''
    Load a given quantum state in qutip.
    ----------
    quantumState: (2**n) numpy array.
    n: number of qubits.
    ----------
    Return the quantum state in qutip.
    '''
    qutipState=0

    for i in range(2**n):
        bitstring=inttobin(i,n)
        qutipBasis=basis(2,int(bitstring[0]))
        for j in range(1,n):
            qutipBasis=tensor(qutipBasis,basis(2,int(bitstring[j])))
        qutipState+=quantumState[i]*qutipBasis

    return Qobj(qutipState)

def noisyEigenData(n,noisyHamiltonian:dict,phiA,phiB,collapseOperators:list,options,deltaT,L,N_poles=4):
    '''
    Return the energy gap between phiA and phiB evaluated by the noisy protocol given by numerical simulation.

    Parameters
    ----------
    n: # of qubits
    noisyHamiltonian: Hamiltonian with systematic error.
    phiA: |\phi_a>
    phiB: |\phi_b>
    collapseOperators: a list which describe the collapse operators and each operator is in `Qobj` form.
    deltaT: deltaT.
    options: qutip.solver.Option()
    L: The number of data points in the signal. We process the signal <O>(k dT), k=0,1,...,L-1.
    N_poles: The number of maximum possible poles the data can be decomposed into. Choosing this number too small will lead to bad fits so act with care.

    Return
    ----------
    The energy gap between phiA and phiB.
    '''
    initState=loadState(1/np.sqrt(2)*(phiA+phiB),n)
    measurement=2*loadState(phiB,n)*loadState(phiA,n).dag()

    tlist=np.linspace(0,L*deltaT,L+1)
    result=mesolve(qutipHamiltonian(noisyHamiltonian),initState,tlist,collapseOperators,[measurement],options=options)

    energyGaps=mp_est(result.expect[0][0:L],1,N_poles=N_poles)[0]/deltaT

    return energyGaps

def secondOrderCorrection(omega0,omega1,omega2,c1,c2):
    '''
    Return the energy gap after second-order correction.

    Parameters
    -----------
    omega0: delta phi ab/delta T.
    omega1: delta phi ab prime/ delta T prime.
    omega2: delta phi ab prime prime/ delta T prime prime.
    c1: H prime = H/c1
    c2: H prime prime = H/c2

    Returns
    -----------
    The energy gap after second-order correction.
    '''
    coefficient=c1*c2/((c2-c1)*(c1-1)*(c2-1))
    return -coefficient*((c1-c2)*omega0+(c2-1)*omega1-(c1-1)*omega2)

def rescalingMitigation(kappa,ham_err_strength,n,hamiltonian:dict,phiA,phiB,collapseOperatorsFunc,hamSysErrorFunc,options,deltaT,L,c_1,c_2,N_poles=4):
    '''
    Return noisy result, first order mitigation result and second order mitigation result by Hamiltonian rescaling method.
    
    Parameters
    ----------
    c_1: rescaling factor c_1, correspond with H/c_1
    c_2: rescaling factor c_2, correspond with H/c_2
    collapseOperatorsFunc: a function which can return the list of collapse operators given kappa.
    hamSysErrorFunc: a function which can return the hamiltonian with system error given hamiltonian, n and hamiltonian error strength.
    other parameters are the same as noisyEigenData.
    The first order result is related to the no rescaling data and c_1 rescaling data.

    Returns
    ----------
    noisyResult, firstResult, secondResult
    '''
    noisyResult=noisyEigenData(n,hamSysErrorFunc(hamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=deltaT,L=L,N_poles=N_poles)

    c1rescaledHamiltonian=hamiltonian.copy()
    c2rescaledHamiltonian=hamiltonian.copy()
    
    for key in c1rescaledHamiltonian.keys():
        c1rescaledHamiltonian[key]/=c_1
    
    for key in c2rescaledHamiltonian.keys():
        c2rescaledHamiltonian[key]/=c_2

    c1Result=noisyEigenData(n,hamSysErrorFunc(c1rescaledHamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=c_1*deltaT,L=L,N_poles=N_poles)
    c2Result=noisyEigenData(n,hamSysErrorFunc(c2rescaledHamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=c_2*deltaT,L=L,N_poles=N_poles)

    print(noisyResult)
    print(c1Result)
    print(c2Result)
    firstResult=(noisyResult-c1Result)/(1-1/c_1)
    secondResult=secondOrderCorrection(noisyResult,c1Result,c2Result,c_1,c_2)

    return noisyResult, firstResult, secondResult

def signLocalPauli(pauli1,pauli2):
    '''
    Return the sign of P_2 P_1 P_2.
    
    Parameters
    ----------
    pauli1, pauli2: the char of local Pauli operator. 'I' 'X' 'Y' 'Z'
    
    Returns
    ----------
    +1 or -1
    '''
    if pauli1=='I' or pauli2=='I':
        return 1
    elif pauli1==pauli2:
        return 1
    else:
        return -1

def pauliTransform(hamiltonian:dict,pauliString):
    '''
    Return transformed Hamiltonian after transformation. The transformation is determined by a Pauli string.
    
    Parameters
    ----------
    hamiltonian: `dict` which stores the information of Hamiltonian.
    pauliString: the Pauli string P. Transform hamiltonian H into PHP.
    
    Return
    ----------
    transformed Hamiltonian: `dict`
    '''
    n=len(pauliString)
    transformedH=hamiltonian.copy()

    for i in range(n):
        for key in hamiltonian.keys():
            transformedH[key]*=signLocalPauli(key[i],pauliString[i])
        
    return transformedH

