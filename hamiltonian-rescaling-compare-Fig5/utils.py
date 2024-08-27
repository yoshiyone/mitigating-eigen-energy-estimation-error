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

def noisyEigenData(n,noisyHamiltonian:dict,phiA,phiB,collapseOperators:list,options,deltaT,L,N_poles=4,cutoff=1e-2):
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
    energyGaps: The energy gap between phiA and phiB.
    N_modes: The actual number of modes retrieved from the signal.
    '''
    initState=loadState(1/np.sqrt(2)*(phiA+phiB),n)
    measurement=2*loadState(phiB,n)*loadState(phiA,n).dag()

    tlist=np.linspace(0,L*deltaT,L+1)
    result=mesolve(qutipHamiltonian(noisyHamiltonian),initState,tlist,collapseOperators,[measurement],options=options,progress_bar=None)

    matrixPencilResult=mp_est(result.expect[0][0:L],1,N_poles=N_poles,cutoff=cutoff)
    energyGaps=matrixPencilResult[0]/deltaT
    N_modes=len(matrixPencilResult[1])

    return energyGaps,N_modes

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

    maxN_modes=max(noisyResult[1],c1Result[1],c2Result[1])

    if maxN_modes != noisyResult[1]:
        noisyResult=noisyEigenData(n,hamSysErrorFunc(hamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=deltaT,L=L,N_poles=maxN_modes,cutoff=1e-12)
    if maxN_modes != c1Result[1]:
        c1Result=noisyEigenData(n,hamSysErrorFunc(c1rescaledHamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=c_1*deltaT,L=L,N_poles=maxN_modes,cutoff=1e-12)
    if maxN_modes != c2Result[1]:
        c2Result=noisyEigenData(n,hamSysErrorFunc(c2rescaledHamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=c_2*deltaT,L=L,N_poles=maxN_modes,cutoff=1e-12)

    print(noisyResult[0])
    print(c1Result[0])
    print(c2Result[0])
    firstResult=(noisyResult[0]-c1Result[0])/(1-1/c_1)
    secondResult=secondOrderCorrection(noisyResult[0],c1Result[0],c2Result[0],c_1,c_2)

    return noisyResult[0], firstResult, secondResult

def signalGenerationSpecific(n,noisyHamiltonian:dict,phiA,phiB,collapseOperators:list,options,deltaT,L):
    '''
    Return the <2|phi_b><phi_a|>-t signal.

    Parameters
    ----------
    kappa: error strength
    n: # of qubits
    hamiltonian: H which is stored into a dictionary.
    refState: |phi_0>
    tState: superposition of states which we focus on.
    jumpOperators: a list which describe the normalized jump operators and each operator is in `Qobj` form.
    options: qutip.solver.Option()
    deltaT: deltaT.
    n_t: tf=n_t*deltaT
    saveDataAddress: Save the signal into a csv file if is not None.

    Returns
    ----------
    The signal <2|phi_b><phi_a|>-t

    '''
    initState=loadState(1/np.sqrt(2)*(phiA+phiB),n)
    measurement=2*loadState(phiB,n)*loadState(phiA,n).dag()

    tlist=np.linspace(0,L*deltaT,L+1)
    result=mesolve(qutipHamiltonian(noisyHamiltonian),initState,tlist,collapseOperators,[measurement],options=options,progress_bar=None)
    
    return result.expect[0]

def oneFactorRichardsonSignal(noisySignal,c1Signal,c1):
    '''
    Return the signal recovered by Richardson extrapolation with single rescaling factor.
    
    Parameters
    ----------
    noisySignal: the signal before rescaling.
    c1Signal: the signal after rescaling with rescaling factor c1.
    c1: the rescaling factor.
    
    Returns
    ----------
    The signal recovered by Richardson extrapolation.
    '''
    return noisySignal*c1/(c1-1)-c1Signal*1/(c1-1)

def twoFactorRichardsonSignal(noisySignal,c1Signal,c2Signal,c1,c2):
    '''
    Return the signal recovered by Richardson extrapolation with single rescaling factor.
    
    Parameters
    ----------
    noisySignal: the signal before rescaling.
    c1Signal: the signal after rescaling with rescaling factor c1.
    c1: the rescaling factor.
    
    Returns
    ---------
    The signal recovered by Richardson extrapolation.
    '''
    return noisySignal*c1*c2/(c1-1)/(c2-1)+c1Signal*c2/(c1-c2)/(c1-1)+c2Signal*(-c1)/(c2-1)/(c1-c2)

def rescalingMitigationCompare(kappa,ham_err_strength,n,hamiltonian:dict,phiA,phiB,collapseOperatorsFunc,hamSysErrorFunc,options,deltaT,L,c_1,c_2,N_poles=4):
    '''
    Return noisy result, first order mitigation result, second order mitigation result by Hamiltonian rescaling method and the standard Richardson extrapolation method with one and two factors.
    
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
    noisyResult, firstResult, secondResult, oneFactorResult, twoFactorsResult
    '''
    noisySignal=signalGenerationSpecific(n,hamSysErrorFunc(hamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=deltaT,L=L)

    c1rescaledHamiltonian=hamiltonian.copy()
    c2rescaledHamiltonian=hamiltonian.copy()
    
    for key in c1rescaledHamiltonian.keys():
        c1rescaledHamiltonian[key]/=c_1
    
    for key in c2rescaledHamiltonian.keys():
        c2rescaledHamiltonian[key]/=c_2

    c1RescaledSignal=signalGenerationSpecific(n,hamSysErrorFunc(c1rescaledHamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=c_1*deltaT,L=L)
    c2RescaledSignal=signalGenerationSpecific(n,hamSysErrorFunc(c2rescaledHamiltonian,n,ham_err_strength),phiA,phiB,collapseOperatorsFunc(kappa),options=options,deltaT=c_2*deltaT,L=L)

    noisyResult=mp_est(noisySignal,1,N_poles=N_poles,cutoff=1e-2)
    c1Result=mp_est(c1RescaledSignal,1,N_poles=N_poles,cutoff=1e-2)
    c2Result=mp_est(c2RescaledSignal,1,N_poles=N_poles,cutoff=1e-2)

    noisyEba=noisyResult[0]/deltaT
    c1Eba=c1Result[0]/deltaT/c_1
    c2Eba=c2Result[0]/deltaT/c_2

    print(noisyEba)
    print(c1Eba)
    print(c2Eba)

    firstEba=(noisyEba-c1Eba)/(1-1/c_1)
    secondEba=secondOrderCorrection(noisyEba,c1Eba,c2Eba,c_1,c_2)

    mitigatedSignal1=oneFactorRichardsonSignal(noisySignal,c1RescaledSignal,c_1)
    mitigatedSignal2=twoFactorRichardsonSignal(noisySignal,c1RescaledSignal,c2RescaledSignal,c_1,c_2)
    
    f_RE=mp_est(mitigatedSignal1,1,100,cutoff=1e-2)[0]/deltaT
    s_RE=mp_est(mitigatedSignal2,1,100,cutoff=1e-2)[0]/deltaT

    return noisyEba,firstEba,secondEba,f_RE,s_RE
