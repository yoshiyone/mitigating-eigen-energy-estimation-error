import numpy as np
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy, expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, thermal_dm)

def ringModel(nuz,nux,J,n)->dict:
    '''
    Return the qubit ring model Hamiltonian in a dictionary.

    H = 0.5 sum (2pi nuz Z_i + 2pi nux X_i) + 0.5 sum 2pi J (X_i X_j + Y_i Y_j)
    '''
    hamiltonian={}
    
    # 0.5 sum (2pi nuz Z_i + 2pi nux X_i)
    for i in range(n):
        if i==0:
            pauliString='Z'
            for it in range(n-1):
                pauliString=pauliString+'I'
        else:
            pauliString='I'
            for it in range(n-1):
                if it==i-1:
                    pauliString=pauliString+'Z'
                else:
                    pauliString=pauliString+'I'
        
        hamiltonian[pauliString]=np.pi*nuz
    
    for i in range(n):
        if i==0:
            pauliString='X'
            for it in range(n-1):
                pauliString=pauliString+'I'
        else:
            pauliString='I'
            for it in range(n-1):
                if it==i-1:
                    pauliString=pauliString+'X'
                else:
                    pauliString=pauliString+'I'
        
        hamiltonian[pauliString]=np.pi*nux

    # 0.5 sum 2pi J (X_i X_j + Y_i Y_j)
    for i in range(n-1):
        if i==0:
            pauliString='X'
            for it in range(n-1):
                if it==0:
                    pauliString=pauliString+'X'
                else:
                    pauliString=pauliString+'I'
        elif i>0 and i<n-2:
            pauliString='I'
            for it in range(n-1):
                if it==i-1 or it==i:
                    pauliString=pauliString+'X'
                else:
                    pauliString=pauliString+'I'
        else:
            pauliString='X'
            for it in range(n-1):
                if it==n-2:
                    pauliString=pauliString+'X'
                else:
                    pauliString=pauliString+'I'
        
        hamiltonian[pauliString]=np.pi*J

    for i in range(n-1):
        if i==0:
            pauliString='Y'
            for it in range(n-1):
                if it==0:
                    pauliString=pauliString+'Y'
                else:
                    pauliString=pauliString+'I'
        elif i>0 and i<n-2:
            pauliString='I'
            for it in range(n-1):
                if it==i-1 or it==i:
                    pauliString=pauliString+'Y'
                else:
                    pauliString=pauliString+'I'
        else:
            pauliString='Y'
            for it in range(n-1):
                if it==n-2:
                    pauliString=pauliString+'Y'
                else:
                    pauliString=pauliString+'I'
        
        hamiltonian[pauliString]=np.pi*J

    return hamiltonian

def localSumCollapseList(n:int,phi=np.pi/2):
    '''
    Return the list of collapse operators sum_i C_local_i
    '''
    C_local=Qobj(np.array([[1.j*np.sin(phi)+np.cos(phi),0],[0,1]],dtype=complex))

    collapseList=[]

    for i in range(n):
        if i == 0:
            temp=C_local
            for j in range(n-1):
                temp=tensor(temp,qeye(2))
            collapseList.append(temp)
        else:
            temp=qeye(2)
            for j in range(n-1):
                if j==i-1:
                    temp=tensor(temp,C_local)
                else:
                    temp=tensor(temp,qeye(2))
            collapseList.append(temp)
    
    return collapseList


def errHamLocalSumZ(hamiltonian,n,error_strength):
    '''
    Return the Hamiltonian with systematic error.
    H_err = H + error_strength * \sum_j Z_j
    
    Parameters
    -------
    hamiltonian: `dict`
    n: # of qubits
    error_strength: error strength

    Returns
    -------
    errHamiltonian
    '''
    errHamiltonian=hamiltonian.copy()
    for i in range(n):
        if i==0:
            zString='Z'
            for j in range(n-1):
                zString+='I'
        else:
            zString='I'
            for j in range(n-1):
                if j==i-1:
                    zString+='Z'
                else:
                    zString+='I'
        
        if zString in errHamiltonian:
            errHamiltonian[zString]+=error_strength
        else:
            errHamiltonian[zString]=error_strength
    
    return errHamiltonian