import numpy as np

'''
Exact Diagonalization

Note: We rewrite the ED code to match the Pauli string.
'''

def inttobin(number,n):
    '''
    Return length n binary string.
    '''
    form='0'+str(n)+'b'
    return format(number,form)

def xLocal(i,j,k,n):
    '''
    Return <i_k|X|j_k> where X is local Pauli X matrix.
    ---------
    n: number of qubits.
    k: Define i=2**(n-1)*i_0+2**(n-2)*i_1+2**(n-3)*i_2+..., the kth qubit is i_k.
    '''
    return int(inttobin(i^j,n)[k])

def iLocal(i,j,k,n):
    '''
    Return <i_k|I|j_k> where I is local identity matrix.
    '''
    ik=inttobin(i,n)[k]
    jk=inttobin(j,n)[k]
    if ik==jk:
        return 1
    else:
        return 0

def zLocal(i,j,k,n):
    '''
    Return <i_k|Z|j_k> where Z is local Pauli Z matrix.
    '''
    ik=inttobin(i,n)[k]
    jk=inttobin(j,n)[k]
    if ik==jk:
        if ik=='0':
            return 1
        else:
            return -1
    else:
        return 0
    
def yLocal(i,j,k,n):
    '''
    Return <i_k|Y|j_k> where Y is local Pauli Y matrix.
    '''
    ik=inttobin(i,n)[k]
    jk=inttobin(j,n)[k]
    if ik==jk:
        return 0
    else:
        if ik=='0':
            return -1.j
        else:
            return 1.j

def pauliMatrixElement(pauliString,i,j,k,n):
    '''
    Return <i_k|P_k|j_k>.
    '''
    if pauliString[k]=='I':
        return iLocal(i,j,k,n)
    elif pauliString[k]=='X':
        return xLocal(i,j,k,n)
    elif pauliString[k]=='Y':
        return yLocal(i,j,k,n)
    elif pauliString[k]=='Z':
        return zLocal(i,j,k,n)
    else:
        print('ED local Pauli matrix element error.')
        quit(1)

def matrixElement(hamiltonian:dict,i,j,n):
    '''
    Return the matrix element <i|H|j>.
    
    Parameters
    ----------
    hamiltonian: `dict` which store the Hamiltonian.
    i: `int`
    j: `int`
    n: number of qubits.
    
    Return
    ----------
    The matrix element of the Hamiltonian.
    '''
    element=0

    for pauliString in hamiltonian.keys():
        tempElement=1

        for k in range(n):
            tempElement*=pauliMatrixElement(pauliString,i,j,k,n)
        
        element+=hamiltonian[pauliString]*tempElement

    return element

def eigenSolver(hamiltonian:dict,n):
    '''
    Return the eigenvalues and eigenstates of given hamiltonian.
    '''
    hamMatrix=np.zeros((2**n,2**n),dtype=complex)

    for i in range(2**n):
        for j in range(2**n):
            hamMatrix[i][j]=matrixElement(hamiltonian,i,j,n)

    eigenvalues, eigenvectors = np.linalg.eigh(hamMatrix)
    eigenvectors=np.transpose(eigenvectors)
    return eigenvalues,eigenvectors

def localTransform(bit,pauli):
    if pauli=='I':
        return bit,1
    elif pauli=='X':
        return bin(1^int(bit,2))[2],1
    elif pauli=='Y':
        return bin(1^int(bit,2))[2],(-1)**int(bit,2)*1.j
    elif pauli=='Z':
        return bit,(-1)**int(bit,2)
    else:
        print("Local pauli transform error.")
        quit(1)

def basisTransform(phi_i:int,pauliString):
    '''
    Return the basis and coefficient after Pauli transform.
    
    Parameters
    ----------
    phi_i: `int` which stand for the basis before Pauli transform.
    pauliString: the Pauli transform.
    
    Return
    ----------
    phi_f: `int` which stand for the basis after transform.
    c: coefficient after transform.
    
    Example
    ----------
    >>> basisTransform(3,'XZ')
    3 -> '11'
    '11' -> 'XZ' -> -1*'01'
    return (1,-1)
    '''
    c=1
    phi_f_str=""
    n=len(pauliString)
    basisString=inttobin(phi_i,n)

    for k in range(n):
        phi_k_str,c_k=localTransform(basisString[k],pauliString[k])
        phi_f_str+=phi_k_str
        c*=c_k
    
    return int(phi_f_str,2),c

def stateTransform(state,pauliString):
    '''
    Transform a state with given Pauli string.

    Parameters
    -----------
    state: numpy array
    pauliString: string which stand for the goal Pauli transform

    Return
    -----------
    finalState: numpy array which stand for the state after transformation
    '''
    n=len(pauliString)
    finalState=np.zeros((2**n),dtype=complex)

    for i in range(2**n):
        phi_f,c_f=basisTransform(i,pauliString)
        finalState[phi_f]=c_f*state[i]
    
    return finalState


if __name__=="__main__":
    testState=np.array([1,0,0,0],dtype=complex)
    print(stateTransform(testState,'XX'))