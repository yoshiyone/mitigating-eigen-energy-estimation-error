import numpy as np
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy, expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, thermal_dm, Options)
from utils import loadState,qutipHamiltonian,pauliTransform
from exact_diagonalization import eigenSolver,stateTransform
from models import ringModel,errHamLocalSumZ,localSumCollapseList
import time

import csv

'''
Load random Pauli
'''
# Specify the file path
csv_file_path = "4Pauli.csv"

# Create an empty list to store the data
randomPauliStrings = []

# Open the CSV file in read mode
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Iterate through each row in the CSV file and append it to the list
    for row in reader:
        randomPauliStrings.append(row)

# # Display the loaded data
# for row in randomPauliStrings:
#     print(row)

'''
Load various a,b.
'''
# Specify the file path
csv_file_path = "100Random2Numbers.csv"

# Create an empty list to store the data
randomStatesList = []

# Open the CSV file in read mode
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Iterate through each row in the CSV file and append it to the list
    for row in reader:
        randomStatesList.append(row)

# # Display the loaded data
# for row in randomStatesList:
#     print(row)

def dataWritingWithHeader(path:str,zippedList):
    '''
    Write a zipped list into a csv file.
    '''
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['t','signal','gamma'])
        csv_writer.writerows(zippedList)

def generateNoisySignal(n,noisyHamiltonian:dict,phiA,phiB,collapseOperators:list,options,deltaT,L):
    '''
    Generate the noisy signal by numerical simulation.

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

    Return
    ----------
    The noisy signal given the initial settings.
    '''
    initState=loadState(1/np.sqrt(2)*(phiA+phiB),n)
    measurement=2*loadState(phiB,n)*loadState(phiA,n).dag()

    tlist=np.linspace(0,L*deltaT,L+1)
    result=mesolve(qutipHamiltonian(noisyHamiltonian),initState,tlist,collapseOperators,[measurement],options=options)

    return tlist,result.expect[0]

# Path: noisy_a_b_{PauliString}.csv
def signalPath(a,b,randomPauli,label):
    return "signals-4Pauli/noisy_"+str(a)+"_"+str(b)+"_"+randomPauli+"_"+str(label)+".csv"

'''
Generate data.
'''

n=6
hamiltonian=ringModel(4,1,4,n)
# print(hamiltonian)
eigenvalues,eigenstates=eigenSolver(hamiltonian,n)
# print(eigenvalues)

options=Options()
options.atol=1e-16
options.rtol=1e-16
# options.max_step=1e-4
options.nsteps=10000000

# kappa=gamma*|deltaE|, ham_err_strength=gamma*beta*|deltaE|
# maxGamma=0.02
# gammaNums=20
# gammaList=np.array([maxGamma*(i+1)/gammaNums for i in range(gammaNums)])
gammaList=np.array([1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,3e-2,5e-2,1e-1])

L=2000
deltaT0=0.0001
beta=0.01

it=1
for randomNums in randomStatesList[0:100]:
    print("Iteration ",it)
    a=int(randomNums[0])
    b=int(randomNums[1])
    print("Random 2 eigenstates:",a,b)
    # initState=1/np.sqrt(2)*(eigenstates[a]+eigenstates[b])
    idealValue=eigenvalues[b]-eigenvalues[a]
    print("Exact diagonalization result:",idealValue)

    gammaLabel=0
    for gamma in gammaList:
        
        starttime=time.time()

        noisySignal=generateNoisySignal(n,noisyHamiltonian=errHamLocalSumZ(hamiltonian,n,gamma*beta*np.abs(idealValue)),phiA=eigenstates[a],phiB=eigenstates[b],collapseOperators=[np.sqrt(gamma*np.abs(idealValue))*i for i in localSumCollapseList(n,phi=np.pi/2)],options=options,deltaT=deltaT0,L=L)

        idString='I'
        for i in range(n-1):
            idString+='I'

        combined_data=list(zip(noisySignal[0],noisySignal[1],[gamma for j in range(L+1)]))
        dataWritingWithHeader(signalPath(a,b,idString,gammaLabel),combined_data)

        randomSampleNum=4
        for i in range(randomSampleNum):
            print("Iteration ",'(',it,i+1,')',f"gamma={gamma}")
            randomPauli=randomPauliStrings[0][i]
            print("Random pauli: ", randomPauli)
            # print(pauliTransform(hamiltonian,randomPauli))
            transformedSignal=generateNoisySignal(n,noisyHamiltonian=errHamLocalSumZ(pauliTransform(hamiltonian,randomPauli),n,gamma*beta*np.abs(idealValue)),phiA=stateTransform(eigenstates[a],randomPauli),phiB=stateTransform(eigenstates[b],randomPauli),collapseOperators=[np.sqrt(gamma*np.abs(idealValue))*i for i in localSumCollapseList(n,phi=np.pi/2)],options=options,deltaT=deltaT0,L=L)
            combined_data=list(zip(transformedSignal[0],transformedSignal[1],[gamma for j in range(L+1)]))
            dataWritingWithHeader(signalPath(a,b,randomPauli,gammaLabel),combined_data)
            
        endtime=time.time()
        print(f"Total runtime for gamma={gamma}:",endtime-starttime)
        gammaLabel+=1

    it+=1