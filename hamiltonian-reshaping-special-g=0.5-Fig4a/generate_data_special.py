import numpy as np
from utils import pauliTransform,noisyEigenData
from models import transversalXYZIsingModel,errHamLocalSumZ,localSumCollapseList
from exact_diagonalization import eigenSolver, stateTransform
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy, expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, thermal_dm)
from qutip.solver import Options
from matrix_pencil import mp_est
import time

'''
Load the random Pauli strings.
'''
import csv

# Specify the file path
csv_file_path = "specialPauli.csv"

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
csv_file_path = "a_b.csv"

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

'''
Load the signals.
'''
def loadSignal(a,b,pauliString,label):
    '''
    Return the signal information.

    Parameters
    ----------
    a: phi_a index.
    b: phi_b index.
    pauliString: the correspond Pauli string used to reshape the Hamiltonian.
    label: From 0 to len(gammaList)

    Returns
    ----------
    tList,signals,gamma
    '''
    data=np.genfromtxt("./signals/noisy_"+str(a)+"_"+str(b)+"_"+str(pauliString)+"_"+str(label)+".csv", delimiter=',', names=True, dtype=None)
    return data["t"],data["signal"],data["gamma"][0]

'''
Generate and store the data of different gamma.
'''

def dataWritingWithHeader(path:str,zippedList):
    '''
    Write a zipped list into a csv file.
    '''
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['gamma','Pauli_string', 'energy_gap'])
        csv_writer.writerows(zippedList)

'''
Generate data from the signals.
'''

n=6
hamiltonian=transversalXYZIsingModel(-0.5,0,0,-1,-1,0,n)
# print(hamiltonian)
eigenvalues,eigenstates=eigenSolver(hamiltonian,n)
# print(eigenvalues)

idString='I'
for i in range(n-1):
    idString+='I'

# # kappa=gamma*|deltaE|, ham_err_strength=gamma*beta*|deltaE|
# # maxGamma=0.02
# # gammaNums=20
# # gammaList=np.array([maxGamma*(i+1)/gammaNums for i in range(gammaNums)])
gammaList=np.array([1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,3e-2,5e-2,1e-1])

L=2000
deltaT0=0.0001
beta=0.01

it=1
for randomNums in randomStatesList[0:10]:
    print("Iteration ",it)
    a=int(randomNums[0])
    b=int(randomNums[1])
    print("Random 2 eigenstates:",a,b)
    # initState=1/np.sqrt(2)*(eigenstates[a]+eigenstates[b])
    idealValue=eigenvalues[b]-eigenvalues[a]
    print("Exact diagonalization result:",idealValue)

    gammas=[]
    pauli_strings=[]
    energy_gaps=[]

    for gammaLabel in range(len(gammaList)):
        starttime=time.time()

        # Load signal
        unmitigatedSignal=loadSignal(a,b,idString,gammaLabel)
        gamma=unmitigatedSignal[2]
        gammas.append(gamma)
        noisySignals=[]
        randomSampleNum=2
        for i in range(randomSampleNum):
            noisySignals.append(loadSignal(a,b,randomPauliStrings[0][i],gammaLabel))

        unmitigatedResult=mp_est(unmitigatedSignal[1],1,N_poles=100,cutoff=1e-10)

        noisyResults=[]
        for i in range(randomSampleNum):
            noisyResult=mp_est(noisySignals[i][1],1,N_poles=100,cutoff=1e-10)
            noisyResults.append(noisyResult)

        # Write the data to files.
        pauli_strings.append(idString)
        energy_gaps.append(unmitigatedResult[0][0]/deltaT0)

        for i in range(randomSampleNum):
            print("Iteration ",'(',it,i+1,')',f"gamma={gamma}")
            randomPauli=randomPauliStrings[0][i]
            print("Random pauli: ", randomPauli)
            # print(pauliTransform(hamiltonian,randomPauli))
            noisyEnergyGap=noisyResults[i][0]/deltaT0

            print(noisyEnergyGap)
            gammas.append(gamma)
            pauli_strings.append(randomPauli)
            energy_gaps.append(noisyEnergyGap[0])

        endtime=time.time()
        print(f"Total runtime for gamma={gamma}:",endtime-starttime)

        print(np.average(energy_gaps[1:]))

    combined_data=list(zip(gammas,pauli_strings,energy_gaps))

    dataWritingWithHeader("data/"+str(a)+'_'+str(b)+"_special"+".csv",combined_data)
    it+=1