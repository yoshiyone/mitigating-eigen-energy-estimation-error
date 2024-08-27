import numpy as np
from models import ringModel,localSumCollapseList,errHamLocalSumZ
from utils import rescalingMitigation
from exact_diagonalization import eigenSolver
from qutip.solver import Options
import time

'''
Load random 2 numbers.
'''

import csv

# Specify the file path
csv_file_path = "100Random2Numbers.csv"

# Create an empty list to store the data
randomtStatesList = []

# Open the CSV file in read mode
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Iterate through each row in the CSV file and append it to the list
    for row in reader:
        randomtStatesList.append(row)

# # Display the loaded data
# for row in randomtStatesList:
#     print(row)

'''
Generate and store the data of different kappa.
'''

def dataWritingWithHeader(path:str,zippedList):
    '''
    Write a zipped list into a csv file.
    '''
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['gamma', 'noisy', 'first_order', 'second_order'])
        csv_writer.writerows(zippedList)

'''
Generate data
'''

n=6
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

hamiltonian=ringModel(4,1,4,n)
# print(hamiltonian)
eigenvalues,eigenstates=eigenSolver(hamiltonian,n)
# print(eigenvalues)

it=1
for randomtNums in randomtStatesList[0:100]:
    print("Iteration ",it)
    a=int(randomtNums[0])
    b=int(randomtNums[1])
    print("a:",a,", b:",b)
    deltaE=eigenvalues[b]-eigenvalues[a]
    print("Exact diagonalization result:",deltaE)

    noisy=[]
    first_order=[]
    second_order=[]

    for gamma in gammaList:
        start_time = time.time()
        energyGapsMitigation=rescalingMitigation(kappa=gamma*np.abs(deltaE),ham_err_strength=gamma*beta*np.abs(deltaE),n=n,hamiltonian=hamiltonian,phiA=eigenstates[a],phiB=eigenstates[b],collapseOperatorsFunc=lambda kappa: [oper*np.sqrt(kappa) for oper in localSumCollapseList(n,np.pi/2)],hamSysErrorFunc=errHamLocalSumZ,options=options,deltaT=deltaT0,L=L,c_1=2,c_2=1.5,N_poles=100)

        print(f"Noisy rate gamma={gamma}","result:",energyGapsMitigation[0])
        noisy.append(energyGapsMitigation[0][0])
        print("First-order correction result:", energyGapsMitigation[1])
        first_order.append(energyGapsMitigation[1][0])
        print("Second-order correction result:", energyGapsMitigation[2])
        second_order.append(energyGapsMitigation[2][0])
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Process runtime: {runtime} s")

        print(np.average([np.abs((energyGapsMitigation[j]-deltaE)/deltaE) for j in range(3)],axis=1))
    
    combined_data=list(zip(gammaList,noisy,first_order,second_order))

    dataWritingWithHeader("data/"+str(a)+"_"+str(b)+".csv",combined_data)

    it+=1