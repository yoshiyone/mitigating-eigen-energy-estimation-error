from matplotlib import pyplot as plt
import numpy as np
from exact_diagonalization import eigenSolver
from models import ringModel
import csv

n=6
hamiltonian=ringModel(4,1,4,n)
# print(hamiltonian)
eigenvalues,eigenstates=eigenSolver(hamiltonian,n)

'''
Load random numbers.
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

'''
Load data.
'''

# kappa=gamma*|deltaE|, ham_err_strength=gamma*beta*|deltaE|
# maxGamma=0.02
# gammaNums=20
# gammaList=np.array([maxGamma*(i+1)/gammaNums for i in range(gammaNums)])
gammaList=np.array([1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,3e-2,5e-2,1e-1])

randomSampleNum=100

un=[]
mi=[]
un_4Pauli=[]
mi_4Pauli=[]

plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 22})
bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
# it=1
for randomNums in randomStatesList[0:100]:
    # print("Iteration ",it)
    a=int(randomNums[0])
    b=int(randomNums[1])
    # print("Random a b:",a,b)
    idealValues=eigenvalues[b]-eigenvalues[a]
    # print("Exact diagonalization result:",idealValues)

    # Read the data
    filename = 'data/'+str(a)+"_"+str(b)+'.csv'
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None)

    unmitigated=np.abs((data['energy_gap'][0::randomSampleNum+1]-idealValues)/idealValues)

    mitigated=np.abs(np.array([(np.average(data['energy_gap'][i*(randomSampleNum+1)+1:(i+1)*(randomSampleNum+1)])-idealValues)/idealValues for i in range(len(gammaList))]))

    un.append(unmitigated)
    mi.append(mitigated)

    # Read the 4 Pauli data
    filename_4Pauli = 'data-4Pauli/'+str(a)+"_"+str(b)+'.csv'
    data_4Pauli = np.genfromtxt(filename_4Pauli, delimiter=',', names=True, dtype=None)

    unmitigated_4Pauli=np.abs((data_4Pauli['energy_gap'][0::5]-idealValues)/idealValues)

    mitigated_4Pauli=np.abs(np.array([(np.average(data_4Pauli['energy_gap'][i*5+1:(i+1)*5])-idealValues)/idealValues for i in range(len(gammaList))]))

    un_4Pauli.append(unmitigated_4Pauli)
    mi_4Pauli.append(mitigated_4Pauli)

    plt.scatter(data['gamma'][0::randomSampleNum+1],unmitigated,color='r',marker='_',alpha=0.5)
    plt.scatter(data['gamma'][0::randomSampleNum+1],mitigated,color='black',marker='_',alpha=0.5)
    plt.scatter(data_4Pauli['gamma'][0::5],mitigated_4Pauli,color='blue',marker='_',alpha=0.5)

for i in range(len(gammaList)):
    vio1=plt.violinplot(np.array(un)[:,i],positions=[gammaList[i]],widths=(gammaList[1]-gammaList[0])/4*gammaList[i]/gammaList[0])
    vio2=plt.violinplot(np.array(mi)[:,i],positions=[gammaList[i]],widths=(gammaList[1]-gammaList[0])/4*gammaList[i]/gammaList[0])
    vio3=plt.violinplot(np.array(mi_4Pauli)[:,i],positions=[gammaList[i]],widths=(gammaList[1]-gammaList[0])/4*gammaList[i]/gammaList[0])
    for pc in vio1['bodies']:
        pc.set_color('red')
    for pc in vio2['bodies']:
        pc.set_color('black')
    for pc in vio3['bodies']:
        pc.set_color('blue')
    for partname in ('cbars','cmins','cmaxes'):
        vp = vio1[partname]
        vp.set_edgecolor('red')
        vp = vio2[partname]
        vp.set_edgecolor('black')
        vp = vio3[partname]
        vp.set_edgecolor('blue')

gammaList=np.insert(gammaList,0,0)

plt.plot(gammaList,np.insert(np.average(un,axis=0),0,0),label='Unmitigated',color='r',linestyle='dashed',linewidth=3)

plt.plot(gammaList,np.insert(np.average(mi,axis=0),0,0),label='Mitigated with\n100 Random Pauli',color='black',linestyle='dashed',linewidth=3)

plt.plot(gammaList,np.insert(np.average(mi_4Pauli,axis=0),0,0),label='Mitigated with\n$\{I^{\otimes n}, X^{\otimes n}, Y^{\otimes n}, Z^{\otimes n}\}$',color='blue',linestyle='dashed',linewidth=3)

plt.xlabel(r'Error strength $\gamma$')
plt.ylabel(r'Relative error $|\hat{E}_{ba}-E_{ba}|/E_{ba}$')
# plt.title('Relative error after averaging on random Pauli')

plt.legend(loc=0, prop={'size': 18})

plt.xscale('log')
plt.yscale('log')

# plt.xlim(0.05,0.25)

plt.savefig("./reshaping100.pdf",bbox_inches="tight",pad_inches=0.1)
plt.show()