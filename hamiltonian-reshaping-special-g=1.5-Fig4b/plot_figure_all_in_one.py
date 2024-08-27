from matplotlib import pyplot as plt
import numpy as np
from exact_diagonalization import eigenSolver
from models import transversalXYZIsingModel
import csv

n=6
hamiltonian=transversalXYZIsingModel(-1.5,0,0,-1,-1,0,n)
# print(hamiltonian)
eigenvalues,eigenstates=eigenSolver(hamiltonian,n)

'''
Load random numbers.
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
Load data.
'''

# kappa=gamma*|deltaE|, ham_err_strength=gamma*beta*|deltaE|
# maxGamma=0.02
# gammaNums=20
# gammaList=np.array([maxGamma*(i+1)/gammaNums for i in range(gammaNums)])
gammaList=np.array([1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,3e-2,5e-2,1e-1])

un=[]
sp_mi=[]

plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 22})
bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
# it=1

randomSampleNum=2

for randomNums in randomStatesList[0:10]:
    # print("Iteration ",it)
    a=int(randomNums[0])
    b=int(randomNums[1])
    # print("Random a b:",a,b)
    idealValues=eigenvalues[b]-eigenvalues[a]
    # print("Exact diagonalization result:",idealValues)

    # Read the data
    filename = 'data/'+str(a)+"_"+str(b)+"_special"+'.csv'
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None)

    unmitigated=np.abs((data['energy_gap'][0::randomSampleNum+1]-idealValues)/idealValues)

    specialMitigated=np.abs(np.array([(np.average(data['energy_gap'][i*(randomSampleNum+1)+1:(i+1)*(randomSampleNum+1)])-idealValues)/idealValues for i in range(len(gammaList))]))

    un.append(unmitigated)
    sp_mi.append(specialMitigated)

    plt.scatter(data['gamma'][0::randomSampleNum+1],unmitigated,color='r',marker='_',alpha=0.5)
    plt.scatter(data['gamma'][0::randomSampleNum+1],specialMitigated,color='black',marker='_',alpha=0.5)

for i in range(len(gammaList)):
    vio1=plt.violinplot(np.array(un)[:,i],positions=[gammaList[i]],widths=(gammaList[1]-gammaList[0])/4*gammaList[i]/gammaList[0])
    vio2=plt.violinplot(np.array(sp_mi)[:,i],positions=[gammaList[i]],widths=(gammaList[1]-gammaList[0])/4*gammaList[i]/gammaList[0])
    for pc in vio1['bodies']:
        pc.set_color('red')
    for pc in vio2['bodies']:
        pc.set_color('black')
    for partname in ('cbars','cmins','cmaxes'):
        vp = vio1[partname]
        vp.set_edgecolor('red')
        vp = vio2[partname]
        vp.set_edgecolor('black')

gammaList=np.insert(gammaList,0,0)

plt.plot(gammaList,np.insert(np.average(un,axis=0),0,0),label='Unmitigated',color='r',linestyle='dashed',linewidth=3)

plt.plot(gammaList,np.insert(np.average(sp_mi,axis=0),0,0),label='Mitigated with\n$\{I^{\otimes n}, X^{\otimes n}\}$',color='black',linestyle='dashed',linewidth=3)

plt.xlabel(r'Error strength $\gamma$')
plt.ylabel(r'Relative error $|\hat{E}_{ba}-E_{ba}|/E_{ba}$')
plt.title(r'$g=1.5$',y=0.9)

plt.legend(loc=4, prop={'size': 18})

plt.xscale('log')
plt.yscale('log')


plt.savefig("./specialReshaping-10-10.pdf",bbox_inches="tight",pad_inches=0.1)
plt.show()