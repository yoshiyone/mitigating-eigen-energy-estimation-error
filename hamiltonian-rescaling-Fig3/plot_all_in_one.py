from matplotlib import pyplot as plt
import numpy as np
from exact_diagonalization import eigenSolver
from models import ringModel

'''
Load random tStates.
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

n=6
hamiltonian=ringModel(4,1,4,n)
# print(hamiltonian)
eigenvalues,eigenstates=eigenSolver(hamiltonian,n)


'''
Load data.
'''

un=[]
f=[]
s=[]


plt.figure(figsize=(8,8))

plt.rcParams.update({'font.size': 22})
bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

for randomNums in randomtStatesList[0:100]:
    a=int(randomNums[0])
    b=int(randomNums[1])
    # Read the data
    filename = 'data/'+str(a)+"_"+str(b)+'.csv'
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None)

    energyGap=eigenvalues[b]-eigenvalues[a]

    # if np.abs(energyGap)<10:
    #     continue

    unmitigated=np.abs((data['noisy']-energyGap)/energyGap)
    first=np.abs((data['first_order']-energyGap)/energyGap)
    second=np.abs((data['second_order']-energyGap)/energyGap)

    un.append(unmitigated)
    f.append(first)
    s.append(second)

    plt.scatter(data['gamma'],unmitigated,c='r',alpha=0.5,marker='_')
    plt.scatter(data['gamma'],first,c='black',alpha=0.5,marker='_')
    plt.scatter(data['gamma'],second,c='b',alpha=0.5,marker='_')
    

for i in range(len(data["gamma"])):
    vio1=plt.violinplot(np.array(un)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    vio2=plt.violinplot(np.array(f)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    vio3=plt.violinplot(np.array(s)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
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

un_avg=np.average(un,axis=0)
f_avg=np.average(f,axis=0)
s_avg=np.average(s,axis=0)

gammaList=np.insert(data['gamma'],0,0)
un_avg=np.insert(un_avg,0,0)
f_avg=np.insert(f_avg,0,0)
s_avg=np.insert(s_avg,0,0)

plt.plot(gammaList,un_avg,label='Unmitigated',c='r',linestyle='dashed',linewidth=3)

plt.plot(gammaList,f_avg,label="First order",c='black',linestyle='dashed',linewidth=3)

plt.plot(gammaList,s_avg,label="Second order",c='b',linestyle='dashed',linewidth=3)

plt.xlabel(r'Error strength $\gamma$')
plt.ylabel(r'Relative error $|\hat{E}_{ba}-E_{ba}|/E_{ba}$')

plt.legend(loc=0, prop={'size': 18})

plt.xscale('log')
plt.yscale('log')

# plt.xlim(0,0.5)
# plt.ylim(1e-6,1e-2)

plt.savefig("./rescaling.pdf",bbox_inches="tight",pad_inches=0.1)
plt.show()