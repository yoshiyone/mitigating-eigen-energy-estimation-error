from matplotlib import pyplot as plt
import numpy as np
from exact_diagonalization import eigenSolver
from models import ringModel

'''
Load random a,b.
'''

import csv

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
fRE=[]
sRE=[]

plt.figure(figsize=(14,6))
plt.rcParams.update({'font.size': 16})
bwith = 2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

for randomNums in randomStatesList[0:100]:
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
    f_RE=np.abs((data['f_RE']-energyGap)/energyGap)
    s_RE=np.abs((data['s_RE']-energyGap)/energyGap)

    un.append(unmitigated)
    f.append(first)
    s.append(second)
    fRE.append(f_RE)
    sRE.append(s_RE)

    plt.scatter(data['gamma'],unmitigated,c='r',alpha=0.5,marker='_')
    plt.scatter(data['gamma'],first,c='black',alpha=0.5,marker='_')
    plt.scatter(data['gamma'],second,c='b',alpha=0.5,marker='_')
    plt.scatter(data['gamma'],f_RE,c='blueviolet',alpha=0.5,marker='_')
    plt.scatter(data['gamma'],s_RE,c='green',alpha=0.5,marker='_')

for i in range(len(data["gamma"])):
    vio1=plt.violinplot(np.array(un)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    vio2=plt.violinplot(np.array(f)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    vio3=plt.violinplot(np.array(s)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    vio4=plt.violinplot(np.array(fRE)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    vio5=plt.violinplot(np.array(sRE)[:,i],positions=[data["gamma"][i]],widths=(data["gamma"][1]-data["gamma"][0])/4*data["gamma"][i]/data["gamma"][0])
    for pc in vio1['bodies']:
        pc.set_color('red')
    for pc in vio2['bodies']:
        pc.set_color('black')
    for pc in vio3['bodies']:
        pc.set_color('blue')
    for pc in vio4['bodies']:
        pc.set_color('blueviolet')
    for pc in vio5['bodies']:
        pc.set_color('green')
    for partname in ('cbars','cmins','cmaxes'):
        vp = vio1[partname]
        vp.set_edgecolor('red')
        vp = vio2[partname]
        vp.set_edgecolor('black')
        vp = vio3[partname]
        vp.set_edgecolor('blue')
        vp = vio4[partname]
        vp.set_edgecolor('blueviolet')
        vp = vio5[partname]
        vp.set_edgecolor('green')

un_avg=np.average(un,axis=0)
f_avg=np.average(f,axis=0)
s_avg=np.average(s,axis=0)
fRE_avg=np.average(fRE,axis=0)
sRE_avg=np.average(sRE,axis=0)

gammaList=np.insert(data['gamma'],0,0)
un_avg=np.insert(un_avg,0,0)
f_avg=np.insert(f_avg,0,0)
s_avg=np.insert(s_avg,0,0)
fRE_avg=np.insert(fRE_avg,0,0)
sRE_avg=np.insert(sRE_avg,0,0)

plt.plot(gammaList,un_avg,label='Unmitigated',c='r',linestyle='dashed',linewidth=3)
plt.plot(gammaList,f_avg,label="First order",c='black',linestyle='dashed',linewidth=3)
plt.plot(gammaList,s_avg,label="Second order",c='b',linestyle='dashed',linewidth=3)
plt.plot(gammaList,fRE_avg,label="One factor Richardson",c='blueviolet',linestyle='dashed',linewidth=3)
plt.plot(gammaList,sRE_avg,label="Two factors Richardson",c='green',linestyle='dashed',linewidth=3)

plt.xlabel(r'Error strength $\gamma$')
plt.ylabel(r'Relative error $|\hat{E}_{ba}-E_{ba}|/E_{ba}$')

plt.legend(loc=0)

plt.xscale('log')
plt.yscale('log')

# plt.xlim(0,0.5)
# plt.ylim(1e-6,1e-2)

plt.savefig("./rescaling.pdf",bbox_inches="tight",pad_inches=0.1)
plt.show()