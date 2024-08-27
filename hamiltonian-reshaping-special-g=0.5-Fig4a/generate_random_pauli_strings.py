import random

def generateRandomPauli(n):
    '''
    Return a length n random Pauli string.
    
    Example
    ----------
    >>> generateRandomPauli(3)
    'XYZ'
    '''
    randomPauliString=''
    for i in range(n):
        randomNum=random.randint(0,3)
        if randomNum == 0:
            randomPauliString+='I'
        elif randomNum == 1:
            randomPauliString+='X'
        elif randomNum == 2:
            randomPauliString+='Y'
        elif randomNum == 3:
            randomPauliString+='Z'
        else:
            print("Random process error.")
            quit(1)
    
    return randomPauliString

n=6
randomPauli=[]

for i in range(100):
    pauliString=generateRandomPauli(n)
    # print(pauliString)
    randomPauli.append(pauliString)

import csv

# Specify the file path
csv_file_path = "100randomPauli.csv"

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the data to the CSV file
    writer.writerows([randomPauli])

print(f"Data has been saved to {csv_file_path}")
