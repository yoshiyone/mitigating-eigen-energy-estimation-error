import random

def generateRandom2Numbers(n:int):
    '''
    Return 2 different random numbers from 0 to n-1.
    '''
    # Generate two different random integers within the range (0, n-1)
    random_int1 = random.randint(0, n-1)
    random_int2 = random.randint(0, n-1)

    # Ensure the two integers are different
    while random_int2 == random_int1:
        random_int2 = random.randint(0, n-1)

    return random_int1, random_int2

if __name__ == "__main__":
    n=6
    random_2_nums=[]

    for i in range(100):
        random2Numbers=generateRandom2Numbers(2**n)
        while random2Numbers in random_2_nums:
            random2Numbers=generateRandom2Numbers(2**n)
        random_2_nums.append(random2Numbers)

    import csv

    # Specify the file path
    csv_file_path = "100Random2Numbers.csv"

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the data to the CSV file
        writer.writerows(random_2_nums)

    print(f"Data has been saved to {csv_file_path}")
