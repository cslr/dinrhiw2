#!/usr/bin/python3

import csv

outputfile='titanic_pred.csv'

with open(outputfile, mode='w') as csv_file, open('titanic_test_output.out', mode='r') as csv_in:
    fieldnames = ['PassengerId', 'Survived']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_reader = csv.reader(csv_in, delimiter=',')

    writer.writeheader()
    passengerId = 892
    lines = 0
    
    for row in csv_reader:
        if(len(row) == 1):
            value = int(round(float(row[0])))
        else:
            value = 0;
            print(f"Error in line {lines} of input.");
            continue

        datum = {'PassengerId': str(passengerId), 'Survived': str(value)}
        
        writer.writerow(datum)
        passengerId = passengerId + 1
        lines = lines + 1
    
    print(f"Created {lines} data lines output file '{outputfile}'.")



    
    
