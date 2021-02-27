#!/usr/bin/python3

import csv

outputfile1='diabetes_input.csv'
outputfile2='diabetes_output.csv'

with open(outputfile1, mode='w') as csv_input, open(outputfile2, mode='w') as csv_output, open('diabetes.csv', mode='r') as csv_in:
    
    fieldnames1 = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    fieldnames2 = ['Outcome']

    writer1 = csv.DictWriter(csv_input, fieldnames=fieldnames1)
    writer2 = csv.DictWriter(csv_output, fieldnames=fieldnames2)
    csv_reader = csv.reader(csv_in, delimiter=',')
    headers = next(csv_reader) # skips headers!

    for row in csv_reader:
        if(len(row) != 9):
            continue

        datum1 = { 'Pregnancies' : str(row[0]),'Glucose' : str(row[1]),'BloodPressure' : str(row[2]),'SkinThickness' : str(row[3]),'Insulin' : str(row[4]),'BMI' : str(row[5]),'DiabetesPedigreeFunction' : str(row[6]),'Age' : str(row[7]) }
        datum2 = { 'Outcome' : str(row[8]) }

        writer1.writerow(datum1)
        writer2.writerow(datum2)
        


