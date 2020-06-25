#!/usr/bin/python3

import csv

outputfile='mnist_digit_pred.csv'

with open(outputfile, mode='w') as csv_out, open('mnist_digit_test_output.out', mode='r') as csv_in:
    fieldnames = ['ImageId', 'Label']

    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    reader = csv.reader(csv_in, delimiter=' ')

    writer.writeheader()
    imageId = 1
    lines = 0

    for row in reader:
        label = 0
        maxactivity = float(row[0])

        # selects the winning label for each pic
        for i in range(len(row)):
            a = float(row[i])
            
            if a >= maxactivity:
                maxactivity = a
                label = i


        datum = {'ImageId' : str(imageId), 'Label' : str(label)}

        writer.writerow(datum)
        imageId = imageId + 1
        lines = lines + 1

    print(f"Processed {lines} data lines to output file '{outputfile}'.")

