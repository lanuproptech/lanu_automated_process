import csv

with open('../NPS_dataset/LR_POLY_VALI_03_AUG_2020.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    count = 0
    for row in csvReader:
        count += 1
        if count < 10:
            print(row)
