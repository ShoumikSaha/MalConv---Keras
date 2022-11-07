import csv

with open('data_label.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',')

    #employee_writer.writerow(['John Smith', 'Accounting', 'November'])
    #employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
    with open('labels/benign.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif(row[0] == '0'):
                print("name:", row[1])
                print("label:", row[2])
                file_name = 'DikeDataset/files/benign/' + row[1] + '.exe'
                label = 0
                employee_writer.writerow([file_name, label])


    with open('labels/malware.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif(row[0] == '0'):
                print("name:", row[1])
                print("label:", row[2])
                file_name = 'DikeDataset/files/malware/' + row[1] + '.exe'
                label = 1
                employee_writer.writerow([file_name, label])
