# load data and consider a single feature (Economy..GDP.per.capita) and the output to be estimated (happiness)
import csv


def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName1)
    inputs1 = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedVariable = dataNames.index(inputVariabName2)
    inputs2 = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs1,inputs2,outputs