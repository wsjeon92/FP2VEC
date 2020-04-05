import numpy as np

index = [[],[],[],[],[],[],[],[],[],[],[],[]] # tox21 dataset has 12 targets
for j in range(12):
    f = open('tox21.csv', 'r')
    line = f.readline() # skip the first line
    dataX = []
    dataY = []
    for i, line in enumerate(f):
        line = line.rstrip().split(',')
        if not line[j]:
            continue
        index[j].append(i)
    f.close()
np.save('tox21_index', index)
