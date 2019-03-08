import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

# generate x

trfile = open('tox21.csv', 'r')
line = trfile.readline()
dataX = []
for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[13])
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp = np.array(fp)
    dataX.append(fp)
trfile.close()

dataX = np.array(dataX)
np.save('tox21_fp', dataX)

# generate y

dataY_concat = []
for j in range(12):
    trfile = open('tox21.csv', 'r')
    line = trfile.readline()
    dataY = []
    for i, line in enumerate(trfile):
        line = line.rstrip().split(',')
        if not line[j]:
            continue    
        val = float(line[j])
        dataY.append(val)
    dataY=np.array(dataY)
    dataY_concat.append(dataY)
    trfile.close()

np.save('tox21_Y', dataY_concat)
