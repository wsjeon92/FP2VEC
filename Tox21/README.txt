### file description

tox21-multitask.py
	main code for the prediction model (FP2VEC and QSAR model)
tox21.csv
	tox21 dataset
tox21_index.npy
	preprocessed file for running code. a set of indexes for input data
tox21_inputgen.py
	input generator for file 'tox21-multitask.py'.

instruction
	first, download four files in the same directory. 
	second, run the file 'tox21_inputgen.py' to prepare input data.
	third, run the file 'tox21-multitask.py' to train the model and see the result. 
