# Practical Work 2: State of the Art Methods for Denoising and Application to Eavesdropped Image Interpretation

## Install
Re-use the 'FCMR40' virtual machine and the setup as is at the end of PW1.  

1. Uninstall the packages of PW1:

		pip3 uninstall -r requirements_pw1.txt

2. Install the packages for PW2:

		pip3 install -r requirements_pw2.txt
		



## 2. Learning-Based Denoising
In this section, you will experience the prototyping of a learning-based denoiser. You will sucessively prepare the data, define the architecture of the neural network, define the optimization process, define the evaluation scheme at training time, launch the training and test the obtained model. As the available hardware is not powerful enough for the computation intensive training, you will check that your process runs well and prematurely stop the process. pre-trained model will be given for testing. 

1. Randomly split the dataset in train/val/test sets. 

		# List the files contained in the data directory
		filenames = listdir('data/in/bsd')
		print('Directory contains {} images!'.format(len(filenames)))

		# Shuffle the array of filenames to ensure random distribution into sets
		np.random.shuffle(filenames)

		# Separate in three sets
		train_ratio = 0.8
		n_train = int(len(filenames) * train_ratio)
		n_val = int((len(filenames) - n_train) /2)

		train_filenames = filenames[0:n_train]
		val_filenames = filenames[n_train: n_train + n_val]
		test_filenames = filenames[n_train + n_val: n_train + 2 * n_val]

		print('{} train samples, {} train samples, {} train samples'.format(len(train_filenames),
				                                                    len(val_filenames),
				                                                    len(test_filenames)))

		# Create directories
		makedirs('data/out/bsd_learning/train/ref', exist_ok=True)
		makedirs('data/out/bsd_learning/val/ref', exist_ok=True)
		makedirs('data/out/bsd_learning/test/ref', exist_ok=True)

		for i, set in enumerate(['train', 'val', 'test']):
		    for f in [train_filenames, val_filenames, test_filenames][i]:
			input_path = path.join('data/in/bsd', f)
			output_path = path.join('data/out/bsd_learning', set, 'ref', f)
			# print(input_path, output_path)
			copyfile(input_path, output_path)
			
2. 