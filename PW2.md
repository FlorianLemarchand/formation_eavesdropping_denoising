# Practical Work 2: State of the Art Methods for Denoising and Application to Eavesdropped Image Interpretation

## Install
Re-use the 'FCMR40' virtual machine and the setup as is at the end of PW1.  

1. Uninstall the packages of PW1:

		pip3 uninstall -r requirements_pw1.txt

2. Install the packages for PW2:

		pip3 install -r requirements_pw2.txt
		



## 1. Learning-Based Denoising
In this section, you will experience the prototyping of a learning-based denoiser. You will sucessively prepare the data, define the architecture of the neural network, define the optimization process, define the evaluation scheme at training time, launch the training and test the obtained model. As the available hardware is not powerful enough for the computation intensive training, you will check that your process runs well and prematurely stop the process. pre-trained model will be given for testing. 

1. Randomly split the dataset in train/val/test sets. Each subset contains a folder 'ref' that contains the clean samples and a folder 'in' that contains the noisy ones.  
_Complete and call the function 'make_learning_set' from 'utils_PW2.py'. You need to write the synthetic noising of an image with an AWGN with sigma 50._

			
3. Define the image generators that will feed the neural network with the previsouly prepared data. 
_Call 

2. Prepare a supervised learning dataset for the denoising problem using noising.
_Complete the function 'data_augmentation' in 'utils_PW2.py' with horizontal, vertical flip and 90° rotation._

2. Define the architecture of the neural network:
To begin you will use a pre-defined architecture.

		# Define the architecture
		n_channels = 1  # 1 for grayscale, 3 for RGB
		model = DnCNN(channels=n_channels, num_of_layers=17)
		
		# Display the architecture
		summary(model, (1, 64, 64), device='cpu')