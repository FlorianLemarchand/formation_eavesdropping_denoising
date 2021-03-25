# Practical Work 2: Learning-Based Methods for Denoising and Application to Eavesdropped Image Enhancement

## Install
Re-use the _FCMR40_ virtual machine and the setup as is at the end of PW1.

1. Uninstall the packages of PW1:

		pip3 uninstall -r requirements_pw1.txt

2. Install the packages for PW2:

		pip3 install -r requirements_pw2.txt
	


## I. Learning-Based Denoising
In this section, you will experience the prototyping of supervised learning-based denoisers. You will sucessively prepare the data, define the architecture of the neural network, define the optimization process, define the evaluation scheme at training time, launch the training and test the obtained model. 

As the available hardware is not powerful enough for the computation intensive training, you will check that your process runs well and prematurely stop the process. Pre-trained model will be given for testing. 

**To avoid multiple duplicates of code, a 'question_solver' utility is used. Just fill the variable 'question' according to the number of the question your are working on to enable the related code parts. **

#### 1. Learning Dataset
The dataset has to be randomly split in training, validation and testing sets. Each subset contains a folder 'ref' that contains the reference samples and a folder 'in' that contains their noisy counterpart.

* Set _question=1_.

* Complete the function _make_learning_set_ from _utils_PW2.py_. You need to write the synthetic noising of an image with an AWGN with sigma 100.

* Run _pw2.py_.

* Have a look at the obtained dataset in 'data/out/bsd_learning'

#### 2. Data Generators	
During the training phase, data generators are responsible for feeding the model with pairs of corresponding noisy/clean images. Data generators have to be efficient to avoid being the bottleneck of the training loop.

* Set _question=2_.

* Complete the function _data_augmentation_ from _utils_PW2.py_. You have to program different transformations applied to input images. These transformations increase the variability of input samples during the training so that the model learns better.

* Run _pw2.py_. A batch from training and validation should be displayed. Check that the augmentation worked correctly.


#### 3. Model and Optimization Process	
Here are defined the parameters of the model architecture, and those of the optimization process.

* Here, the CNN architecture is _DnCNN_, for grayscale images (_n_channels=1_), composed of 17 layers (_num_of_layers=17_). 

* The loss function is the MSE (Mean Square Error).

* The optimizer is _Adam_ with default parameters and learning rate 0.001 (_lr=0.001_).

* The training is set to last 2000 epochs. An epoch is most of the time defined as the training over all the image pairs of the dataset. 

* Set _question=3_
* Run _pw2.py_. The architecture of the model is displayed on the console as well as useful information like the number of parameters it contains.

#### 4. Training Loop
The training loops over the entiere dataset _n_epochs_ times. The dataset is divided in batches of images. Each batch is forward passed through the model. The loss is computed between the output of the model and the reference batch. The loss is then backward passed through the model. An optimization step is done. 
At the end of an epoch, the validation set is passed through the current model and the loss is computed to check that the model generalizes well on unseen data. Also, the score on validation set drives the selection of the best model to be saved. 

Different metrics are logged during training and validation so that it can be monitored. An unique log directory is created and its location displayed on the console. 

* Set _question=4_.

* Run _pw2.py_. Check that everything goes well. As explained before, the learning is really slow, stop it and you will use pre-trained models.

* Pre-trained logs and models have been placed in 'logs'. To display the logs, open a terminal from this directory. Connect to the virtual environnement and run:
		
		tensorboard --logdir='.'
		
* For now, look at _dncnn_sigma50_ and _dncnn_sigma100_ that are trained for denoising AWGN of sigma 50 and 100 respectively.


#### 5. Testing
Now that you have trained models, you need to evaluate them on unseen data. 

* Set _question=5_.

* Run _pw2.py_. The testing is ran, metrics are displayed for each image as well as the averages values on the test set. Keep trace of the average metrics as we will use them later on. Noisy/denoised/reference image triplets are stored in _data/out/out_dncnn_.


## II. Towards Eavesdropped Data Enhancement

You saw on part I how to design a denoiser for AWGN noise. You will now do the same for an eavesdropping dataset. You will compare the results with those obtained using the basic filtering techniques of PW1. 

#### 6. Eavesdropped Images Enhancement
The model and training parameters are almost the same as those of AWGN part. The only difference is the data used to train the model. As earlier, a pre-trained model is used for testing. The data to be used is in _data/in/intercept_. 

* Set _question=6_.

* Have a look at _make_learning_set_intercept_ as it differs from previous code. The noisy images are not anymore generated from references since their are real-world samples.

* Run _pw2.py_. Have a look and keep trace of the average metrics. How does it perform in PSNR compared to AWGN( Question 5)? Look at the resulting images in _data/out/out_dncnn17_intercept_.

* Change the model to DnCNN with 20 layers. You need to change _output_dir_, _saved_model_path_ and the parameters of the DnCNN model. Run _pw2.py_ and compare the results with the model with 17 layers. You can do that comparison with the pre-trained _RedNet30_ available in _logs/natural_intercept_rednet30_. If you have time, you can also test the model trained for AWGN on the eavedropped samples.

* Training logs and weights of a DnCNN model that has been trained without data augmentation is available at _logs/natural_intercept_dncnn_20_no_augmentation_. You can check the tensorboard display of the logs and observe what is overfitting.

#### 7. Comparison Table with Basic and Advanced Filtering Denoising
During PW1 and PW2, you have implemented and used different denoising techniques. As the computation is long on the available hardware, here is a comparison table of the PSNR obtained using the different methods you have used during the PWs. 

| Dataset/Algorithm | No Filtering | Mean Filter 3x3 | Mean Filter 5x5 | Median Filter 5x5  | FFT + Thresholding | BM3D  Sigma 50 | DnCNN17 | DnCNN20 | DnCNN20 No  Augmentation |
|:-----------------:|:------------:|:---------------:|:---------------:|:------------------:|:------------------:|:--------------:|:-------:|:-------:|:------------------------:|
|    Eavesdropped   | 10.37        |      10.89      |      10.93      |        10.83       |        10.91       | 10.99          |  14.39  |  16.49  |           11.31          |

Some comparison images are available at _data/out/comparison_methods_. The images are arranged as follows:

|   Noisy   | Mean 3x3 | Mean 5x5 | Median 5x5 |        FFT + Threshold       |
|:---------:|:--------:|:--------:|:----------:|:----------------------------:|
| Reference |   BM3D   |  DnCNN17 |   DnCNN20  | DnCNN20 No Data Augmentation |

####  8. Optional: Inference Optimization
Ensemble inference is the idea of averaging different transformed versions of a noisy image passed through a model to enhance the results. 

First the input image is transform using invertible transforms like rotations or flips. The transformed images are passed through the model. The filtered transformed images are transformed back to their original configuration and the result of the ensemble inference is the average of all the transformed back images. 

The underlying idea is that each part of the network is not trained the same and passing transforms enables the resulting average image to benefit from the strenghs of all parts of the network. 


* Set _question=8_.

* Have a look at the _ensemble_inference_ function and its usage in the code corresponding to question 8. 

* Run _pw2.py_. The performance enhancement obtained is not very large compared to the computation increase (several images to pass through the model). But still, it is an improvement! 
