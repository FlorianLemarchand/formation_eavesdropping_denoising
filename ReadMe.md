# Towards Eavesdropped Image Denoising

Related Lecture Slides --> [link](https://florianlemarchand.github.io/ressources/pdfs/2020-06-11-Towards_Eavesdropped_image_denoising.pdf)

## Practical Work 1 -- PW1 -- Basics of Image Processing and Denoising

Subject --> [PW1.html](./PW1.html) 

### 1. Image Manipulations
In this section, you will use some basics code lines to manipulate images.

### 2. Synthetic Image Noising and Quality Assessement
In this section you will experience synthetic image noising and quality assessment. For that end, the scikit-image python package is used.

### 3. Denoising Using Basic Filtering
In this section you will take a step towards denoising. Basic filtering will be used and the quality measured and observed to highlight the limits of such basic processings. 

### 4. Advanced Filtering
In this section, you will experiment an example of advanced filtering. This method, while more efficient, is also more dedicated. You will see that it does not adapt well to noise distributions slightly different from what the method is made for. 

## Practical Work 2 -- PW2 -- Learning-Based Methods for Denoising and Application to Eavesdropped Image Enhancement 
Subject --> [PW2.html](./PW2.html) 

### I. Prototyping a Learning-Based Denoising Method
In this section, you will experience the prototyping of a learning based denoiser. The PyTorch Framework is used. To begin with a simple case study, the target is the denoising of an Additive White Gaussian Noise (AWGN) with standard deviation 100. 
#### 1. Learning Dataset 
#### 2. Data Generators 
#### 3. Model and Optimization Process 
#### 4. Training Loop 
#### 5. Testing

### II. Towards Eavesdropped Data Enhancement 
After experiencing the denoising of a synthetic AWGN, you will now target real-world noisy images, namely eavesdropped images.
#### 6. Eavesdropped Images Enhancement 
#### 7. Comparison Table with Basic and Advanced Filtering Denoising 
#### 8. Optional: Inference Optimization 
