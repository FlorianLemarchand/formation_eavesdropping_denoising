# Practical Work 1: Basics of Image Processing and Denoising

## Install

1. Open 'FCMR40' virtual machine using VirtualBox. Password: fcmr40 

2. Get the code repository

    Open a terminal:
    Ctrl + Alt + T

    `cd ~/Desktop`

    `git clone https://github.com/FlorianLemarchand/formation_eavesdropping_denoising.git`
    
    The digital version of this practical work guidance is available at the repository root ('PW1.html'). Digital version will be useful to copy paste snippets of code.
    
3. A little update and new installs:

    `sudo sed -i -re 's/([a-z]{2}\.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list`

    `sudo apt-get update && sudo apt-get dist-upgrade`
    This command can be long (few minutes). You can read the following of the subject during that time.

    `sudo apt install virtualenv`
   
    `sudo apt install libtiff-dev`
    
    `sudo apt install spyder3`



4. Setup the python environment

    `cd formation_eavesdropping_denoising`
       
    * Create a virtual environement:

    `virtualenv -p python3 venv`

    * Connect to this virtual environement:

    `source venv/bin/activate`

    The command line should now look like:

    `(venv) toto:~$`

    From now, each python package will be installed in this virtual environement. To exit the virtual environement:
    
    `deactivate`

5. Install the required packages:

    `pip3 install -r requirements_pw1.txt`
    
    * You can display the installed packages and their versions using:
    
    `pip3 list`


## 1. Image Manipulations
In this section, you will use some basics code lines to manipulate images.

For simplicty, use Spyder IDE to write and run the code. You need to tell Spyder to run under the venv. From the console with venv activated, run:

`spyder3`

When Spyder opens, go to 'Tools --> Preferences --> Python Interpreter', check 'Use the following Python interpreter' and select '/home/fcmr40/Desktop/formation_eavesdropping_denoising/venv/bin/python3'.

Now, you can open the script 'pw1.py' and start editing it. The run shortcut is 'F5'. Be sure to select the 'Python Console' tab on right bottom of the IDE before pressing 'F5' to run. If everything is fine, a success message should be printed!

Through this PW you will update the script with given snippets and code you will develop. New snippets often use outputs from previous ones. Nevertheless, do not hesitate to clean regularily the script to avoid the console and the display to be flooded. Block comment shortcut : 'Ctrl + 4', Block uncomment shortcut : 'Ctrl + 5'.


   1. Load images 'im1.jpg' and 'im2.jpg' from 'data' directory.
   
        im1 = imread('data/in/im1.jpg')
   
   2. Print their shapes, means, standard deviations (std), mins ans maxs: 
        
        # Print the shapes, means, std, mins, maxs of the images
        print('Im1 ==> Shape: {} / Mean: {} / Std: {} / Min: {} / Max: {}'.format(im1.shape,
                                                                                  np.round(np.mean(im1), 2),
                                                                                  np.round(np.std(im1), 2),
                                                                                  np.min(im1),
                                                                                  np.max(im1)))
        
        print('Im2 ==> Shape: {} / Mean: {} / Std: {} / Min: {} / Max: {}'.format(im1.shape,
                                                                                  np.round(np.mean(im2), 2),
                                                                                  np.round(np.std(im2), 2),
                                                                                  np.min(im2),
                                                                                  np.max(im2)))

      *  _Which dimensions are the height, width and number of channels? Note that this order is not the same for all libraries!_

       * _Which of the two images has the highest average luminance? Which of the two images has the highest color deviation?_

   3. Save/Display the images:
    
        # Save the patches
        makedirs('data/out', exist_ok=True)
        imsave('data/out/im1.jpg', im1)
        imsave('data/out/im2.jpg', im2)
        
        # Display the images
        plt.subplot(2,1,1)
        plt.title('Image 1')
        plt.imshow(im1)
        plt.subplot(2,1,2)
        plt.title('Image 2')
        plt.imshow(im2)
        plt.show()   
        
   	You may change image plotting size by running this code before the preceding commands:

        plt.figure(figsize=(15,15))

   	* _Check that the images have well been saved in './data/out/'._
        
      
   4. Play with Python indexing to manipulate images and display them:
                
        # Crop patchs
        crop_1 = im1[0:150, 0:150, :]
        crop_2 = im2[0:100, 150:250, :]
      
        # Flip the image
        vflip = im1[::-1, :, :]
        hflip = im1[:, ::-1, :]
      
        # Rotate the image
        rota90 = vflip.transpose(1, 0, 2)
        rota180 = im1[::-1, ::-1, :]
        rota270 = hflip.transpose(1, 0, 2) 
      
        # Down-Sample the image
        stride4 = im1[0:-1:4, 0:-1:4, :]
        stride8 = im1[0:-1:8, 0:-1:8, :]    
	* _Print/display the resulting shapes and images to understand the commands._ 
          
   5. Compute horizontal and vertical gradients
        
        # Transform to grayscale
        im1_gray = rgb2gray(im1)      
        print('Grayscale image shape:', im1_gray.shape)
        
        # Compute horizontal and vertical gradients
        vgrad = im1_gray[0:-2, :] - im1_gray[1:-1, :]
        hgrad = im1_gray[:, 0:-2] - im1_gray[:, 1:-1]
        
	* _Print/display the resulting shapes and images to understand the commands._ 

## 2. Synthetic Image Noising and Quality Assessement

In this section you will experience synthetic image noising and quality assessment. For that end, the scikit-image python package is used.

1. Generate a map of white Gaussian noise with standard deviation 100. Display its distribution through an histogram. Add the noise to im1. 
       
        sigma = 100
        # Cast the image and parameters to float : each value coded on 64-bit float with value in [0,1]
        im1_gray = img_as_float(im1_gray)      
        sigma_float = sigma/255.
        variance = np.square(sigma_float) # variance = np.square(std)
        
        # generate a noisy value for each pixel of im1_gray
        noise_map = np.random.normal(0., sigma_float,  im1_gray.shape)      
              
        # Add noise to image
        im_noise = im1_gray + noise_map
        
        # Avoid values out of the [0.,1.] range
        im_noise = np.clip(im_noise, 0., 1.)
        
        # Display the value histogram of the noise map, the original and noisy images
        vec_noise = np.reshape(noise_map, noise_map.size)  # Vectorize noise_map
        plt.subplot(2,2,1)
        plt.imshow(im1_gray, cmap='gray')
        plt.subplot(2,2,2)
        plt.hist(vec_noise, bins=1000)
        plt.subplot(2,2,3)
        plt.imshow(im_noise, cmap='gray')
        plt.show()
      
 2. Add a display of the noise. Use the previous methods to display the histograms of im1 and of noised im1. What can we say about these histograms? You may remove the extreme pixel values from the noised image histogram plot. Also draw the gradients of noised im1 and its distribution. What can we say about these histograms?
        
 3. Use a library to do the same noising with only one line:
   
        im_noise_lib = random_noise(im1_gray, 'gaussian', mean=0., seed=0, var=variance, clip=True)
       * _Display 'im_noise_lib' and its histogram._ 
       
   4. Quality assessment of the noisy image. Evaluate PSNR, MSE and SSIM metrics of the noisy im1 with respect to its reference. In the following, you can use  the given _print\_psnr\_ssim_ function:
   
		# Measure the quality of the noisy images with respect to the clean image
		psnr = compare_psnr(im_noise, im1_gray)
		mse = compare_mse(im_noise, im1_gray)
		ssim = compare_ssim(im_noise, im1_gray)
		print('im_noise: PSNR: {} / MSE: {} / SSIM: {}'.format(psnr, mse, ssim))
       
   5. Display different intensities of Gaussian noise applied to im1:
       
        # Noise im1 with different sigmas       
        sigmas = range(20, 161, 20)        
        ncol = 3
        nrow = int((len(sigmas) + 2) / ncol)
        
        print('{} image to display on {} columns and {} rows'.format(len(sigmas) + 1, ncol, nrow))
        
        plt.subplot(nrow, ncol, 1)
        psnr = np.round(compare_psnr(im1_gray, im1_gray), 2)
        ssim = np.round(compare_ssim(im1_gray, im1_gray), 2)
        plt.title('Sigma: {} \nPSNR:{}/SSIM:{}'.format(0, psnr, ssim))
        plt.imshow(im1_gray, cmap='gray')
        for i, sigma in enumerate(sigmas):
            var = np.square(sigma / 255.)  # Compute variance from sigma
            im_noise = random_noise(im1_gray, 'gaussian', mean=0, var=var, seed=0, clip=True)  # Noise the image
            psnr = np.round(compare_psnr(im_noise, im1_gray), 2)  # Compute PNSR
            ssim = np.round(compare_ssim(im_noise, im1_gray), 2)  # Compute SSIM
            plt.subplot(nrow, ncol, i + 2)
            plt.title('Sigma: {} \n{}/{}'.format(sigma, psnr, ssim))
            plt.imshow(im_noise, cmap='gray')
        
        plt.show()
      
   6. Try other noise types in the 'random_noise' function [(Documentation)](https://scikit-image.org/docs/0.13.x/api/skimage.util.html#skimage.util.random_noise). Measure the resulting PSNR and SSIM.
   
   7. Apply sequentially two different noise types and measure the metrics. 
   
## 3. Denoising using basic filtering

In this section you will take a step towards denoising. Basic low-pass filtering will be used and the quality measured and observed to highlight the limits of such basic processings. 
   
   1. Filtering Framework

      	# Measure noisy PSNR/SSIM
      	print_psnr_ssim(im_noise_lib, im1_gray, 'Noisy')
      
      	# Filter loop function
      	def filter_loop(noisy_image, kernel, padding_type='zeros'):
			height, width = noisy_image.shape
			kernel_size = kernel.shape[0]
		    
			# Initialize the output array
			output = np.zeros(noisy_image.shape)
		    
			# Generate padding
			padding_size = int(kernel_size/2)
		    
			if padding_type is 'zeros':
			    padded_input = np.zeros((height + 2 * padding_size, width + 2 * padding_size))
			elif padding_type is 'ones':
			    padded_input = np.ones((height + 2 * padding_size, width + 2 * padding_size))
		    
			padded_input[padding_size:padding_size+height, padding_size:padding_size+width] = noisy_image
		    
			# Loop over the image
			for i in range(0, height):
			    for j in range(0, width):
				output[i, j] = np.sum(np.multiply(kernel, padded_input[i:i+kernel_size, j:j+kernel_size]))
		    
			return output
      
      	# Use function to mean filter
      	hand_denoised = filter_loop(im_noise_lib, np.ones((3, 3)) / 9.)
      
      	# Measure denoised PSNR/SSIM
      	print_psnr_ssim(hand_denoised, im1_gray, 'Mean Denoised')
      	
   
   2. Easier Filtering using convolve tool: 

	      # Measure noisy PSNR/SSIM
	      print_psnr_ssim(im_noise_lib, im1_gray, 'Noisy')
	      
	      # Filter
	      def mean_filter(input, kernel_size):
			kernel = np.ones((kernel_size, kernel_size))
			output = 1 / np.square(kernel_size) * convolve(input, kernel)
			return output
	      mean_denoised = mean_filter(im_noise_lib, 3)
	      
	      # Measure denoised PSNR/SSIM
	      print_psnr_ssim(gauss_denoised, im1_gray, 'Mean Denoised using Convolve')
* _Try different filter kernel sizes._ 
	   
3. Use scipy.ndimage library to try other filters: maximum_filter, minimum_filter, median_filter.
_In your opinion, which one gives the better denoising (objectively and subjectively)?_
	
4. Low Pass Transform Denoising
	
		# Transform the image and shift the result
		fft_clean = fftshift(fft2(im1_gray))
		fft_noisy = fftshift(fft2(im_noise_lib))

		shape = fft_noisy.shape
		middle_y, middle_x = int((shape[0] + 1.)/2.), int((shape[1]+1.)/2.)


		# Keep only a part of the coefficients
		percentage_keep = 0.2
		fft_thresh = fft_noisy.copy()  # init an output image
		fft_thresh[:, :] = 0.  # Set to 0. +0.j
		half_y_keep = int(((int(percentage_keep * shape[0]))+1)/2.)  # Compute the size of the window to keep
		half_x_keep = int(((int(percentage_keep * shape[1]))+1)/2.)
		fft_thresh[middle_y - half_y_keep: middle_y + half_y_keep, middle_x - half_x_keep: middle_x + half_x_keep] = \
		  fft_noisy[middle_y - half_y_keep: middle_y + half_y_keep, middle_x - half_x_keep: middle_x + half_x_keep]

		# Inverse transform
		im_denoised = ifft2(ifftshift(fft_thresh)).real
		im_denoised = np.clip(im_denoised, 0., 1.)  # Ensure the values stay in [0., 1.]

		print_psnr_ssim(im_denoised, im1_gray, 'Denoised')

		# Display
		if True:
			plt.subplot(3, 2, 1)
			plt.imshow(im1_gray, cmap='gray')
			plt.title('Clean Image')

			plt.subplot(3, 2, 2)
			plt.imshow(np.abs(fft_clean), norm=LogNorm())
			plt.title('Clean Spectrum')
			plt.colorbar()

			plt.subplot(3, 2, 3)
			plt.imshow(im_noise_lib, cmap='gray')
			plt.title('Noisy Image')

			plt.subplot(3, 2, 4)
			plt.imshow(np.abs(fft_noisy), norm=LogNorm())
			plt.title('Noisy Spectrum')
			plt.colorbar()

			plt.subplot(3, 2, 5)
			plt.imshow(im_denoised, cmap='gray')
			plt.title('Denoised Image')

			plt.subplot(3, 2, 6)
			plt.imshow(np.abs(fft_thresh), norm=LogNorm(vmin=0.1))
			plt.title('Denoised Spectrum')
			plt.colorbar()

			plt.show()



## 4. Advanced Filtering
In this section, you will experiment an example of advanced filtering. This method, while more efficient, is also more dedicated. You will see that it does not adapt well to noise distributions slightly different from what the method is made for. 

1. Re-use previous code to load grayscale 'im1' and synthetically noise it with an additive white Gaussian noise (AWGN) with standard deviation 100. 
	
2. Use the 'bm3d' filter to denoise the resulting sample. Be careful as bm3d takes int8 image as input. You can use 'img_as_ubyte' to cast the image. Compare the enhancement obtained with the results of basics filters. 

3. Instead of an AWGN, noise 'im1' with a Salt & Pepper distribution. Does it work well? 