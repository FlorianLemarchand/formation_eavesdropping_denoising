"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

# ========= Imports =========
from math import log10
from utils_PW2 import *
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision.utils import save_image, make_grid

# ========= Question selector ============
question = 1

# ========= Write your code hereunder =========
# Question 1
if question_solver(question, 1):
    # If it does not exist yet: create the learning dataset using BSD original data
    if not path.exists('data/out/bsd_learning'):
        make_learning_set()

# Question 2
if question_solver(question, 2):
    # Define the batch_size, the number of samples to be fed to the network before the evaluation of the loss function
    batch_size = 8

    # Get number of available threads
    num_threads = torch.multiprocessing.cpu_count()

    # Train datagenerator
    train_dataset = CustomDataset('data/out/bsd_learning/train/in', 'data/out/bsd_learning/train/ref')
    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_threads)

    # Validation datagenerator
    val_dataset = CustomDataset('data/out/bsd_learning/val/in', 'data/out/bsd_learning/val/ref', is_test=True)
    val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if question is 2:
        # Display a batch from each of the generators
        train_batch = next(iter(train_generator))
        train_grid = make_grid(torch.cat(train_batch))

        val_batch = next(iter(val_generator))
        val_grid = make_grid(torch.cat(val_batch))
        plt.subplot(2, 1, 1)
        plt.imshow(np.transpose(train_grid, (1, 2, 0)))
        plt.title("Training Batch")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 1, 2)
        plt.imshow(np.transpose(val_grid, (1, 2, 0)))
        plt.title("Validation Batch")
        plt.xticks([])
        plt.yticks([])

        plt.show()

# Question 3
if question_solver(question, 3):
    # Define the architecture of the CNN
    model = DnCNN(channels=1, num_of_layers=17)

    # Move model to GPU memory if available, default is cpu
    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        print('Model moved to GPU!')
        device = 'cuda'

    # Display the architecture
    if question is 3:
        summary(model, (1, 321, 481), device=device)

    # Define the loss function
    mse_loss = torch.nn.MSELoss()

    # Define the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 2000  # max number of times the entire dataset will be fed to the model
    best_val_loss = float('inf')  # Init the best value of the loss

# Question 4
if question_solver(question, 4):
    # Set the log directory to store information throughout the training
    log_dir = generate_logdir('./logs')
    print('Log directory is: {}'.format(log_dir))
    tensorboard = Logger(log_dir)

    # Training Loop
    for epoch in range(n_epochs):
        for it, (noisy_batch, target_batch) in enumerate(train_generator):
            # Zero the gradients
            opt.zero_grad()

            # Move images to GPU memory (if possible)
            if torch.cuda.is_available():
                noisy_batch = noisy_batch.cuda()
                target_batch = target_batch.cuda()

            # Forward propagation of inputs
            denoised_batch = model(noisy_batch)

            # Compute loss and other metrics
            loss = mse_loss(denoised_batch, target_batch)
            psnr = 10 * log10(1 / loss.item())

            # Backward propagation of loss
            loss.backward()

            # Optimisation step
            opt.step()

            # Log train
            current_iteration = epoch * len(train_dataset) + it * batch_size
            # Display the loss and PSNR over the training batch
            print("Train || Epoch:{}, Iter:{} || MSE:{} || PNSR: {}".format(epoch,
                                                                            it * batch_size,
                                                                            np.round(loss.item(), 4),
                                                                            np.round(psnr, 4)))

            # Log to tensorboard
            tensorboard.add_scalar('Train/MSE', loss.item(), current_iteration)
            tensorboard.add_scalar('Train/PSNR', psnr, current_iteration)

        # Log val
        with torch.no_grad():
            val_mse_sum = 0  # init the sum of validation losses
            current_iteration = (epoch + 1) * len(train_dataset)
            for it, (val_noisy_batch, val_target_batch) in enumerate(val_generator):
                # Move images to GPU memory (if available)
                if torch.cuda.is_available():
                    val_noisy_batch = val_noisy_batch.cuda()
                    val_target_batch = val_target_batch.cuda()

                # Forward propagation of inputs
                val_denoised_batch = model(val_noisy_batch)

                # Compute loss and add it to the sum
                val_mse_sum += mse_loss(val_denoised_batch, val_target_batch) * val_noisy_batch.shape[0]

            # Compute the loss and metrics means over the validation set
            val_mse = val_mse_sum.item() / len(val_dataset)
            val_psnr = 10 * log10(1 / val_mse)

            # Display the mean loss and PSNR
            print("\tVal || Epoch:{} || MSE: {} || PNSR: {}".format(epoch,
                                                                    np.round(val_mse, 4),
                                                                    np.round(val_psnr, 4)))

            # Log to tensorboard
            tensorboard.add_scalar('Val/MSE', val_mse, current_iteration)
            tensorboard.add_scalar('Val/PSNR', val_psnr, current_iteration)

            # Save score and model as the best if validation loss is better
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                torch.save(model.state_dict(), path.join(log_dir, 'best_model.pth'))
                print("\tNew best model saved!")

# Question 5
if question_solver(question, 5):
    with torch.no_grad():  # do not use gradients for tensors, save memory
        # Create directory for output images
        makedirs('data/out/out_dncnn', exist_ok=True)

        model = DnCNN(channels=1, num_of_layers=17)

        # Load model
        saved_model_path = 'logs/dncnn_sigma100/best_model.pth'
        print('Loading Model from: {}'.format(saved_model_path))

        device = 'cpu'
        if torch.cuda.is_available():
            model = model.cuda()  # Move model to GPU memory if available
            print('Model moved to GPU!')
            device = 'cuda'
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Init the mse sum of test samples
        test_mse_sum = 0

        # Create the test datagenerator
        test_dataset = CustomDataset('data/out/bsd_learning/test/in', 'data/out/bsd_learning/test/ref', is_test=True)
        test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Define the loss function
        mse_loss = torch.nn.MSELoss()

        # Test Loop
        for it, (test_noisy, test_target) in enumerate(test_generator):
            # Move images to GPU memory (if available)
            if torch.cuda.is_available():
                test_noisy = test_noisy.cuda()
                test_target = test_target.cuda()

            # Forward propagation of inputs
            test_denoised = model(test_noisy)

            # Save noisy/denoised/ref images
            save_image(torch.cat((test_noisy, test_denoised, test_target), dim=3),
                       path.join('data/out/out_dncnn', '{}.png'.format(it + 1)))

            # Compute loss and metrics
            test_mse = mse_loss(test_denoised, test_target)
            test_psnr = 10 * log10(1 / test_mse)

            # Print Loss and PSNR for image
            print("Test || Sample {}||MSE: {} || PNSR: {}".format(it + 1,
                                                                  np.round(test_mse.item(), 4),
                                                                  np.round(test_psnr, 4)))

            test_mse_sum += test_mse.item()

        # Compute average loss and metrics over test set
        avg_test_mse = test_mse_sum / len(test_dataset)
        avg_test_psnr = 10 * log10(1 / avg_test_mse)

        # Print average Loss and PSNR for test set
        print("\tMean Test || MSE: {} || PNSR: {}".format(np.round(avg_test_mse, 4),
                                                          np.round(avg_test_psnr, 4)))

# Question 6
if question_solver(question, 6):
    with torch.no_grad():  # do not use gradients for tensors, save memory
        data_dir = 'data/out/bsd_learning_intercept'
        output_dir = 'data/out/out_dncnn17_intercept'  # TO CHANGE Q6

        # Create the dataset if it does not exist
        if not path.exists(data_dir):
            make_learning_set_intercept()

        # Create directory for output images
        makedirs(output_dir, exist_ok=True)

        # Set the model
        model = DnCNN(channels=1, num_of_layers=17)  # TO CHANGE Q6

        # Load model
        saved_model_path = 'logs/natural_intercept_dncnn_17/best_model.pth' # TO CHANGE Q6
        print('Loading Model from: {}'.format(saved_model_path))

        device = 'cpu'
        if torch.cuda.is_available():
            model = model.cuda() # Move model to GPU memory if available
            print('Model moved to GPU!')
            device = 'cuda'
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Init the mse sum of test samples
        test_mse_sum = 0

        # Create the test datagenerator
        test_dataset = CustomDataset(path.join(data_dir, 'test/in'),
                                     path.join(data_dir, 'test/ref')
                                     , is_test=True)
        test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Define the loss function
        mse_loss = torch.nn.MSELoss()

        # Test Loop
        for it, (test_noisy, test_target) in enumerate(test_generator):
            # Move images to GPU memory (if available)
            if torch.cuda.is_available():
                test_noisy = test_noisy.cuda()
                test_target = test_target.cuda()

            # Forward propagation of inputs
            test_denoised = model(test_noisy)

            # Save noisy/denoised/ref images
            save_image(torch.cat((test_noisy, test_denoised, test_target), dim=3),
                       path.join(output_dir, '{}.png'.format(it + 1)))

            # Compute loss and metrics
            test_mse = mse_loss(test_denoised, test_target)
            test_psnr = 10 * log10(1 / test_mse)

            # Print Loss and PSNR for image
            print("Test || Sample {}||MSE: {} || PNSR: {}".format(it + 1,
                                                                  np.round(test_mse.item(), 4),
                                                                  np.round(test_psnr, 4)))

            test_mse_sum += test_mse.item()

        # Compute average loss and metrics over test set
        avg_test_mse = test_mse_sum / len(test_dataset)
        avg_test_psnr = 10 * log10(1 / avg_test_mse)

        # Print average Loss and PSNR for test set
        print("\tMean Test || MSE: {} || PNSR: {}".format(np.round(avg_test_mse, 4),
                                                          np.round(avg_test_psnr, 4)))

# Question 8
if question_solver(question, 8):
    with torch.no_grad():  # do not use gradients for tensors, save memory
        data_dir = 'data/out/bsd_learning_intercept'
        output_dir = 'data/out/out_dncnn20_intercept_ensemble'

        # Create the dataset if it does not exist
        if not path.exists(data_dir):
            make_learning_set_intercept()

        # Create directory for output images
        makedirs(output_dir, exist_ok=True)

        # Set the model
        model = DnCNN(channels=1, num_of_layers=20)

        # Load model
        saved_model_path = 'logs/natural_intercept_dncnn_20/best_model.pth'
        print('Loading Model from: {}'.format(saved_model_path))

        device = 'cpu'
        if torch.cuda.is_available():
            model = model.cuda()  # Move model to GPU memory if available
            print('Model moved to GPU!')
            device = 'cuda'
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Init the mse sum of test samples
        test_mse_sum = 0

        # Create the test datagenerator
        test_dataset = CustomDataset(path.join(data_dir, 'test/in'),
                                     path.join(data_dir, 'test/ref')
                                     , is_test=True)
        test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Define the loss function
        mse_loss = torch.nn.MSELoss()

        # Test Loop
        for it, (test_noisy, test_target) in enumerate(test_generator):
            # Move images to GPU memory (if available)
            if torch.cuda.is_available():
                test_noisy = test_noisy.cuda()
                test_target = test_target.cuda()

            # Forward propagation of inputs
            test_denoised = ensemble_inference(model, test_noisy)

            # Save noisy/denoised/ref images
            save_image(torch.cat((test_noisy, test_denoised, test_target), dim=3),
                       path.join(output_dir, '{}.png'.format(it + 1)))

            # Compute loss and metrics
            test_mse = mse_loss(test_denoised, test_target)
            test_psnr = 10 * log10(1 / test_mse)

            # Print Loss and PSNR for image
            print("Test || Sample {}||MSE: {} || PNSR: {}".format(it + 1,
                                                                  np.round(test_mse.item(), 4),
                                                                  np.round(test_psnr, 4)))

            test_mse_sum += test_mse.item()

        # Compute average loss and metrics over test set
        avg_test_mse = test_mse_sum / len(test_dataset)
        avg_test_psnr = 10 * log10(1 / avg_test_mse)

        # Print average Loss and PSNR for test set
        print("\tMean Test || MSE: {} || PNSR: {}".format(np.round(avg_test_mse, 4),
                                                          np.round(avg_test_psnr, 4)))
