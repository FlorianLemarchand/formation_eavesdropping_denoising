"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

# ========= Imports =========
import torch
import numpy as np
from math import log10
from os import makedirs, path
from torchsummary import summary

from skimage.metrics import peak_signal_noise_ratio, \
    mean_squared_error, \
    structural_similarity

from utils_PW2_complete import DnCNN, CustomDataset, make_learning_set, generate_logdir, Logger, ensemble_inference, REDNet10


# ========= Utils =========
def print_psnr_ssim(im1, im2, label):
    psnr = np.round(peak_signal_noise_ratio(im1, im2), 2)
    ssim = np.round(structural_similarity(im1, im2), 2)
    print('{}: PSNR: {} / SSIM: {}'.format(label, psnr, ssim))


def print_line_break():
    print('\n')


# ========= Write your code hereunder =========
# make_learning_set()

batch_size = 128

train_dataset = CustomDataset('data/out/bsd_learning/train/in', 'data/out/bsd_learning/train/ref')
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dataset = CustomDataset('data/out/bsd_learning/val/in', 'data/out/bsd_learning/val/ref')
val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset('data/out/bsd_learning/test/in', 'data/out/bsd_learning/test/ref', test=True)
test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the architecture
n_channels = 1  # 1 for grayscale, 3 for RGB
model = DnCNN(channels=n_channels, num_of_layers=17)
# model = REDNet10()

# Move model to GPU memory is available
device = 'cpu'
if torch.cuda.is_available():
    model = model.cuda()
    print('Model moved to GPU!')
    device = 'cuda'

# Display the architecture
summary(model, (1, 64, 64), device=device)

# Define the loss function
mse_loss = torch.nn.MSELoss()

# Define the optimizer
opt = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 100, 0.1)
# Set log directory
log_dir = generate_logdir('./logs')
print('Log directory is: {}'.format(log_dir))
tensorboard = Logger(log_dir)

# Training Loop
n_epochs = 1000
best_val_loss = 1
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

        # Compute loss
        loss = mse_loss(denoised_batch, target_batch)
        psnr = 10 * log10(1 / loss.item())

        # Back prop of loss
        loss.backward()

        # Optimisation step
        opt.step()

        # Log train
        if it % 25 is 0:
            current_iteration = epoch * len(train_dataset) + it * batch_size
            print("Train || Epoch:{}, Iter:{} || MSE:{} || PNSR: {}".format(epoch,
                                                                            it * batch_size,
                                                                            np.round(loss.item(), 4),
                                                                            np.round(psnr, 4)))
            tensorboard.add_scalar('Train/MSE', loss.item(), current_iteration)
            tensorboard.add_scalar('Train/PSNR', psnr, current_iteration)
            tensorboard.add_image_single('Train',
                                         torch.cat((
                                             noisy_batch[0],
                                             denoised_batch[0],
                                             target_batch[0]), dim=-1),
                                         current_iteration)

    # Log val
    with torch.no_grad():
        val_mse_sum = 0
        current_iteration = (epoch + 1) * len(train_dataset)
        for it, (val_noisy_batch, val_target_batch) in enumerate(val_generator):
            # Move images to GPU memory (if available)
            if torch.cuda.is_available():
                val_noisy_batch = val_noisy_batch.cuda()
                val_target_batch = val_target_batch.cuda()

            # Forward propagation of inputs
            val_denoised_batch = model(val_noisy_batch)

            # Compute loss
            val_mse_sum += mse_loss(val_denoised_batch, val_target_batch) * val_noisy_batch.shape[0]

        val_mse = val_mse_sum.item() / len(val_dataset)
        val_psnr = 10 * log10(1 / val_mse)
        print("\tVal || Epoch:{} || MSE: {} || PNSR: {}".format(epoch,
                                                                np.round(val_mse, 4),
                                                                np.round(val_psnr, 4)))
        tensorboard.add_scalar('Val/MSE', val_mse, current_iteration)
        tensorboard.add_scalar('Val/PSNR', val_psnr, current_iteration)
        tensorboard.add_image_single('Val',
                                     torch.cat((
                                         val_noisy_batch[0],
                                         val_denoised_batch[0],
                                         val_target_batch[0]), dim=-1),
                                     current_iteration)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), path.join(log_dir, 'best_model.pth'))
            print("\tNew best model saved!")

    scheduler.step()

# TEST CODE

# with torch.no_grad():
#     # Load model
#     saved_model_path = '/home/flemarch/Documents/Florian/THESE/ENSEIGNEMENT/2020-2021/formation_supelec/TP/formation_eavesdropping_denoising/logs/20_10_2020_14_41_4/best_model.pth'
#     checkpoint = torch.load(saved_model_path)
#     model.load_state_dict(checkpoint)
#     tensorboard = Logger(path.split(saved_model_path)[0])
#     test_mse_sum = 0
#
#     for it, (test_noisy_batch, test_target_batch) in enumerate(test_generator):
#         # Move images to GPU memory (if available)
#         if torch.cuda.is_available():
#             test_noisy_batch = test_noisy_batch.cuda()
#             test_target_batch = test_target_batch.cuda()
#
#         # Forward propagation of inputs
#         # test_denoised_batch = ensemble_inference(model, test_noisy_batch)
#         test_denoised_batch = model(test_noisy_batch)
#
#
#         # Compute loss
#         test_mse = mse_loss(test_denoised_batch, test_target_batch)
#         test_psnr = 10 * log10(1 / test_mse)
#         print("Test || Sample {}||MSE: {} || PNSR: {}".format(it + 1,
#                                                               np.round(test_mse.item(), 4),
#                                                               np.round(test_psnr, 4)))
#
#         test_mse_sum += test_mse.item()
#         tensorboard.add_scalar('Test/MSE', test_mse, it + 1)
#         tensorboard.add_scalar('Test/PSNR', test_psnr, it + 1)
#         tensorboard.add_image_single('Test',
#                                      torch.cat((
#                                          test_noisy_batch[0],
#                                          test_denoised_batch[0],
#                                          test_target_batch[0]), dim=-1),
#                                      it + 1)
#
#     avg_test_mse = test_mse_sum / len(test_dataset)
#     avg_test_psnr = 10 * log10(1 / avg_test_mse)
#     print("\tMean Test || MSE: {} || PNSR: {}".format(np.round(avg_test_mse, 4),
#                                                       np.round(avg_test_psnr, 4)))