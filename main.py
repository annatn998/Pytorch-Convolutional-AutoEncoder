import numpy as np
import random
from preparatory_functions import *
from training_functions import *
from error_functions import *
from autoencoder import AutoEncoder
import torch
import torch.nn as nn 
import logging
import gc
# erase disk memory so we re-train the same each time 
gc.collect()
seed = 100
torch.manual_seed(seed)
logging.basicConfig(level=logging.INFO)

# load & sample data
logging.info('loading dataset ....')
dataset_zip = np.load('./dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
imgs_full = dataset_zip['imgs']
sample_imgs = random.sample(list(imgs_full), 1000)
transformed_baseline_imgs = transform_background_colors(sample_imgs)
anomaly_imgs = np.array([create_anomalous_dataset(i) for i in transformed_baseline_imgs])
transformed_images = data_transformations(transformed_baseline_imgs)
transformed_anomaly_images = data_transformations(anomaly_imgs)
logging.info('finished loading base images and transforming anomalies!')


logging.info('preparing images for training ...')
data_loaders = torch.utils.data.DataLoader(transformed_images, batch_size=20, shuffle=False)
data_loaders_anomaly = torch.utils.data.DataLoader(transformed_anomaly_images, batch_size=20, shuffle=False)

logging.info('preparing model parameters ...')
model = AutoEncoder(channels=3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

logging.info('Training start!')
training_outputs, trained_model = training_loop(epochs=1, data_loader=data_loaders, model=model, criterion=criterion, optimizer=optimizer)
predictions = eval_loop(model=trained_model, data_loader=data_loaders)
anomaly_predictions = eval_loop(model=trained_model, data_loader=data_loaders_anomaly)
latent_space_predictions = eval_loop(model=trained_model, data_loader=data_loaders, latent_space=True)
average_recond_error, average_density, stdev_recon_error, stdev_density, density_list, recon_error_list = calc_density_and_recon_error(dataloader=data_loaders, 
                                                                                                                                       model=trained_model, 
                                                                                                                                       latent_space_images=latent_space_predictions)
average_recond_error_anomaly, average_density_anomaly, stdev_recon_error_anomaly, stdev_density_anomaly, density_list_anomaly, recon_error_list_anomaly = calc_density_and_recon_error(dataloader=data_loaders_anomaly, 
                                                                                                                                                                                       model=trained_model, 
                                                                                                                                                                                       latent_space_images=latent_space_predictions)
logging.info(f'average reconstruction error for images: {average_recond_error}')
logging.info(f'average reconstruction error for anomalous images: {average_recond_error_anomaly}')

# next steps is to create the for-loop and show the accuracy of the images if they've been detected as anomally or not 
correct = 0
incorrect = 0 
index = 0
error_interval = 0.001

anomalies_detected = 0
anomalies_missed = 0
normal_image_detected = 0
normal_image_missed = 0

# using the average reconstruction error for anomalies you can check whether or not it falls above or below the average error
for error in recon_error_list_anomaly:
    if error >= average_recond_error_anomaly - error_interval:
        anomalies_detected += 1
    else: 
        anomalies_missed += 1 

for error in recon_error_list:
    if error >= average_recond_error_anomaly - error_interval:
        normal_image_missed += 1 
    else: 
        normal_image_detected += 1


logging.info(f'true positives {normal_image_detected}')
logging.info(f'false negatives {normal_image_missed}')
logging.info(f'true negative {anomalies_detected}')
logging.info(f'false positive {anomalies_missed}')

