import numpy as np
import random
from preparatory_functions import *
from training_functions import *
from autoencoder import AutoEncoder
import torch
import torch.nn as nn 
import torch.nn.functional as F


# load & sample data
dataset_zip = np.load('./dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
imgs_full = dataset_zip['imgs']
sample_imgs = random.sample(list(imgs_full), 1000)
transformed_baseline_imgs = transform_background_colors(sample_imgs)
anomaly_imgs = np.array([create_anomalous_dataset(i) for i in transformed_baseline_imgs])
transformed_images = data_transformations(transformed_baseline_imgs)
transformed_anomaly_images = data_transformations(anomaly_imgs)

data_loaders = torch.utils.data.DataLoader(transformed_images, batch_size=20, shuffle=False)
data_loaders_anomaly = torch.utils.data.DataLoader(transformed_anomaly_images, batch_size=20, shuffle=False)

model = AutoEncoder(channels=3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_outputs = training_loop(epochs=10, data_loader=data_loaders, model=model, criterion=criterion, optimizer=optimizer)
predictions = eval_loop(model=model, data_loader=data_loaders)
anomaly_predictions = eval_loop(model=model, data_loader=data_loaders_anomaly)
