import numpy as np
import torch.nn.functional as F
from sklearn.neightbors import KernelDensity

def calc_density_and_recon_error(dataloader, model, latent_space_images):
    """
    function: calculate the reconstruction error and density of the latent space vs fully generated image
    args:
        dataloader (DataLoader): data loader for pytorch model
        model (AutoEncoder): pytorch model
        latent_space_images (list): list of latent space images (just used to calculate the shape for the density model)
    return: 
        average_recond_error (float): average reconstruction error
        average_density (float): average density
        stdev_recon_error (float): standard deviation of reconstruction error
        stdev_density (float): standard deviation of density
    """
    reshaped_images = np.array([i.squeeze().reshape(-1) for i in latent_space_images[0]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(reshaped_images)
    density_list = []
    recon_error_list = []

    for img in dataloader:
        encoded_img = model.latent_space_img(img) # create a compressed version of the image using encoder
        # flatten the encoded image (latent space)
        encoded_img = np.array(imgs.reshape(-1).detach().numpy() for imgs in encoded_img)

        ## flatten the original input image
        density = kde.score_samples(encoded_img)[0]

        # fully reconstructed image
        reconstruction = model(img)
        reconstruction_error = F.mse_loss(reconstruction, img, reduction='mean').item()

        density_list.append(density)
        recon_error_list.append(reconstruction_error)

    average_recond_error = np.mean(np.array(recon_error_list))
    average_density = np.mean(np.array(density_list))

    stdev_recon_error = np.std(np.array(recon_error_list))
    stdev_density = np.std(np.array(density_list))

    return average_recond_error, average_density, stdev_recon_error, stdev_density, density_list, recon_error_list 
