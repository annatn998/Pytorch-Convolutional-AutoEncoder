import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import torch 

def calc_density_and_recon_error(dataloader, model, latent_space_images, height):
    """
    function: calculate the reconstruction error and density of the latent space vs fully generated image
    args:
        dataloader (DataLoader): data loader for pytorch model
        model (AutoEncoder): pytorch model
        latent_space_images (list): list of latent space images (just used to calculate the shape for the density model)
        height (float): height of the distribution (bandwidth in sklearn smaller bandwith = narrow peaks, larger bandwidth = wide peaks) 
    return: 
        average_recond_error (float): average reconstruction error
        average_density (float): average density
        stdev_recon_error (float): standard deviation of reconstruction error
        stdev_density (float): standard deviation of density
    """
    density_list = []
    recon_error_list = []


    for images, latent_images in zip(dataloader, latent_space_images):
        for img, latent_img in zip(images, latent_images):
            N = img.numel()
            M = latent_img.numel()

            # fit the kernel density model to the flattened out original image
            kde = KernelDensity(kernel='gaussian', bandwidth=height).fit(img.reshape(1, -1).detach().numpy())

            with torch.no_grad():
                # create a compressed version of the image using encoder
                encoded_img = model.latent_space_image(img) 
                # fully reconstructed image
                reconstruction = model(img)

            # pad the latent space prediction to be the same size as the original image for prediction of the kernel density

            # encoded_img should be a 1D tensor
            pad_amount = N - M
            reshaped_images = F.pad(encoded_img.reshape(1, -1), (0, pad_amount), mode='constant', value=0)
            reshaped_images.reshape(1, -1).detach().numpy()

            ## flatten the original input image
            density = kde.score_samples(reshaped_images)[0]
            
            reconstruction_error = F.mse_loss(reconstruction, img, reduction='mean').item()
            density_list.append(density)
            recon_error_list.append(reconstruction_error)

    average_recond_error = np.mean(np.array(recon_error_list))
    average_density = np.mean(np.array(density_list))

    stdev_recon_error = np.std(np.array(recon_error_list))
    stdev_density = np.std(np.array(density_list))
    
    return average_recond_error, average_density, stdev_recon_error, stdev_density, density_list, recon_error_list