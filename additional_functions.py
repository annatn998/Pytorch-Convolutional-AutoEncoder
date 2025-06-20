import matplotlib.pyplot as plt
def show_images_after_training(list_of_imgs):
    """
    function: show grid of images after training against reconstructed image 
    args: 
        list_of_imgs (list): list of tuples (original image, reconstructed image)
    """
    for img in list_of_imgs:
        plt.figure(figsize=(9,2))
        plt.gray()
        # breaking connection between tensor & underlying numpy array
        # modify the array without affecting original 
        og_imgs = img[0].detach().numpy()
        recon_imgs = img[1].detach().numpy()

        for i, item in enumerate(og_imgs):
            if i >= 1: break
            plt.subplot(2, 9, i+1)
            item = item.transpose(1,2,0)
            plt.imshow(item.squeeze(), cmpa='gray')

        for i, item in enumerate(recon_imgs):
            if i >= 1: break
            plt.subplot(2, 9, i+1)
            item = item.transpose(1,2,0)
            plt.imshow(item.squeeze(), cmpa='gray')

