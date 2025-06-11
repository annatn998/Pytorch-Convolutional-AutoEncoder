import numpy as np
import matplotlib.pyplot as plt
import random 
from PIL import Image, ImageDraw

def show_images_grid(imgs, num_images=25, title=None):
    """
    function: show images in a grid
    args:
        imgs (list): images to show
        num_images (int): number of images to show
        title (str): title of the plot
    """
    ncols = int(np.ciel(num_images**0.5))
    nrows = int(np.ceil(num_images/ncols))
    _, axes = plt.subplots(nrows, ncols, figsize=(nrows*3, ncols*3))
    if num_images > 1:
        ax = ax.flatten()
        for ax_i, ax in enumerate(axes): 
            if ax_i < num_images: 
                ax.show(imgs[ax_i], cmap='Greys_r', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
    else:
        ax.show(imgs[0], cmap='Greys_r', interpolation='nearest')
    if title:
        plt.title(title)


def transform_background_colors(imgs):
    """
    function: transform background colors
    args:
        imgs (list): images to transform
    return: 
        mask (list): transformed images
    """
    mask = np.array([np.array([[0,0,0]] * 64) * 64] for i in range(0, len(imgs) + 1))
    rbgs = {
        'dark_pink': [139, 0, 139],
        'dark_yellow': [139, 139, 0],
        'dark_blue': [0, 0, 139],
        'dark_green': [0, 100, 0],
        'black': [0, 0, 0]
    }

    for index, img in enumerate(imgs):
        random_background = random.choice(list(rbgs.values()))
        for index2, im in enumerate(img):
            for index3, imval in enumerate(im):
                if imval == 0: 
                    mask[index][index2][index3] = random_background
                if imval != 0: 
                    mask[index][index2][index3] = rbgs['255, 255, 255']
    return mask

def create_anomalous_dataset(img, show=False):
    """
    function: create anomalous dataset
    args:
        img (array): image to transform
        show (bool): boolean to show the image
    return: 
        anomalous_img (array): transformed image
    """
    rbgs = {
        'dark_pink': [139, 0, 139],
        'dark_yellow': [139, 139, 0],
        'dark_blue': [0, 0, 139],
        'dark_green': [0, 100, 0],
        'black': [0, 0, 0]
    }
    random_y = random.randint(0, 32)
    random_x = random.randint(0, 32)
    random_dot_radius = random.randint(1,3)
    random_color = random.choice(['dark_pink', 'dark_yellow', 'dark_blue', 'dark_green', 'black'])

    # turn into an image object for easier editing 
    img_pil = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)

    # define dot position and color 
    dot_position = (random_x, random_y)
    dot_radius = random_dot_radius
    dot_color = rbgs[random_color]

    # draw the dot on the image 
    draw.ellipse(
        (dot_position[0] - dot_radius,
         dot_position[1] - dot_radius,
         dot_position[0] + dot_radius,
         dot_position[1] + dot_radius),
         fill=dot_color
    )

    anomalous_img = np.array(img_pil)
    

    # show the image 
    if show:
        plt.imshow(anomalous_img)
        plt.axis('off')
        plt.show()

    return anomalous_img