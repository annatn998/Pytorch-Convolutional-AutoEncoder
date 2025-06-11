from torchvision import transforms
import numpy as np 
import torch

def data_transformations(data):
    """
    function: pytorch data transformations this will normalize the data & transform it to a tensor for pytorch model 
    args:
        data (list): images to transform
    return: 
        transformed_images (list): transformed images
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transformed_images = [transform(np.array(i, dtype=np.uint8)) for i in data]
    
    return transformed_images


def training_loop(epochs, data_loader, model, criterion, optimizer, verbose: bool = True): 
    """
    function: training loop for pytorch model
    args:
        epochs (int): number of epochs to train
        data_loader (DataLoader): data loader for pytorch model
        model (AutoEncoder): pytorch model
        criterion (MSELoss): loss function for pytorch model
        optimizer (Adam): optimizer for pytorch model
        verbose (bool): boolean to print loss
    return: 
        outputs (list): list of outputs
    """
    outputs = []

    for e in range(epochs): 
        for img in data_loader:
            img = img.float()
            recon = model(img)
            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs.append((e, img, recon)) 
        if verbose:
            print(f"Epoch {e} loss: {loss}")

    return outputs

def eval_loop(model, data_loader):
    """
    function: evaluation loop for pytorch model
    args:
        model (AutoEncoder): train pytorch model
        data_loader (DataLoader): data loader for pytorch model
    return: 
        predictions (list): list of predictions
    """

    model.eval()
    predictions = []

    with torch.no_grad():
        for img in data_loader:
            preds = model(img)
            predictions.append(preds)
    return predictions 

