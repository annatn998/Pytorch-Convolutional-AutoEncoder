# 🧠 PyTorch Autoencoder for Anomaly Detection on dSprites Dataset

This repository contains a PyTorch implementation of an autoencoder designed to perform anomaly detection on the [dSprites dataset](https://github.com/deepmind/dsprites-dataset). The model learns a compact latent representation of the sprites and uses reconstruction error to identify anomalous inputs.

## 📦 Features

- PyTorch-based autoencoder architecture  
- Anomaly detection via reconstruction loss  
- Supports training and testing on the dSprites dataset  
- Simple, single-file interface (`main.py`)  
- Easily extendable for other binary image datasets  

## 📁 Dataset

This project uses the dSprites dataset created by DeepMind, which consists of 2D shapes (hearts, ellipses, squares) generated from a ground truth factor space. To use it, create a directory called `data`, then download the dataset file and save it into that folder using the following commands:

`mkdir -p data && wget https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true -O data/dsprites.npz`

This will ensure that `main.py` can locate the dataset at `data/dsprites.npz`.

## 🛠️ Setup and Installation

To get started, clone this repository and set up your Python environment:

`git clone https://github.com/yourusername/pytorch-autoencoder-dsprites.git && cd pytorch-autoencoder-dsprites`

(Optional but recommended) Create a virtual environment:

`python -m venv venv && source venv/bin/activate` (on Windows: `venv\Scripts\activate`)

Next, install the required Python packages:

`pip install torch torchvision matplotlib numpy`

Alternatively, if a `requirements.txt` is provided, install everything via:

`pip install -r requirements.txt`

Contents of `requirements.txt` (if using):
torch
torchvision
matplotlib
numpy




## 🚀 Running the Autoencoder

After setting up the environment and downloading the dataset, you can train the model using:

`python main.py`

If your `main.py` script supports command-line arguments, you can customize training:

`python main.py --epochs 50 --batch_size 128 --learning_rate 0.001`

After training, the autoencoder will attempt to reconstruct the input images. Inputs with poor reconstructions (i.e., high reconstruction error) are considered anomalous. You can visualize the loss or outputs saved to an `outputs/` folder, if implemented.



