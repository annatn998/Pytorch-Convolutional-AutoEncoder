import torch.nn as nn 

class AutoEncoder(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels (int): The number of channels in the input image

        Initialize AutoEncoder object, building the encoder and decoder networks
        """
        super(AutoEncoder, self).__init__()
        self.channels = channels 

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3 padding=1),
            nn.Relu(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3 padding=1),
            nn.Relu(),
            nn.Conv2d(inchannels=32, out_channels=64, kernel_size=3, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTransposed2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.Relu(),
            nn.ConvTransposed2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Relu(),
            nn.ConvTransposed2d(in_channels=16, out_channels=self.channels, kernel_size=3, padding=1)
            nn.Sigmoid()
        )

    def forward(self, x): 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def latent_space_image(self, x): 
        encoded = self.encoder(x)
        return encoded