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
            nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.channels, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.Sigmoid()
        )


    def forward(self, x): 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def latent_space_image(self, x): 
        encoded = self.encoder(x)
        return encoded