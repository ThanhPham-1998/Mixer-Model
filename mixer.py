import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class MLPBlock(nn.Module):
    def __init__(self, x_dim, hidden_dim, drop_out=0.5):
        """
        - x_dim: dimention of input and output layer
        - hidden_dim: dimention of hidden layer
        """
        super().__init__()
        hidden_dim = x_dim if hidden_dim is None else hidden_dim
        self.Linear_1 = nn.Linear(x_dim, hidden_dim)
        self.dropout_1 = nn.Dropout(p=drop_out)
        self.Linear_2 = nn.Linear(hidden_dim, x_dim)
        self.dropout_2 = nn.Dropout(p=drop_out)


    def forward(self, x):
        """
        - shape input: (n_samples, n_channels, n_patches)
        """
        result = self.dropout_1(F.gelu(self.Linear_1(x)))
        return self.dropout_2(self.Linear_2(result))

class MixerLayer(nn.Module):

    def __init__(self, n_patches, hidden_dim, token_mlp_dim, channel_mlp_dim, drop_out: float=0.3):

        """
            - n_patches: number of patches the image split up into.
            - hidden_dim: dimention of patch embedding.
            - token_mlp_dim: dimention of hidden layer for MLPBlock when mode token mixing is turn.
            - channel_mlp_dim: dimention of hidden layer for MLPBlock when mode channel mixing is turn.

        """
        super().__init__()

        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.token_MLPBlock = MLPBlock(n_patches, token_mlp_dim, drop_out)
        self.chanel_MLPBlock = MLPBlock(hidden_dim, channel_mlp_dim, drop_out)

    def forward(self, x):
        result = self.norm_1(x)                     # N,n_patches,hidden_dim
        result = result.permute(0, 2, 1)            # N,hidden_dim,n_patches
        result = self.token_MLPBlock(result)
        result = result.permute(0, 2, 1)
        x = x + result
        result = self.norm_2(x)
        return x + self.chanel_MLPBlock(result)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):

        """
            - in_channels: The channels of input image
            - out_channels: out channels 
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(x)


class MixerModel(nn.Module):
    def __init__(self, image_size, patch_size, token_mlp_dim, channel_mlp_dim, 
                hidden_dim, n_classes, n_blocks, drop_out=0.3):
        """
                - image_size: int   (h x w)
                - patch_size: int   (h' x w')
                - token_mlp_dim:
                - channel_mlp_dim:
                - n_classes: number of classes for classification.
                - hidden_dim:
                - n_blocks: number of Mixer layers

        """
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_embedder = ConvBlock(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(n_patches=n_patches,
                    hidden_dim=hidden_dim, 
                    token_mlp_dim=token_mlp_dim, 
                    channel_mlp_dim=channel_mlp_dim,
                    drop_out=drop_out
                    ) 
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        x = self.patch_embedder(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        x = self.mixer_layers(x)
        x = self.norm(x) 
        x = x.mean(axis=1)
        return self.fc(x)
    

if __name__ == '__main__':
    model = MixerModel(image_size=224, patch_size=14, token_mlp_dim=256, channel_mlp_dim=256, hidden_dim=128,
                   n_classes=5, n_blocks=8)
    inputs = torch.randn(5,3,224,224)
    target = torch.randint(5,(5,))
    pred = model(inputs)
    print(target, pred)
    loss = nn.CrossEntropyLoss()(pred, target)
    print(loss)

    




