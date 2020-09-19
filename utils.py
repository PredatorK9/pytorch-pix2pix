import torch
import torch.nn as nn


def apply_weights(model):
    '''
    A small function for weight initialization of the model.
    '''
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class EncoderBlock(nn.Module):
    '''
    Returns a block of encoding unit for the Unet model.\n

    Args:\n
    > in_features: Number of input channels.\n
    > out_features: Number of output channels.\n
    > batch_norm: Use batch normalization or not (default=```True```).
    '''
    def __init__(self, in_features, out_features, batch_norm=True):
        super(EncoderBlock, self).__init__()
        self.batch = batch_norm

        self.conv = nn.Conv2d(in_features, out_features,
            kernel_size=4, stride=2, padding=1, bias=not batch_norm)
        
        if self.batch:
            self.batch_norm = nn.BatchNorm2d(out_features)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batch:
            x = self.batch_norm(x)
        return self.leaky_relu(x)


class DecoderBlock(nn.Module):
    '''
    Returns a block of decoding unit for the Unet model.\n

    Args:\n
    > in_features: Number of input channels.\n
    > out_features: Number of output channels.\n
    > dropout: Use dropout or not (default=```True```).
    '''
    def __init__(self, in_features, out_features, dropout=True):
        super(DecoderBlock, self).__init__()
        self.drop = dropout

        self.conv_transpose = nn.ConvTranspose2d(in_features, out_features,
            kernel_size=4, stride=2, padding=1, bias=False)

        self.batch_norm = nn.BatchNorm2d(out_features)

        if self.drop:
            self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x, skip_x):
        '''
        x: input tensor\n
        skip_x: skip connection tensor\n
        '''
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        if self.drop:
            x = self.dropout(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.relu(x)