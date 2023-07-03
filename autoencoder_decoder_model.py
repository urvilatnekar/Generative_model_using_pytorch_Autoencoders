# importing requirerd libraries
import torch.nn as nn #ontains classes and functions for defining and working with neural networks.
import torch.nn.functional as F #provides various functions as activation functions or other functional operations within a neural network
import torch #provides functionalities for tensor operations and building and training neural networks
from torchinfo import summary #summary function  to generate a summary of the model's architecture

# Encoder for Autoencoder
class Encoder(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        # the Encoder class inherits the basic functionality provided by nn.Module, such as parameter tracking, gradient computation, and network traversal.
        # This allows the Encoder class to be used as a building block for more complex neural network models.
        
        self.batchsize = batch_size
        self.conv1 = nn.Conv2d(1,32,3,1,padding=1)#defines a convolutional layer with 1 input channel, 32 output channels, a kernel size of 3x3, a stride of 1, and a padding of 1. The layer will perform convolution operations on the input data, applying 32 different filters to extract features from the input.
        self.conv2 = nn.Conv2d(32,64,3,2,padding=1)#a stride of 2 will further downsample the feature maps by a factor of 2.
        self.conv3 = nn.Conv2d(64,64,3,2,padding=1)
        self.conv4 = nn.Conv2d(64,64,3,1,padding=1)#a stride of 1 keeps the spatial dimensions of the feature maps unchanged.
        self.LRelu = nn.LeakyReLU()# The Leaky ReLU activation function introduces a small slope for negative values, which helps mitigate the "dying ReLU" problem.
        self.fc1 = nn.Linear(3136,2)#64*7*7 creates a fully connected layer corresponding to the flattened feature maps from the previous convolutional layers
    

    # function for forward propogation
    def forward(self,x):
        layer1 = self.LRelu(self.conv1(x))# applies the first convolutional layer conv1 to the input tensor x nd then applies the Leaky ReLU activation function LRelu to the output of conv1
        layer2 = self.LRelu(self.conv2(layer1))#applies the second convolutional layer conv2 to the output of layer1 and then applies the Leaky ReLU activation function.
        layer3 = self.LRelu(self.conv3(layer2))
        layer4 = self.LRelu(self.conv4(layer3))
        flat = layer4.view(self.batchsize,-1)#flattens the feature maps of layer4 into a 1D vector, while preserving the batch dimension 
        flat_shape = flat.size()[1]
        encoder_out = self.fc1(flat)#passes the flattened tensor flat through a fully connected layer fc1 output of this layer represents the encoded representation of the input data.
        return encoder_out

# Decoder for Autoencoder
class Decoder(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(2,3136)#represents a fully connected layer that takes an input of size 2 and outputs a tensor of size 3136
        self.Dconv1 = nn.ConvTranspose2d(64,64,3,1,padding=1)#convolutional transpose layers used to upsample the input tensor and reconstruct the original image
        self.Dconv2 = nn.ConvTranspose2d(64,64,3,2,padding=1,output_padding=1)
        self.Dconv3 = nn.ConvTranspose2d(64,32,3,2,padding=1,output_padding=1)
        self.Dconv4 = nn.ConvTranspose2d(32,1,3,1,padding=1)
        self.LRelu = nn.LeakyReLU()
    
    # function for forward propogation
    def forward(self,x):
        fc = self.fc1(x)
        reshaped = fc.view(self.batch_size,64,7,7)
        layer1 = self.LRelu(self.Dconv1(reshaped))
        layer2 = self.LRelu(self.Dconv2(layer1))
        layer3 = self.LRelu(self.Dconv3(layer2))
        layer4 = self.Dconv4(layer3)
        out = torch.sigmoid(layer4)#passed through the sigmoid activation function to obtain the reconstructed output image.

        return out

class Encoder_Decoder_model(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size

        # initializing obejct for Encoder
        self.enc = Encoder(self.batch_size)

        # initializing obejct for Decoder
        self.dec = Decoder(self.batch_size)
        self.sigmoid_act = torch.nn.Sigmoid()

    # function for forward propogation
    def forward(self,img):
        # img as input for encoder
        enc_out = self.enc(img)#input image img is passed through the enc attribute, which represents the encoder module (Encoder class).
         #This results in the encoded representation of the image, stored in the enc_out variable
        
        # encoder output as input for decoder
        dec_out = self.dec(enc_out)#enc_out is then passed through the dec attribute whcihperforms the decoding operation and generates the reconstructed output.
       
        out = self.sigmoid_act(dec_out)
        return out,self.dec
#The reconstructed output, dec_out, is returned along with the self.dec attribute, which represents the decoder module itself


#Encoder_model = Encoder(10)
#print(summary(Encoder_model,input_size=(10,1,28,28)))

#Decode_model = Decoder(10)
#print(summary(Decode_model,input_size=(10,2)))