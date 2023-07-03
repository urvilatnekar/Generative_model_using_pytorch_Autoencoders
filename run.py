# importing required libraries
import torch
from model.autoencoder_decoder_model import Encoder_Decoder_model
from data.data_utils import get_dl
from train import train_model

# setting seed
torch.manual_seed(0)

# parameters for model building
batchsize = 32
n_epoch = 10

# downloading and transforming the data
train_loader,test_loader = get_dl(batchsize)

# putting data into dictionary
dl = {}
dl["train"] = train_loader
dl["valid"] = test_loader

# autoencoder model
model = Encoder_Decoder_model(batchsize)

# defining device for model training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# learning rate
lr = 0.001

# selecting optimizer(Adam)
optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-05)

# MSE loss function 
loss_fn = torch.nn.MSELoss()

# training the model
model,dec_model = train_model(n_epoch,model,dl,optim,loss_fn)

#saving the model
torch.save(model.state_dict())
torch.save(dec_model.state_dict())
