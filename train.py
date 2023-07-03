#importing required libraires
import torch
import numpy as np


# function to train the model 
def train_model(no_of_epoch,model,dataloaders,optimizer,lossfn):
    # setting the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    best_loss = 100000000.0

    # for each epoch model will be trained or evaluated
    for epoch in range(no_of_epoch):
        print('Epoch {}/ {}'.format(epoch+1,no_of_epoch))
        print('-' * 10)
        for phase in ['train','valid']:
            train_loss=[]
            valid_loss = []
            if phase=='train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for inputs,labels in dataloaders[phase]:
                # putting inputs and labels in device
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # training the model with the given inputs if phase is train
                    gen_img,dec_model = model(inputs)
                    # calculting the loss
                    loss = lossfn(gen_img,inputs)
                    # for training phase backpropogation of model architecture 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.detach().cpu().numpy())
                    else:
                        valid_loss.append(loss.detach().cpu().numpy())
            # printing the loss            
            if phase == 'train':
                print("Train Loss : {}".format(np.mean(train_loss)))
            else:
                print("Valid Loss : {}".format(np.mean(valid_loss)))

    return model,dec_model
