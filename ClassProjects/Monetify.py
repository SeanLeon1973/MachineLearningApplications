# -*- coding: utf-8 -*-

import numpy
import cv2
import os

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn

import keras
import keras.layers as layers
from keras.layers import Input

import numpy as np

import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class Generator(nn.Module):
    def __init__(self, nz, nc , ngf ):
        super(Generator, self).__init__()

        main = nn.Sequential(

            nn.Conv2d         (nc        ,
                               ngf       ,
                               4         ,
                               2         ,
                               1         ,
                               bias=False  ),
            nn.BatchNorm2d    (ngf         ),
            nn.LeakyReLU      (0.2         ,
                               inplace=True),

            nn.ConvTranspose2d(ngf       ,
                               ngf       ,
                               4         ,
                               2         ,
                               1         ,
                               bias=False  ),
            nn.BatchNorm2d    (ngf         ),
            nn.LeakyReLU      (0.2         ,
                               inplace=True),

            nn.Conv2d         (256       , 
                               int(ngf/2),
                               4         ,
                               2         ,       
                               1         ,
                               bias=False  ),
            nn.BatchNorm2d    (int(ngf/2)  ),
            nn.LeakyReLU      (0.2         ,
                               inplace=True),
            
            nn.ConvTranspose2d(int(ngf/2),
                               int(ngf/4),
                               4         ,
                               2         ,
                               1         ,
                               bias=False  ),
            nn.BatchNorm2d    (int(ngf/4)  ),
            nn.LeakyReLU      (0.2         ,
                               inplace=True),
            
            nn.Conv2d         (int(ngf/4),
                               int(ngf/8),
                               4         ,
                               2         ,
                               1         ,
                               bias=False  ),
            nn.BatchNorm2d    (int(ngf/8)  ),
            nn.LeakyReLU      (0.2         ,
                               inplace=True),

            nn.ConvTranspose2d(int(ngf/8) ,
                               int(ngf/16),
                               4          ,
                               2          ,
                               1          ,
                               bias=False  ),
            nn.BatchNorm2d    (int(ngf/16) ),
            nn.LeakyReLU      (0.2         ,
                               inplace=True),
            
            nn.Conv2d         (int(ngf/16),
                               int(ngf/32),
                               4          ,
                               2          ,
                               1          ,
                               bias=False  ),
            nn.BatchNorm2d    (int(ngf/32) ),
            nn.LeakyReLU      (0.2         , 
                               inplace=True),  

        )

        main.add_module('final_layer_deconv', nn.ConvTranspose2d(int(ngf/32),
                                                                 nc         , 
                                                                 4          , 
                                                                 2          ,
                                                                 1          , 
                                                                 bias=False ))
        main.add_module('final_layer_tanh', nn.Tanh())

        self.main = main

    def forward(self, input):

        output = input
        for layer in self.main:
            output = layer(output)
            
        return output  
    
    
class Transpose(nn.Module):
    def forward(self, input):
        
        return input[ 0 ][ 0 ].reshape( input[ 0 ][ 0 ].shape[ 1 ], 
                                        input[ 0 ][ 0 ].shape[ 0 ])
    
class Discriminator(nn.Module):
    def __init__(self, nz, nc, ndf ):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d     (nc        ,
                           ndf       ,
                           4         ,
                           2         ,
                           1         ,
                           bias=False  ),
            nn.LeakyReLU  (0.2         ,
                           inplace=True),
            
            nn.Conv2d     (ndf       ,
                           ndf * 2   ,
                           4         ,
                           2         ,
                           1         ,
                           bias=False  ),
            nn.BatchNorm2d(ndf * 2     ),
            nn.LeakyReLU  (0.2         ,
                           inplace=True),
            
            nn.Conv2d     (ndf * 2   ,
                           ndf * 4   ,
                           4         ,
                           2         ,
                           1         ,
                           bias=False  ),
            nn.BatchNorm2d(ndf * 4     ),
            nn.LeakyReLU  (0.2         , 
                           inplace=True),
            
            nn.Conv2d     (ndf * 4   , 
                           ndf * 8   ,
                           4         , 
                           2         ,
                           1         ,
                           bias=False  ),
            nn.BatchNorm2d(ndf * 8     ),
            nn.LeakyReLU  (0.2         ,
                           inplace=True), 
        )
        
        main.add_module('final_layers_conv'   , nn.Conv2d (ndf * 8   ,
                                                           1         ,
                                                           4         ,
                                                           1         ,
                                                           0         ,
                                                           bias=False))
        main.add_module('final_layers_linear' , nn.Linear (13        ,
                                                           1         ,
                                                           bias=False))
        main.add_module('final_layers_transp' , Transpose (          ))
        main.add_module('final_layers_output' , nn.Linear (13        ,
                                                           1         ,
                                                           bias=False))
        main.add_module('final_layers_sigmoid', nn.Sigmoid(          ))
        
        self.main = main

    def forward(self, input):

        output = input
        for layer in self.main:
            
            output = layer(output)
        
        return output.view(-1, 1)[ 0 ]   
        
if __name__=="__main__":
    
    WORK_DIR   = \
        "D:/Classes/CS795MLAPPs/Project2_Monet/gan-getting-started/"
    OUTPUT_DIR = \
        "D:/Classes/CS795MLAPPs/Project2_Monet/AnimeGANData/Monetify_output/"
        
    folders = [ WORK_DIR + "monet_jpg/",
                WORK_DIR + "photo_jpg/"] 
    
    netG = Generator    ( 10, 3, 256 )
    netD = Discriminator( 10, 3, 256 )
    
    optimizerG = optim.Adam(netG.parameters(), lr = 0.0001, betas = (0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr = 0.0001, betas = (0.5, 0.999))
    
    dataset_train = dset.ImageFolder(
    root="D:/Classes/CS795MLAPPs/Project2_Monet/AnimeGANData/Train",
    transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    )
    
    dataset_test = dset.ImageFolder(
    root="D:/Classes/CS795MLAPPs/Project2_Monet/AnimeGANData/Test",
    transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    )  
    
    dataset_train_nonmonet = dset.ImageFolder(
    root="D:/Classes/CS795MLAPPs/Project2_Monet/AnimeGANData/nonMonetTrain",
    transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    )
    
    dataset_test_nonmonet = dset.ImageFolder(
    root="D:/Classes/CS795MLAPPs/Project2_Monet/AnimeGANData/nonMonetTest",
    transform=transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    )  
    
    BATCH_SIZE = 1
    
    train_loader = torch.utils.data.DataLoader(dataset_train         , 
                                               batch_size =BATCH_SIZE,
                                               shuffle    =True      ,
                                               num_workers=2         )
    
    test_loader = torch.utils.data.DataLoader(dataset_test          ,
                                              batch_size =BATCH_SIZE,
                                              shuffle    =True      ,
                                              num_workers=2         )    
    
    train_real_images = torch.utils.data.DataLoader(dataset_train_nonmonet,
                                                    batch_size =BATCH_SIZE,
                                                    shuffle    =True      ,
                                                    num_workers=2         )

    test_real_images = torch.utils.data.DataLoader(dataset_test_nonmonet ,
                                                   batch_size =BATCH_SIZE,
                                                   shuffle    =True      ,
                                                   num_workers=2         )    

    # Initialize Network Weights
    netG.apply(weights_init)
    print(netG)
    
    netD.apply(weights_init)
    print(netD)
    
    criterion = nn.BCELoss()
    criterion_MSE = nn.MSELoss()
    
    # Discriminator Input
    input = torch.FloatTensor(BATCH_SIZE, 3, 256, 256)
    # Generator Input
    noise = torch.FloatTensor(BATCH_SIZE, 3, 256, 256)
    
    label = torch.FloatTensor(BATCH_SIZE)
    real_label = 1
    fake_label = 0
        
    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    
    criterion = nn.BCELoss()
    
    max_epochs = 20
    
    # Constant for the gradient penalty portion of this work
    lambda_ = 10
    
    # capture discrimnator error and generator error for plotting
    discriminator_error = []
    generator_error = []
    
    # capture test and train accuracies for plotting
    test_accuracy = []
    train_accuracy = []
    
    temp, _ = next( iter( dataset_train_nonmonet ) )
    
    fixed_image = temp.unsqueeze(0)
    
    for epoch in range(0, max_epochs):
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0   
        index = 0 
        # for each image in train data
        for data, real_data in zip( iter( train_loader ), iter( dataset_train_nonmonet ) ):
            start_iter = time.time()

            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            
            input.resize_(real_cpu.size()).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label - 0.1)
            
            output = netD(input)
            
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()
            # train with fake
            real_image, _ = real_data
            real_image = real_image.unsqueeze(0)
            noise.resize_(real_image.size()).copy_(real_image) 

            fake = netG(noise)
            label.data.fill_(fake_label)
            
            output = netD(fake.detach())
            
            errD_fake = criterion(output, label)
            errD_fake.backward() # gradients for fake/real will be accumulated
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step() 
            
            
            # Gradient Penalty
            alpha = torch.rand(batch_size, 1).expand(input.size())
            
            temp = ( 
                input.data + 0.5 * input.data.std() * torch.rand(input.size())
                    )
            
            x_hat = Variable(     alpha  * input.data + \
                             (1 - alpha) * temp, 
                             requires_grad=True)
            pred_hat = netD(x_hat) 
            gradients = grad(outputs      = pred_hat                   , 
                             inputs       = x_hat                      , 
                             grad_outputs = torch.ones(pred_hat.size()),
                             create_graph = True                       , 
                             retain_graph = True                       , 
                             only_inputs  = True                       )[0]
            
            gradient_penalty = lambda_ * (
                (gradients.norm(2, dim=1) - 1) ** 2
                ).mean()
            gradient_penalty.backward()

            optimizerD.step()
    
            netG.zero_grad()
            label.data.fill_(real_label)
            output = netD(fake)
            
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()
            
            end_iter = time.time()
            
            discriminator_error.append( errD.item() )
            generator_error.append( errG.item() )
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
                  % (epoch              , 
                     max_epochs         , 
                     index              , 
                     len(train_loader)  ,
                     errD.item()        , 
                     errG.item()        ,
                     D_x                ,
                     D_G_z1             ,
                     D_G_z2             ,
                     end_iter-start_iter))
            
            if index % 50 == 0:
                # the first 64 samples from the mini-batch are saved.
                vutils.save_image(real_cpu[0:64,:,:,:],
                        '%s/real_samples.jpg' % (OUTPUT_DIR), nrow=8)
                
                vutils.save_image(fixed_image[0:64,:,:,:],
                        '%s/fixed_image.jpg' % (OUTPUT_DIR), nrow=8)                
                
                fake = netG(fixed_image)
                vutils.save_image(fake.data[0:64,:,:,:],
                        '%s/fake_samples_epoch_%03d_%d.jpg' 
                        % ((OUTPUT_DIR) ,
                           epoch        ,
                           index + epoch), nrow=8)
                
            index += 1
                
        # Test Discriminator
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # for each image in test data
        for data, real_data in zip( iter( test_loader ), iter( dataset_test_nonmonet ) ):
            
            image, _ = data
            input.resize_(image.size()).copy_(image)             
            prediction_real = netD.forward(input).detach()

            real_image, _ = real_data
            real_image = real_image.unsqueeze(0)
            noise.resize_(real_image.size()).copy_(real_image) 
            
            fake = netG.forward(noise).detach()
            prediction_fake = netD.forward( fake ).detach()          
            
            if (prediction_real[ 0 ] ) >= 0.5: 
                TP += 1
            else:
                FP += 1
                
            if (prediction_fake[ 0 ] ) < 0.5:
                TN += 1
            else:
                FN += 1
                
        temp =(TP+TN)/(TP+TN+FN+FP) 
        test_accuracy.append( temp )  
        

    print( "TP: %d TN %d FP %d FN %d" % (TP,TN,FP,FN))
        
    plt.plot( range( 0, len( test_accuracy ) ), 
              test_accuracy                   ,
              color='tab:green'               )
    
    plt.title(  "Test Accuracy" )
    plt.xlabel( "Epoch"         )
    plt.ylabel( "Accuracy"      )
    
    plt.savefig( OUTPUT_DIR + "Accuracy.jpg",
                 bbox_inches="tight"        )  

    plt.show()            
        
    plt.plot( range( 0, len( discriminator_error ) ),
              discriminator_error                   )
    
    plt.title(  "Discriminator Error" )
    plt.xlabel( "Over All Epochs"     )
    plt.ylabel( "Error"               )
    
    plt.savefig( OUTPUT_DIR + "DiscrimnatorError.jpg",
                 bbox_inches="tight"                 )
    
    plt.show()
    
    plt.plot( range( 0, len( generator_error ) ),
              generator_error                   )
    
    plt.title(  "Generator Error" )
    plt.xlabel( "Over All Epochs" )
    plt.ylabel( "Error"           )
    
    plt.savefig( OUTPUT_DIR + "GeneratorError.jpg",
                 bbox_inches="tight"              )        
    
    
    
    