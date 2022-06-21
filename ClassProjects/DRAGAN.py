# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
from torch.autograd import Variable, grad
from torch.nn.init import xavier_normal
from torchvision import datasets, transforms
import torchvision.utils as vutils
import shutil
import os

import matplotlib.pyplot as plt

# SRC: https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py
def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            xavier_normal(param)

if __name__ == '__main__':
    
    OUTPUT_DIR = "D:/Classes/CS795MLAPPs/Projec2_Monet/DRAGANData/Output/"
    
    batch_size = 1
    z_dim = 256
    h_dim = 256
    y_dim = 256
    max_epochs = 100
    lambda_ = 10

    dataset_train = datasets.ImageFolder(
    root="D:/Classes/CS795MLAPPs/Projec2_Monet/DRAGANData/Train",
    transform=transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
    )
    
    dataset_test = datasets.ImageFolder(
    root="D:/Classes/CS795MLAPPs/Projec2_Monet/DRAGANData/Test",
    transform=transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
    )    

    train_loader = torch.utils.data.DataLoader(dataset_train        , 
                                               batch_size=batch_size,
                                               shuffle=True         , 
                                               num_workers=1        )
    
    test_loader = torch.utils.data.DataLoader(dataset_test         ,
                                              batch_size=batch_size,
                                              shuffle=True         ,
                                              drop_last=True       ,
                                              num_workers=1        )
    
    class Transpose4d(torch.nn.Module):
        def forward(self, input):
            return input[ 0 ][ 0 ].reshape( input[ 0 ][ 0 ].shape[ 1 ],
                                            input[ 0 ][ 0 ].shape[ 0 ])
        
    class Transpose2d(torch.nn.Module):
        def forward(self, input):
            return input.reshape( input.shape[ 1 ],
                                  input.shape[ 0 ])       
        
        
    generator = torch.nn.Sequential(
            torch.nn.Linear( 256, 512 ),
            torch.nn.Linear( 512, 1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.Linear(2048, 1024),
            torch.nn.Linear(1024, 512 ),
            torch.nn.Linear( 512, 256 )
            )
    
    discriminator = torch.nn.Sequential(
            torch.nn.Conv2d( 3, 256, 4, 2, 1 ),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.Sigmoid(),
            Transpose4d(),
            torch.nn.Linear(128, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
            Transpose2d(),  
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
            )

    # Init weight matrices (xavier_normal)
    xavier_init(generator)
    xavier_init(discriminator)

    # Generator Optimizer
    opt_g = torch.optim.Adam(generator.parameters())
    # Discriminator Optimizer
    opt_d = torch.optim.Adam(discriminator.parameters())

    criterion = torch.nn.BCELoss()
    # Input matrix real
    X = Variable(torch.FloatTensor(1,3, y_dim, h_dim))
    # Input matrix fake
    z = Variable(torch.FloatTensor(1,3, z_dim, z_dim))
    # ground_truth labels
    labels = Variable(torch.FloatTensor(batch_size))

    # capture error values throughout training
    discriminator_error = []
    generator_error = []
    
    # capture accuracies throughout training
    test_accuracy = []
    train_accuracy = []
    
    # Train
    for epoch in range(max_epochs):
        
        TP = 0
        FP = 0
        TN = 0
        FN = 0        
        # For each image in train set
        for batch_idx, (data, target) in enumerate(train_loader):
            X.data.copy_(data)
            # Update discriminator
            # train with real
            discriminator.zero_grad()
            pred_real = discriminator(X)
            labels.data.fill_(1.0)
            
            loss_d_real = criterion(pred_real[ 0 ], labels)
            loss_d_real.backward()
            
            if (pred_real[ 0 ] ) >= 0.5: 
                TP += 1
            else:
                FP += 1

            # train with fake
            z.data.normal_(0, 1)
            fake      = generator.forward(z).detach()
            pred_fake = discriminator(fake)
            
            labels.data.fill_(0.0)
            loss_d_fake = criterion(pred_fake[ 0 ], labels)
            loss_d_fake.backward()

            # gradient penalty
            alpha = torch.rand(batch_size, 1).expand(X.size())
            x_hat = Variable(alpha * X.data + \
                             (1 - alpha) * \
                                 (X.data + 0.5 * \
                                  X.data.std() * \
                                  torch.rand(X.size())), 
                             requires_grad=True
                            )
            pred_hat = discriminator(x_hat)
            gradients = grad(outputs     =pred_hat                   , 
                             inputs      =x_hat                      , 
                             grad_outputs=torch.ones(pred_hat.size()),
                             create_graph=True                       , 
                             retain_graph=True                       , 
                             only_inputs =True                       )[0]
            gradient_penalty = lambda_ * \
                                  ((gradients.norm(2, dim=1) - 1)**2).mean()
            gradient_penalty.backward()

            loss_d = loss_d_real + loss_d_fake + gradient_penalty
            opt_d.step()

            # Update generator
            generator.zero_grad()
            z.data.normal_(0, 1)
            gen = generator(z)
            pred_gen = discriminator(gen)
            labels.data.fill_(1.0)
            loss_g = criterion(pred_gen[ 0 ], labels)
            loss_g.backward()
            opt_g.step()
            
            if (pred_gen[ 0 ] ) < 0.5:
                TN += 1
            else:
                FN += 1

            discriminator_error.append( loss_d.item() )
            generator_error.append( loss_g.item() )
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch            , 
                     max_epochs       , 
                     batch_idx        , 
                     len(train_loader),
                     loss_d.item()    ,
                     loss_g.item()    ))

            if batch_idx % 100 == 0:
                vutils.save_image(data,
                        OUTPUT_DIR + "real_samples.jpg")
                fake = generator(z)
                vutils.save_image(fake,
                        OUTPUT_DIR + "fake_samples_epoch_%03d.jpg" % epoch)
                
        temp = (TP+TN)/(TP+TN+FN+FP) 
        train_accuracy.append( temp )
        
            
        # Test Discriminator
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # for each image in test set
        for batch_idx, (data, target) in enumerate(test_loader):
            
            prediction_real = discriminator.forward(data).detach()
            
            z.data.normal_(0, 1)
            fake = generator.forward(z).detach()
            prediction_fake = discriminator.forward( fake ).detach()          
            
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
        
    # Train and Test Accuracies Plot
    plt.plot( range( 0, len( train_accuracy ) ), 
              train_accuracy                   , 
              color='tab:red'                  )
    plt.plot( range( 0, len( test_accuracy  ) ), 
              test_accuracy                    , 
              color='tab:green'                )
    
    plt.title(  "Train and Test Accuracy" )
    plt.xlabel( "Epoch"                   )
    plt.ylabel( "Accuracy"                )
    
    plt.savefig( OUTPUT_DIR + "Accuracy.jpg",
                 bbox_inches='tight'        )
                
            
            
