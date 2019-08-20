#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse

import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict



class Net(object):
    
    def __init__(self, trainloaders, validloaders, testloaders, model_type, epochs, learning_rate, hidden_units, device):
        self.model_type = model_type
        self.trainloaders = trainloaders
        self.validloaders = validloaders
        self.testloaders = testloaders
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.device = self._check_device(device)
    
    def _check_device(self, device):
        if torch.cuda.is_available() and device == 'cuda':
            return 'cuda'
        else:
            return 'cpu'
    
    def generate_model(self, model_type):
        
        model = self._model_define(model_type)
        
        # 冻结特征工程，不需要重新训练    
        for param in model.parameters():
            param.required_grad = False
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        # optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001, momentum=0.9)
        return model, criterion, optimizer
    
    # TODO: Build and train your network
    def train(self, model, criterion, optimizer):
        print_every = 33
        steps = 0
        running_loss = 0
        # change to cuda
        model.to(self.device)

        for e in range(self.epochs):
        #     if e == 2:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = param_group['lr'] * 0.5

            for ii, (inputs, labels) in enumerate(self.trainloaders):
                if ii % 2 == 0:
                    print('====currenti', ii)
                steps += 1
        #         print(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss, valid_accuracy = self.valid_train(model, self.validloaders, criterion)
                    print("Epoch: {}/{}... ".format(e + 1, epochs),
                          "Train Loss: {:.4f}.. ".format(running_loss / print_every),
                          "Valid Loss: {:.4f}.. ".format(valid_loss),
                          "Valid Accuracy: {:.2f}% .. ".format(valid_accuracy * 100),)

                    running_loss = 0
                    model.train()
    
    def valid_train(self, models, dataloaders, criterion):
    
        valid_loss = 0 
        valid_accuracy = 0
        valid_len = len(dataloaders)

        model.to(self.device)
        models.eval()   # 取消dropout

        with torch.no_grad():

            for ii, (images, labels) in enumerate(dataloaders):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = models.forward(images)

                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(1)[1])
                valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

        return valid_loss / valid_len, valid_accuracy / valid_len
    
    def valid_model(self):
        result = self.valid_train(model, self.testloaders, criterion)
        print('Loss: {:.4f}..   Accuracy: {:.2f}% .. '.format(result[0], result[1] * 100))
     
    def save_checkpoint(self, model, criterion, optimizer, filepath, cat_to_name):
        checkpoint = {
                'state_dict': model.state_dict(),
        #         'class_to_idx': train_datasets.class_to_idx,
                'class_to_idx': cat_to_name,  # 不确定是不是这样转
                'criterion': criterion,
                'optimizer': optimizer,
                'model_type': self.model_type
            }
        torch.save(checkpoint, filepath)
        return True
    
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    
    def _model_define(self, model_type):
        if model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
            # model = models.resnet50(pretrained=True)
            # VGG16
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(25088, 4096)),
                ('relu1', nn.ReLU()),
                ('drop1', nn.Dropout(0.5)),
                ('fc2', nn.Linear(4096, self.hidden_units)),
                ('relu2', nn.ReLU()),
                ('drop2', nn.Dropout(0.5)),
                ('fc3', nn.Linear(self.hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
        elif model_type == 'resnet':
            # Resnet
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(2048, self.hidden_units)),
                ('relu1', nn.ReLU()),
                ('drop1', nn.Dropout(0.5)),
                ('fc3', nn.Linear(self.hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
        return model
    
    def load_model(self, filepath):
        
        state_dict = torch.load(filepath, map_location=lambda storage, loc: storage)
        
        model = self._model_define(state_dict['model_type'])
        model.classifier = classifier
        model.class_to_idx = state_dict['class_to_idx']
        model.load_state_dict(state_dict['state_dict'])
        criterion = state_dict['criterion']
        optimizer = state_dict['optimizer']
        return model, criterion, optimizer

    
    def main(self, load_file=None):
        if load_file:
            model, criterion, optimizer = load_model(load_file)
        model, criterion, optimizer = self.generate_model(self.model_type)
        self.train(model, criterion, optimizer)
        print()
        print()
        self.valid_model()
            