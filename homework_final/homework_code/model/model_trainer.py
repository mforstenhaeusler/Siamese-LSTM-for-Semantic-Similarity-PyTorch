from typing import Coroutine
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time 
import wandb

from .utils import plotConfusionMatrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(
        self, 
        model, 
        hparams, 
        train_dataloader, 
        val_dataloader, 
        train_indices, 
        val_indices, 
        log_to_wandb,
        lr_scheduler_enabler=True,
        ):
        """ 
        This Class fits the model 

        Params:
        -------  
        model : nn.Module
                Pytorch NN Model that is spposed to be fitted/trained
        hparams : dict
                  Dictionary of Hyperparametes  
        train_dataloader : torch.utils.data.DataLoader
                           Training DataLoader
        val_dataloader : torch.utils.data.DataLoader
                         Validation DataLoader 
        test_dataloader : torch.utils.data.DataLoader
                          Testing DataLoader
        train_indices : list 
        val_indices :  list
        test_indices : list 
        lr_scheduler_enabler : bool
                               if True enables Learning rate scheduler, if False disables it
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler_enabler = lr_scheduler_enabler
        self.hparams = hparams 
        self.learning_rate = hparams['learning_rate']
        self.epochs = hparams['epoch']
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.log_to_wandb = log_to_wandb

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == "cuda":
            self.threshold = hparams['threshold'].to(self.device)
        else:
            self.threshold = hparams['threshold'].to(self.device)

        self.model = model
        self.optimizer = self.optimization()
        self.loss_fn = self.loss()
        self.lr_scheduler = self.learning_rate_scheduler() 

        self.data = dict()
        self.data["train_loss"] = list()
        self.data["train_acc"] = list()
        self.data["val_loss"] = list()
        self.data["val_acc"] = list()

        if self.log_to_wandb:
            self.init_wandb()


    def train_epoch(self, epoch):
        """ Trains an epoch """
        self.model.train()

        loss_history = []
        correct_total = 0
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch [{epoch+1}/{self.epochs}]  Training")
                if self.device == "cuda":
                    q1, q2 = batch['q1_token'].to(self.device), batch['q2_token'].to(self.device)
                    q1_len, q2_len = batch['q1_lengths'].to(self.device), batch['q2_lengths'].to(self.device)
                    y = torch.FloatTensor(batch['labels']).to(self.device)
                else:
                    q1, q2 = batch['q1_token'], batch['q2_token']
                    q1_len, q2_len = batch['q1_lengths'], batch['q2_lengths']
                    y = torch.FloatTensor(batch['labels'])
                
                # Reset the gardients 
                self.optimizer.zero_grad()

                # Model forward and predictions
                similarity = self.model(q1, q2, q1_len, q2_len)
                y_pred = (similarity > self.threshold).float() * 1
                correct = self.inferece(y_pred, y)
                correct_total += correct

                # Calculate the loss 
                loss = self.loss_fn(similarity, y)
                loss_history.append(loss.item())

                # Calculate gradients by performign the backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()

                if i % 100 == 0:
                    tepoch.set_postfix(train_loss=np.mean(loss_history), train_acc=f'{(correct/y.size()[0])*100} %' )
            
            # Enable learning rate scheduler  
            if self.lr_scheduler_enabler:
                self.lr_scheduler.step()

        return  np.mean(loss_history), (correct_total/len(self.train_indices))*100

    def evaluate(self):
        """ Validates an epoch """
        self.model.eval()

        loss_history = []
        correct_total = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if self.device == "cuda":
                    q1, q2 = batch['q1_token'].to(self.device), batch['q2_token'].to(self.device)
                    q1_len, q2_len = batch['q1_lengths'].to(self.device), batch['q2_lengths'].to(self.device)
                    y = torch.FloatTensor(batch['labels']).to(self.device)
                else:
                    q1, q2 = batch['q1_token'], batch['q2_token']
                    q1_len, q2_len = batch['q1_lengths'], batch['q2_lengths']
                    y = torch.FloatTensor(batch['labels'])

                # Model forward and predictions
                similarity = self.model(q1, q2, q1_len, q2_len)
                y_pred = (similarity > self.threshold).float() * 1
                correct = self.inferece(y_pred, y)
                correct_total += correct

                # Calculate the loss 
                loss = self.loss_fn(similarity, y)
                loss_history.append(loss.item())

        avg_val_acc =  correct_total/len(self.val_indices) * 100 
        return np.mean(loss_history), avg_val_acc
    
    def inferece(self, y_pred, y):
        """ Performs inference """
        return (y_pred == y).sum().item()

    def fit(self):
        """ Fits the model """
        train_loss = 0
        val_loss = 0
        val_acc = 0
        for e in range(self.epochs):
            train_loss, train_acc = self.train_epoch(e)
            val_loss, val_acc = self.evaluate()
            print(f'Epoch [{e+1}/{self.epochs}] Validation: val_loss: {val_loss} val_acc: {val_acc} %')
            
            self.data["train_loss"].append(train_loss)
            self.data["train_acc"].append(train_acc)
            self.data["val_loss"].append(val_loss)
            self.data["val_acc"].append(val_acc)

            if self.log_to_wandb:
                self.log_metrics_to_wandb(train_loss, train_acc, val_loss, val_acc)
            time.sleep(0.5)

        if self.log_to_wandb:
            wandb.finish()

    def test(self):
        """ Tests the model """
        self.model.eval()

        predictions = []
        labels_list = []
        loss_history = []
        correct_total = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if self.device == "cuda":
                    q1, q2 = batch['q1_token'].to(self.device), batch['q2_token'].to(self.device)
                    q1_len, q2_len = batch['q1_lengths'].to(self.device), batch['q2_lengths'].to(self.device)
                    y = torch.FloatTensor(batch['labels']).to(self.device)
                else:
                    q1, q2 = batch['q1_token'], batch['q2_token']
                    q1_len, q2_len = batch['q1_lengths'], batch['q2_lengths']
                    y = torch.FloatTensor(batch['labels'])

                # Model forward and predictions
                similarity = self.model(q1, q2, q1_len, q2_len)
                y_pred = (similarity > self.threshold).float() * 1
                predictions.append(y_pred), labels_list.append(y)
                correct = self.inferece(y_pred, y)
                correct_total += correct

                # Calculate the loss 
                loss = self.loss_fn(similarity, y)
                loss_history.append(loss.item())
        
        # Calculate the accuracy
        avg_val_acc =  correct_total/len(self.val_indices) * 100 
        print('- - - Model Performance - - -')
        print(f'\nModel Accuracy:  {avg_val_acc}')
        print(f'Correct predictions: {correct_total}, Incorret predictions: {len(self.val_indices) - correct_total}')
        print('')
        cm = plotConfusionMatrix(np.hstack(predictions), np.hstack(labels_list),['similar', 'dissimilar'], title="Confusion Matrix Plot of Test Set")
        print(f'TP: {cm[0,0]}')
        print(f'FP: {cm[1,0]}')
        print(f'FN: {cm[0,1]}')
        print(f'TN: {cm[1,1]}')
        print(f'\nPercision Score: {precision_score(np.hstack(predictions), np.hstack(labels_list))}')
        print(f'Recall Score: {recall_score(np.hstack(predictions), np.hstack(labels_list))}')
        print(f'F1 Score: {f1_score(np.hstack(predictions), np.hstack(labels_list))}')

        # adopted from https://www.codegrepper.com/code-examples/python/roc+curve+pytorch
        fpr, tpr, threshold = roc_curve(np.hstack(predictions), np.hstack(labels_list))
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid()
        plt.show()
        # adopted from https://www.codegrepper.com/code-examples/python/roc+curve+pytorch

    def predict(self, test_sample_dict):
        """ Uses the model to predict the similarity of a given input pair of questions"""
        self.model.eval()
        
        print('question 1:', test_sample_dict['q1_text'])
        print('question 2:', test_sample_dict['q2_text'])
        print('tokens  q1:', test_sample_dict['q1_token'])
        print('tokens  q2:', test_sample_dict['q2_token'])

        q1, q2 = test_sample_dict['q1_token'], test_sample_dict['q2_token']
        q1_len, q2_len = test_sample_dict['q1_lengths'], test_sample_dict['q2_lengths']
        y = torch.FloatTensor(test_sample_dict['labels'])
        
        # Model forward and predictions
        similarity = self.model(q1, q2, q1_len, q2_len)
        y_pred = (similarity > self.threshold).float() * 1
        
        print(f'\n\nModel predicts {y_pred.item()} --> Actual value {y.item()}')
        if y_pred.item() == y.item():
            print(f'Model prediction is correct :)')

            if y_pred.item() == 1.0:
                print(f'\nThe questions {test_sample_dict["q1_text"]} and {test_sample_dict["q2_text"]} are similar!')
            else:
                print(f'\nThe questions {test_sample_dict["q1_text"]} and {test_sample_dict["q2_text"]} are dissimilar!')    
        else:
            print(f'Model prediction is inaccurate :(')
            if y_pred.item() == 1.0:
                print(f'\nThe questions {test_sample_dict["q1_text"]} and {test_sample_dict["q2_text"]} should be dissimilar!')
            else:
                print(f'\nThe questions {test_sample_dict["q1_text"]} and {test_sample_dict["q2_text"]} should be similar!')  
        
    def optimization(self):
        """ Initializes the optimizer """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def learning_rate_scheduler(self):
        """ Initializes the learning rate scheduler """
        return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def loss(self):
        """ Initializes the loss """
        return nn.MSELoss() #nn.CrossEntropyLoss()
    
    def return_data(self):
        """ Output the data """
        return self.data

    def init_wandb(self):
        """ init weight & biases """
        # capture hyperparameters
        config = self.hparams
        # initialize wandb
        wandb.init(project="IBM-Praktikum Homework 2", entity="maxifor", config=config)

    def log_metrics_to_wandb(self, train_loss, train_acc, val_loss, val_acc):
        """ log metric to weights and biases """
        wandb.log({"train_loss": train_loss,})
        wandb.log({"train_acc": train_acc,})
        wandb.log({"val_loss": val_loss,})
        wandb.log({"val_acc": val_acc,})

    