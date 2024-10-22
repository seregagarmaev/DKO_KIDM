import numpy as np
import torch
import torch.nn as nn
from src.utils import BatteryDataset, get_eigenvalues
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

class KIDM(nn.Module):
    def __init__(self, params):
        super(KIDM, self).__init__()
        self.params = params
        self.lr = LinearRegression()
        
        self.encoder = nn.Sequential(
            nn.Linear(params['indim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim']),
        )
        
        self.control_encoder = nn.Sequential(
            # nn.Linear(params['control_dim'], params['hidden_dim']),
            nn.Linear(params['control_dim']+params['obsdim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['obsdim']*params['obsdim']),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(params['obsdim']+params['control_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.SELU(inplace=True),
            nn.Linear(params['hidden_dim'], params['outdim']),
        )
    
    def encode(self, x0t0, u0):
        inputs = torch.cat((x0t0, u0), dim=1)
        y0 = self.encoder(inputs)
        return y0
    
    def encode_control(self, y0, u1):
        inputs = torch.cat((y0, u1), dim=1)
        operators = self.control_encoder(inputs)
        
        operators = operators.view((operators.size(0), self.params['obsdim'], self.params['obsdim']))
        return operators
    
    def decode(self, y, u):
        inputs = torch.cat((y, u), dim=1)
        x_hat = self.decoder(inputs)
        return x_hat
    
    def forward(self, x0t0, u0, u1):
        y0 = self.encode(x0t0, u0)
        y1_hat = self.next_observables(y0, u1)
        x1t1_hat = self.decode(y1_hat, u1)
        return x1t1_hat
    
    def reconstruct(self, x0t0, u0):
        y0 = self.encode(x0t0, u0)
        x0t0_hat = self.decode(y0, u0)
        return x0t0_hat
    
    def next_observables(self, y0, u1):
        K = self.encode_control(y0, u1)
        y1_hat = torch.bmm(K, y0.view(y0.size(0), y0.size(1), 1)).view(y0.size(0), y0.size(1))
        return y1_hat
    
    def predict_RUL(self, data, test_data, train_batteries):
        self.eval()
        
        train_dataset = BatteryDataset(data, self.params['window_size'], self.params['horizon'], self.params['subsampling_step'])
        rul_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False)
        train_sample = next(iter(rul_dataloader))
        
        test_dataset = BatteryDataset(test_data, self.params['window_size'], self.params['horizon'], self.params['subsampling_step'])
        rul_test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)
        test_sample = next(iter(rul_test_dataloader))
        
        train_observables_kidm = self.encode(
        torch.cat((train_sample['x0'], train_sample['t0']), dim=1).to(self.params['device']),
        train_sample['u0'].to(self.params['device']))
        
        test_observables_kidm = self.encode(
        torch.cat((test_sample['x0'], test_sample['t0']), dim=1).to(self.params['device']),
        test_sample['u0'].to(self.params['device']))
        
        train_observables_kidm = train_observables_kidm.to('cpu').detach().numpy()
        test_observables_kidm = test_observables_kidm.to('cpu').detach().numpy()
        
        train_batteries = np.unique(train_sample['battery_number'])
        test_batteries = np.unique(test_sample['battery_number'])

        for train_battery in train_batteries:
            index = train_sample['battery_number'] == train_battery
            train_observables_kidm[index] = gaussian_filter1d(train_observables_kidm[index], self.params['sigma'], axis=0)

        for test_battery in test_batteries:
            index = test_sample['battery_number'] == test_battery
            test_observables_kidm[index] = gaussian_filter1d(test_observables_kidm[index], self.params['sigma'], axis=0)
            
        self.lr.fit(train_observables_kidm, train_sample['RUL'].numpy())
        lr_pred = self.lr.predict(test_observables_kidm)
        return lr_pred, test_sample['RUL'].numpy(), test_sample['time'].numpy()
    
    def get_eigenvalues(self, data):
        self.eval()
        dataset = BatteryDataset(data, self.params['window_size'], self.params['horizon'], self.params['window_size'])
        dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
        sample = next(iter(dataloader))
        
        y0 = self.encode(
            torch.cat((sample['x0'], sample['t0']), dim=1).to(self.params['device']),
            sample['u0'].to(self.params['device'])
        )
        KoopmanOperator_pred = self.encode_control(
            y0,
            sample['u1'].to(self.params['device'])
        ).view((-1, self.params['obsdim'], self.params['obsdim']))
        KoopmanOperator_pred = KoopmanOperator_pred.to('cpu').detach().numpy()
        eigvals, eigvecs = get_eigenvalues(KoopmanOperator_pred)
        return eigvals, eigvecs