import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.set_default_dtype(torch.float64)


class BatteryDataset(Dataset):
    """Li-ion battery dataset."""

    def __init__(self, df, window_size, horizon, subsampling_step, transform=None):
        """
        Arguments:
            df (pandas.DataFrame): Dataframe with simulated Li-ion batteries.
            window_size (int): Size of the slicing window.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df.copy()
        self.ss = subsampling_step
        self.ids = self.get_ids(df, window_size, horizon)
        self.horizon = horizon
        self.ws = window_size
        self.transform = transform

    def get_ids(self, data, ws, horizon):
        data = data.copy()
        batteries = data['battery_number'].unique()
        ids = pd.DataFrame()
        for battery in batteries:
            battery_data = data[data['battery_number'] == battery]
            l = len(battery_data)
            r = l % ws
            battery_data = battery_data.iloc[:l-r-ws*horizon, :]
            ids = pd.concat((ids, battery_data[[]]), axis=0)
        ids = ids.iloc[::self.ss, :]
        ids['id'] = np.arange(len(ids))
        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        index = self.ids[self.ids['id'] == idx].index[0]
        
        sample = {}
        sample.update({f"x{n}": np.empty(shape=[self.ws,]) for n in range(self.horizon+1)})
        sample.update({f"t{n}": np.empty(shape=[self.ws,]) for n in range(self.horizon+1)})
        sample.update({f"u{n}": np.empty(shape=[self.ws,]) for n in range(self.horizon+1)})
        

        for n in range(self.horizon+1):
            sample[f'x{n}'][:] = self.data.loc[index+self.ws*n:index+self.ws*n+self.ws-1, 'v']
            sample[f't{n}'][:] = self.data.loc[index+self.ws*n:index+self.ws*n+self.ws-1, 'temperature']
            sample[f'u{n}'][:] = self.data.loc[index+self.ws*n:index+self.ws*n+self.ws-1, 'inputs']
        sample['q'] = self.data.loc[index+self.ws, 'q']
        sample['R0'] = self.data.loc[index+self.ws, 'R0']
        sample['RUL'] = self.data.loc[index+self.ws, 'RUL']
        sample['time'] = self.data.loc[index+self.ws, 'time']
        sample['battery_number'] = self.data.loc[index+self.ws, 'battery_number']

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def gen_dataloader(data, params, shuffle=True):
    dataset = BatteryDataset(data, params['window_size'], params['horizon'], params['subsampling_step'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=shuffle)
    return dataloader

def trainer(model, optimizer, dataloader):
    model.train()
    device = model.params['device']
    mseLoss = nn.MSELoss()
    horizon = model.params['horizon']
    total_loss = 0
    final_prediction_loss = 0
    final_reconstruction_loss = 0
    n_batches = len(dataloader)
    
    for sample in dataloader:
        # reconstruction loss
        x0t0 = torch.cat((sample['x0'], sample['t0']), dim=1).to(device)
        x0t0_hat = model.reconstruct(x0t0, sample['u0'].to(device))
        reconstruction_loss = mseLoss(x0t0, x0t0_hat)
        
        # prediction loss
        prediction_loss = 0
        obs_loss = 0
        y0_hat = model.encode(x0t0, sample['u0'].to(device))
        for i in range(1, horizon+1):
            x1t1 = torch.cat((sample[f'x{i}'], sample[f't{i}']), dim=1).to(device)
            y1_hat = model.next_observables(y0_hat, sample[f'u{i}'].to(device))
            y1 = model.encode(x1t1, sample[f'u{i}'].to(device))
            obs_loss += mseLoss(y1, y1_hat)
            if i % 1 == 0:
                x1t1_hat = model.decode(y1_hat, sample[f'u{i}'].to(device))
                prediction_loss += mseLoss(x1t1, x1t1_hat)
            y0_hat = y1_hat
        
        loss =  prediction_loss + reconstruction_loss + obs_loss
        
        final_prediction_loss += prediction_loss.item()
        final_reconstruction_loss += reconstruction_loss.item()
        total_loss = total_loss + loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        # plot_grad_flow(model.cpu().named_parameters())
        # model.to(device)
        optimizer.step()

    return total_loss / n_batches, (final_prediction_loss / n_batches, final_reconstruction_loss / n_batches)

def loss_calculator(model, dataloader):
    model.eval()
    device = model.params['device']
    mseLoss = nn.MSELoss()
    horizon = model.params['horizon']
    total_loss = 0
    final_prediction_loss = 0
    final_reconstruction_loss = 0
    n_batches = len(dataloader)
    
    for sample in dataloader:
        # reconstruction loss
        x0t0 = torch.cat((sample['x0'], sample['t0']), dim=1).to(device)
        x0t0_hat = model.reconstruct(x0t0, sample['u0'].to(device))
        reconstruction_loss = mseLoss(x0t0, x0t0_hat)
        
        # prediction loss
        prediction_loss = 0
        obs_loss = 0
        y0_hat = model.encode(x0t0, sample['u0'].to(device))
        for i in range(1, horizon+1):
            x1t1 = torch.cat((sample[f'x{i}'], sample[f't{i}']), dim=1).to(device)
            y1_hat = model.next_observables(y0_hat, sample[f'u{i}'].to(device))
            y1 = model.encode(x1t1, sample[f'u{i}'].to(device))
            obs_loss += mseLoss(y1, y1_hat)
            if i % 1 == 0:
                x1t1_hat = model.decode(y1_hat, sample[f'u{i}'].to(device))
                prediction_loss += mseLoss(x1t1, x1t1_hat)
            y0_hat = y1_hat
        
        # fast dynamic constraint
        # fast_dynamic_constraint = 0
        # y0_hat = model.encode(x0t0, sample['u0'].to(device))
        # for i in range(1, horizon+1):
        #     x1t1 = torch.cat((sample[f'x{i}'], sample[f't{i}']), dim=1).to(device)
        #     y1_hat = model.encode(x1t1, sample[f'u{i}'].to(device))
        #     # fast_dynamic_constraint += mseLoss(y0_hat, y1_hat)
        #     fast_dynamic_constraint = torch.mean(F.relu(y1_hat[:, 3:] - y0_hat[:, 3:]))
        #     y0_hat = y1_hat
        
        loss =  prediction_loss + reconstruction_loss + obs_loss
        
        final_prediction_loss += prediction_loss.item()
        final_reconstruction_loss += reconstruction_loss.item()
        total_loss = total_loss + loss.item()

    return total_loss / n_batches, (final_prediction_loss / n_batches, final_reconstruction_loss / n_batches)

def train_KIDM(model, optimizer, scheduler, train_dataloader, test_dataloader, n_epochs, early_stopping=5):
    prev_test_loss = []
    steps_without_improvement = 0
    for epoch in range(1, n_epochs+1):
        train_loss, (prediction_loss, reconstruction_loss) = trainer(model, optimizer, train_dataloader)
        train_loss, (prediction_loss, reconstruction_loss) = loss_calculator(model, train_dataloader)
        print(f'Epoch {epoch}:')
        print('Training MSE loss:', round(train_loss, 4))
        print('Prediction MSE Loss:', round(prediction_loss, 4))
        print('Reconstruction MSE Loss:', round(reconstruction_loss, 4))

        test_loss, (test_prediction_loss, test_reconstruction_loss) = loss_calculator(model, test_dataloader)
        print()
        print('Test MSE loss:', round(test_loss, 4))
        print('Test prediction MSE Loss:', round(test_prediction_loss, 4))
        print('Test reconstruction MSE Loss:', round(test_reconstruction_loss, 4))
        print('=' * 50)
        scheduler.step(test_loss)

        if epoch > early_stopping:
            if test_loss > min(prev_test_loss):
                if steps_without_improvement < early_stopping:
                    steps_without_improvement += 1
                    prev_test_loss = prev_test_loss[1:] + [test_loss]
                else:
                    break
            else:
                steps_without_improvement = 0
                prev_test_loss = prev_test_loss[1:] + [test_loss]
        else:
            prev_test_loss.append(test_loss)
    return model

def get_eigenvalues(data):
    values = []
    vectors = []
    for i in range(data.shape[0]):
        matrix = data[i]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        values += eigenvalues.tolist()
        vectors += eigenvectors.tolist()
    return np.array(values), np.array(vectors)