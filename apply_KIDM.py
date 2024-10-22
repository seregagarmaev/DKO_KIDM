from config import battery_varying_load_params, kidm_battery_params
from src.data_processing import prepare_battery_data
from src.utils import gen_dataloader, train_KIDM, BatteryDataset
from src.models import KIDM
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def main():
    train, test = prepare_battery_data(battery_varying_load_params)

    train_dataloader = gen_dataloader(train, kidm_battery_params)
    test_dataloader = gen_dataloader(test, kidm_battery_params, shuffle=False)

    model = KIDM(kidm_battery_params).to(kidm_battery_params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=kidm_battery_params['lr'], weight_decay=kidm_battery_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    kidm = train_KIDM(model, optimizer, scheduler, train_dataloader, test_dataloader, kidm_battery_params['n_epochs'], early_stopping=kidm_battery_params['early_stopping'])


    # Predict RUL
    train_batteries = np.random.choice(train['battery_number'].unique(), 1, replace=False)
    rul_train = train[train['battery_number'].isin(train_batteries)]
    test_batteries = test['battery_number'].unique()
    rul_test = test[test['battery_number'] == test_batteries[0]]
    rul_pred, rul_true, timesteps = model.predict_RUL(rul_train, rul_test, train_batteries)

    plt.figure(figsize=(6,4))
    plt.plot(timesteps, rul_pred, label='RUL pred')
    plt.plot(timesteps, rul_true, label='RUL ref')
    plt.xlabel('Timestep')
    plt.ylabel('Normalized RUL')
    plt.legend()
    plt.grid()
    plt.savefig('RUL_predicted.png')
    plt.show()

    # Get Eigenvalues of the predicted Koopman Operators
    eigvals, eigvecs = model.get_eigenvalues(test)

    plt.figure(figsize=(6, 4))
    plt.scatter([x.real for x in eigvals], [x.imag for x in eigvals], color='tab:blue', alpha=0.1, label=r'$I_d \sim U(1V, 1.5V)$')
    plt.xlabel(r'Re($\lambda$)')
    plt.ylabel(r'Im($\lambda$)')
    plt.legend()
    plt.grid()
    plt.savefig('KO_eigenvalues.png')
    plt.show()
    
if __name__ == "__main__":
    main()