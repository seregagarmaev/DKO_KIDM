battery_varying_load_params = {
    'path': 'data/',
    'n_train': 1,
    'n_test': 1,
    'features_to_scale': ['temperature', 'v', 'q', 'R0', 'inputs'],
    'number_of_trajectories': 2,
}

kidm_battery_params = {
    'n_epochs': 10,
    'window_size': 200,
    'device': 'cuda:2',
    'batch_size': 64,
    'indim': 200*3,
    'hidden_dim': 100,
    'obsdim': 5,
    'control_dim': 200,
    'outdim': 200*2,
    'lr': 1e-4,
    'early_stopping': 5,
    'weight_decay': 1e-7,
    'horizon': 10,
    'subsampling_step': 1000,
    'sigma': 2,
}