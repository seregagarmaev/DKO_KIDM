import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def read_battery_data(params):
    train= []
    test = []

    for i in range(params['n_train'] + params['n_test']):
        battery_data = pd.read_feather(f'{params["path"]}/{i}.feather')
        battery_data['battery_number'] = i + 1
        if i < params['n_train']:
            train.append(battery_data)
        else:
            test.append(battery_data)
    train = pd.concat(train, axis=0).dropna()
    test = pd.concat(test, axis=0).dropna()
    return train, test

def filter_by_EOL(data, q_min=6080):
    data = data[data['q'] >= q_min].reset_index(drop=True)
    return data

def add_RUL(data, unit_column='battery_number'):
    """Adds RUL to Li-ion battery data."""
    eol_df = data.groupby(unit_column).max()['time'].reset_index()
    data = data.merge(eol_df.rename(columns={'time': 'EOL'}), how='left', on=unit_column)
    data['RUL'] = data['EOL'] - data['time']
    del data['EOL']
    return data

def scale_battery_data(train, test, params):
    rul_scaler = MinMaxScaler()
    train['RUL'] = rul_scaler.fit_transform(train[['RUL']])
    test['RUL'] = rul_scaler.transform(test[['RUL']])
    
    feature_scaler = StandardScaler()
    train.loc[:, params['features_to_scale']] = feature_scaler.fit_transform(train.loc[:, params['features_to_scale']])
    test.loc[:, params['features_to_scale']] = feature_scaler.transform(test.loc[:, params['features_to_scale']])
    
    return train, test

def prepare_battery_data(battery_varying_load_params):
    train, test = read_battery_data(battery_varying_load_params)
    
    train = filter_by_EOL(train)
    test = filter_by_EOL(test)
    
    train = add_RUL(train)
    test = add_RUL(test)
    
    train, test = scale_battery_data(train, test, battery_varying_load_params)
    
    return train, test