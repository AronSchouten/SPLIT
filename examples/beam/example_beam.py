import numpy as np
import scipy.io as sio


def get_data(train, test):
    data_train = sio.loadmat(train)
    data_test = sio.loadmat(test)

    training_data = {'t': data_train['t'].flatten(),
                     'x': data_train['x'].T,
                     'dx': data_train['dx'].T,
                     'ddx': data_train['ddx'].T}
    
    val_data = {'t': data_test['t'].flatten(),
                     'x': data_test['x'].T,
                     'dx': data_test['dx'].T,
                     'ddx': data_test['ddx'].T}
    
    linear_parameters = {'m': data_train['m'].item(),
                         'c': data_train['c'].item(),
                         'k': data_train['k'].item(),
                         'phi1': data_train['phi1'].T.astype(np.float32),
                         'phi1TMc': (data_train['Mc'].T*data_train['phi1']).astype(np.float32)}
    
    return training_data, val_data, linear_parameters