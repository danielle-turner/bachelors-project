from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import glob

load = True
model = 'nn_no_tsne_21sphere.h5'
split_ent = False
train = True

if not train and not load:
    print('Either train or load must be True - tests performed on initial '
          'weights will converge ')

print('*** LOADING FILES... ***')

files = glob.glob('PureWigs21\\PureWigs21\\*.dat')

X = []
Y = []

for file in files:
    W = np.genfromtxt(file, dtype=None, delimiter=',')
    for x in range(len(W)):
        X.append(W[x,:])
        if split_ent:
            if 'Sep' in file:
                Y.append(0)
            elif 'Pure' in file:
                Y.append(1)
            elif 'Max' in file:
                Y.append(2)
        else:
            if 'Sep' in file:
                Y.append(1)
            else:
                Y.append(0)

print('*** {} FILES LOADED ***'.format(len(X)))

X = np.array(X)
Y = np.asarray(Y)
Y.shape = [-1,1]

X_full = np.concatenate((X,Y), axis=1)
np.random.shuffle(X)
X = X_full[:19500,:441]
Y = X_full[:19500, 441]
test_X = X_full[19500:,:441]
test_Y = X_full[19500:, 441]

if load:
    print('*** LOADING MODEL... ***')
    model = load_model(model)
    print('*** MODEL LOADED ***')
else:
    print('*** BUILDING MODEL... ***')
    model = Sequential()
    model.add(Dense(12, input_dim=441, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mse'])
    print('*** MODEL BUILT ***')

if train:
    print('*** FITTING MODEL... ***')
    model.fit(X, Y, epochs=100, batch_size=10)
    print('*** MODEL FIT ***')
    
scores = model.evaluate(test_X, test_Y)
