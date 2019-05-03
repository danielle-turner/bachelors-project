import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import load_model
import keras
import glob
from sklearn import manifold
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from numpy.random import shuffle
import matplotlib.pyplot as plt

"""
    The definitions for encoder(input_data), decoder(conv3), and
            autoencoder(input_data), are adapted from
https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
"""
c = 1000
loss = 0

def encoder(input_data):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   name='01')(input_data)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='02')(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='03')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='04')(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   name='encoder')(pool2)
    return conv3

def decoder(conv3):
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='06')(conv3)
    up1 = UpSampling2D((2, 2),
                   name='07')(conv4)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   name='08')(up1)
    up2 = UpSampling2D((2, 2),
                   name='09')(conv5)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',
                   name='10')(up2)
    return decoded

def autoencoder(input_data):
    # Encoder
    conv = encoder(input_data)
    # Decoder
    decoded = decoder(conv)
    return decoded

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

class Loss:
    def __init__(self, c):
        self.c = c
        
    def set_loss(self, c):
        self.c = c

    def loss_function(self, y, return_results=False):
        data = []
        for result in y:
            data.append(np.array(result).flatten())
        data = np.array(data)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                             perplexity=30.0, learning_rate=200.0,
                             angle=0.01)
        X_tsne = tsne.fit_transform(data)
        last_db = 100
        n = []
        for n_cluster_solve in range(30):
            n_cluster_solve += 2
            spectral = SpectralClustering(n_clusters=n_cluster_solve,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
            result = spectral.fit_predict(data)
            db_score = metrics.davies_bouldin_score(X_tsne, result)
            n.append((n_cluster_solve, db_score))
            if db_score < last_db:
                last_db = db_score
                c = last_db
                
        self.c = c
        if return_results:
            return n
        return c
    
    def return_loss(self, y_pred, y_true):
        return (keras.losses.mean_squared_error(y_true, y_pred) + self.c)

filepath_autoencoder = 'entanglement_autoencoder_V2_3qb.h5'
filepath_model = 'entanglement_model_V2_3qb.h5'

preload_autoencoder = True
preload_model = True
train_autoencoder = False
train_model = True
test_model = True

print('*** DEEP EMBEDDED DETERMINATION ***')
print('Unsupervised clustering for an unknown number of clusters.')

print('*** LOADING FILES... ***')

file = 'wigner_functions_ea_3.npy'
X = []
Y = []

W = np.load(file)
for i in range(len(W)):
    data = np.array(W[i,:]).reshape((10, 10, 1))
    data_2d = np.zeros((12,12,1))
    data_2d[:10,:10] = data
    X.append(data_2d)

print('*** {} FILES LOADED ***'.format(len(X)))

X = np.array(X)
Y = np.array(Y)

print('### AUTOENCODER SETTINGS ###')
      
batch_size = 128
epochs = 30
inChannel = 1
x, y = 12, 12
input_img = Input(shape=(x, y, inChannel))

print('Batch size = {}, Epochs = {}, Input shape = {},{}'.format(batch_size,
      epochs, x, y))

if preload_autoencoder:
    print('*** LOADING AUTOENCODER... ***')
    autoencoder_model = Model(input_img, autoencoder(input_img))
    autoencoder_model.load_weights(filepath_autoencoder)
    autoencoder_model.compile(loss='mean_squared_error', optimizer=RMSprop())
    print('*** AUTOENCODER LOADED ***')
    autoencoder_model.summary()
else:
    print('*** BUILDING AUTOENCODER... ***')
    autoencoder_model = Model(input_img, autoencoder(input_img))
    autoencoder_model.compile(loss='mean_squared_error', optimizer=RMSprop())
    print('*** AUTOENCODER BUILT ***')
    autoencoder_model.summary()

if train_autoencoder:
    print('*** TRAINING AUTOENCODER... ***')
    autoencoder_train = autoencoder_model.fit(X, X, batch_size=batch_size,
                                              epochs=epochs, verbose=1,
                                              validation_split=0.2)
    print('*** AUTOENCODER TRAINED FOR {} EPOCHS ***'.format(epochs))
    
    print('*** SAVING AUTOENCODER WEIGHTS... ***')
    autoencoder_model.save_weights(filepath_autoencoder)
    print('*** AUTOENCODER WEIGHTS SAVED ***')

print('### MODEL SETTINGS ###')
      
batch_size = 128
epochs = 1
tries = 100
runs = 100
l = Loss(10)

print('Batch size = {}, Epochs = {}, Tries = {}'.format(batch_size,
      epochs, tries))

if preload_model:
    print('*** LOADING MODEL... ***')
    encoder_model = load_model(filepath_model, custom_objects={'return_loss':
                               l.return_loss})
    print('*** MODEL LOADED ***')
    encoder_model.summary()
    current_X = X.copy()
    current_Y = Y.copy()
    shuffle_in_unison(current_X, current_Y)
    results = encoder_model.predict(np.array(current_X[:200]))
    data = []
    for result in results:
        data.append(np.array(result).flatten())
    data = np.array(data)
    last_loss = l.loss_function(data)
else:
    print('*** BUILDING MODEL... ***')
    encoder_model = Model(input_img, encoder(input_img))
    encoder_model.load_weights(filepath_autoencoder, by_name=True)
    encoder_model.compile(loss=l.return_loss, optimizer=RMSprop())
    print('*** MODEL BUILT ***')
    encoder_model.summary()

if train_model:
    layer = 'encoder'
    enc = Model(inputs=autoencoder_model.input,
                outputs=autoencoder_model.get_layer(layer).output)
    Y = enc.predict(X)
    print('*** TRAINING MODEL... ***')
    if not preload_model:
        last_loss = 3
    count = 0
    for run in range(runs):
        current_X = X.copy()
        current_Y = Y.copy()
        shuffle_in_unison(current_X, current_Y)
        model_train = encoder_model.fit(current_X, current_Y,
                                        batch_size=batch_size,
                                        epochs=epochs,verbose=1,
                                        validation_split=0.1)
        results = encoder_model.predict(np.array(current_X[:200]))
        data = []
        for result in results:
            data.append(np.array(result).flatten())
        data = np.array(data)
        c = l.loss_function(data)
        if c < last_loss:
            count = 0
            print('New loss ', c, ' less than last loss ', last_loss)
            l = Loss(c)
            encoder_model.compile(loss=l.return_loss, optimizer=RMSprop())
            encoder_model.save(filepath_model)
            last_loss = c
            if c < 0.05:
                print('Model has converged to a very low loss.')
                print('*** MODEL TRAINED ***')
                break
        else:
            print('No improvement on last loss ', last_loss)
            encoder_model = load_model(filepath_model,
                                       custom_objects={'return_loss':
                                       l.return_loss})
            l = Loss(last_loss)
            encoder_model.compile(loss=l.return_loss, optimizer=RMSprop())
            count += 1
            if count == tries:
                print('Model is no longer converging. Current loss ',last_loss)
                break
            
if test_model:
    current_X = X.copy()
    shuffle(current_X)
    results = encoder_model.predict(np.array(current_X)[:1000])
    data = []
    for result in results:
        data.append(np.array(result).flatten())
    data = np.array(data)
    c = np.array(l.loss_function(data, True))
    plt.plot(c[:,0], c[:,1])
    plt.show()
