import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import load_model
import glob
from sklearn import manifold
import matplotlib.pyplot as plt

"""
Much of the code below, including but not limited to the definitions for
encoder(input_data), decoder(conv3), and autoencoder(input_data), is adapted
from https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
"""

def encoder(input_data):
    #encoder
    #input = 20 x 20 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_data)
    #21 x 21 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   name='encoder')(pool2)
    #7 x 7 x 128 (small and thick)
    return conv3

def decoder(conv3):
    #decoder
    #input = 7 x 7 x 128 (small and thick)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    #7 x 7 x 128
    up1 = UpSampling2D((2, 2))(conv4)
    # 14 x 14 x 128
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    # 14 x 14 x 64
    up2 = UpSampling2D((2, 2))(conv5)
    # 21 x 21 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    # 21 x 21 x 1
    return decoded

def autoencoder(input_data):
    # Encoder
    conv = encoder(input_data)
    # Decoder
    decoded = decoder(conv)
    return decoded

train = False

batch_size = 128
epochs = 50
inChannel = 1
x, y = 20, 20
input_img = Input(shape = (x, y, inChannel))

print('*** DEEP EMBEDDED DETERMINATION ***')
print('Unsupervised clustering for an unknown number of clusters.')

print('### SETTINGS ###')
print('Batch size = {}, Epochs = {}, Input shape = {},{}'.format(batch_size,
      epochs, x, y))

if train:
    print('*** BUILDING MODEL... ***')
    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
    print('*** MODEL BUILT ***')
    
    autoencoder.summary()

print('*** LOADING FILES... ***')
files = glob.glob('PureWigs21\\PureWigs21\\*.dat')

X = []
Y = []

for file in files:
    W = np.genfromtxt(file, dtype=None, delimiter=',')
    for x in range(len(W)):
        data_2d = np.array(W[x,:]).reshape((21, 21, 1))[:20, :20, :]
        X.append(data_2d)
        if 'Sep' in file:
            Y.append(1)
        else:
            Y.append(0)
            
from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(X, X, 
                                                             test_size=0.2, 
                                                             random_state=13)
train_X = np.array(train_X)
valid_X = np.array(valid_X)
train_ground = np.array(train_ground)
valid_ground = np.array(valid_ground)

print('*** {} FILES LOADED ***'.format(len(X)))

print('*** FITTING MODEL ***')
if train:
    autoencoder_train = autoencoder.fit(train_X, train_ground,
                                        batch_size=batch_size, epochs=epochs,
                                        verbose=1,
                                        validation_data=(valid_X,
                                                         valid_ground))
    autoencoder.save('deep_embedded_determination_model.h5')
else:
    autoencoder = load_model('deep_embedded_determination_model.h5')
print('*** MODEL FIT ***')

layer_name = 'encoder'
encoder_model = Model(inputs=autoencoder.input,
                      outputs=autoencoder.get_layer(layer_name).output)
encoded = encoder_model.predict(np.array(X))

encoded = np.array(encoded)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
                     perplexity=30.0, learning_rate=200.0)
X_tsne = np.abs(tsne.fit_transform(encoded))
plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
plt.show()
plt.savefig('deep_embedded_determination_cluster_p30_lr200.png')
plt.clf()
