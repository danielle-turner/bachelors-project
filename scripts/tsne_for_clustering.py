# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:46:43 2019

@author: User
"""

from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
#import tsne_V2 as tsne

bell_11 = np.load('wigner_bell_state_11_all_10split.npy').flatten()
bell_01 = np.load('wigner_bell_state_01_all_10split.npy').flatten()
bell_10 = np.load('wigner_bell_state_10_all_10split.npy').flatten()
bell_00 = np.load('wigner_bell_state_00_all_10split.npy').flatten()
bell_11.shape = (-1, 1)
bell_01.shape = (-1, 1)
bell_10.shape = (-1, 1)
bell_00.shape = (-1, 1)

spin_uu = np.load('wigner_spin_state_uu_all_10split.npy').flatten()
spin_ud = np.load('wigner_spin_state_ud_all_10split.npy').flatten()
spin_du = np.load('wigner_spin_state_du_all_10split.npy').flatten()
spin_dd = np.load('wigner_spin_state_dd_all_10split.npy').flatten()

spin_uu.shape = (-1, 1)
spin_ud.shape = (-1, 1)
spin_du.shape = (-1, 1)
spin_dd.shape = (-1, 1)

X = np.concatenate((bell_11, bell_01, bell_10, bell_00,
                    spin_uu, spin_ud, spin_du, spin_dd), axis=1).T

#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=5.0, learning_rate=50.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p5_lr50.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=30.0, learning_rate=50.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p30_lr50.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=50.0, learning_rate=50.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p50_lr50.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=100.0, learning_rate=50.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p100_lr50.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=5.0, learning_rate=200.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p5_lr200.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=30.0, learning_rate=200.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p30_lr200.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=50.0, learning_rate=200.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p50_lr200.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=100.0, learning_rate=200.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p100_lr200.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=5.0, learning_rate=500.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p5_lr500.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=30.0, learning_rate=500.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p30_lr500.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=50.0, learning_rate=500.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p50_lr500.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=100.0, learning_rate=500.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p100_lr500.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=5.0, learning_rate=1000.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p5_lr1000.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=30.0, learning_rate=1000.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p30_lr1000.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=50.0, learning_rate=1000.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p50_lr1000.png')
#plt.clf()
#
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,
#                     perplexity=100.0, learning_rate=1000.0)
#X_tsne = np.abs(tsne.fit_transform(X))
#plt.scatter(X_tsne[:4,0], X_tsne[:4,1], c='blue')
#plt.scatter(X_tsne[4:,0], X_tsne[4:,1], c='red')
#plt.show()
#plt.savefig('cluster_p100_lr1000.png')
#plt.clf()