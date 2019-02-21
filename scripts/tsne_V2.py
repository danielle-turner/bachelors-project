# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:16:51 2018

@author: Danielle Turner, B513740
"""

import numpy as np

def hyperspherical_to_cartesian(datapoint):
    #
    # Converts hyperspherical coordinates to cartesian coordinates.
    #
    # Inputs :
    #       datapoint : The spherical coordinates of the datapoint to be
    #                   transformed into cartesian coordinates.
    #
    coordinates = []
    for i in range(len(datapoint)):
        xi = np.cos(datapoint[i])
        j = i
        while j > 0:
            j -= 1
            xi *= np.sin(datapoint[j])
        coordinates.append(xi)
    xi = np.sin(datapoint[i])
    j = i
    while j > 0:
        j -= 1
        xi *= np.sin(datapoint[j])
    coordinates.append(xi)
    coordinates = np.array(coordinates)
    return coordinates

def hypersphere_arclength(datapoint1, datapoint2):
    #
    # Determines the arclength between two points on a hypersphere.
    #
    # Inputs :
    #       datapoint1 : The hyperspherical coordinates of a datapoint.
    #       datapoint2 : The hyperspherical coordinates of another datapoint.
    #
    coordinates1 = hyperspherical_to_cartesian(datapoint1)
    coordinates2 = hyperspherical_to_cartesian(datapoint2)
    angle = np.dot(coordinates1, coordinates2)
    arclength = np.arccos(angle)
    return arclength

def conditional_similarity(datapoints, i, j, n, sigma_sq):
    #
    # Returns the conditional similarity between datapoints i and j, as given
    # in 'Visualising Data using t-SNE' by Laurens van der Maaten and
    # Geoffrey Hinton in Journal of Machine Learning Research 9 (2008)
    # 2579-2605 to be equal to:
    #
    # P_ij = exp(-(d**2) / (2*sigma_sq)) / sum(exp(-(d**2) / (2*sigma_sq)))
    #
    # where d is the euclidean distance between points i and j, and sigma_sq
    # is the variance about point i - determined using the function 
    # get_similarity_matrix.
    #
    # Input:
    #       datapoints : numpy ndarray for all datapoints X=(x1, x2, ..., xn)
    #                    in the n-dimensional data space Rn.
    #       i :          integer for the index of datapoint1.
    #       j :          integer for the index of datapoint2.
    #       n :          integer for the number of datapoints in the
    #                    n-dimensional data space Rn.
    #       sigma_sq :   numpy ndarray for all current variances.
    #
    sigma_sq_i = sigma_sq[i]
    distance = hypersphere_arclength(datapoints[i], datapoints[j])
    numerator = np.exp(-(distance**2) / (2*sigma_sq_i))
    denominator = 0
    for k in range(n):
        if not k == i:
            distance = hypersphere_arclength(datapoints[i], datapoints[k])
            denominator += np.exp(-(distance**2) / (2*sigma_sq_i))
    return numerator / denominator

def symm_conditional_similarity(datapoints, i, j, n, sigma_sq):
    #
    # Gives the symmetrical conditional similarity, given in 'Visualising Data
    # using t-SNE' by Laurens van der Maaten and Geoffrey Hinton in Journal of
    # Machine Learning Research 9 (2008) 2579-2605 to be equal to:
    #
    # Pij = (Pi_j + Pj_i) / (2*n)
    #
    # where Pi_j and Pj_i are the conditional similarities for (i,j) and (j,i)
    # respectively, and n is the total number of datapoints.
    #
    # Input:
    #       datapoints : numpy ndarray for all datapoints X=(x1, x2, ..., xn)
    #                    in the n-dimensional data space Rn.
    #       i :          integer for the index of datapoint1.
    #       j :          integer for the index of datapoint2.
    #       n :          integer for the number of datapoints in the
    #                    n-dimensional data space Rn.
    #       sigma_sq :   numpy ndarray for all current variances.
    #
    Pi_j = conditional_similarity(datapoints, j, i, n, sigma_sq)
    Pj_i = conditional_similarity(datapoints, i, j, n, sigma_sq)
    return (Pi_j + Pj_i) / (2*n)

def shannon_entropy(D, sigma_sq):
    #
    # Returns the Shannon Entropy of datapoints i and j as given in
    # 'Visualising Data using t-SNE' by Laurens van der Maaten and
    # Geoffrey Hinton in Journal of Machine Learning Research 9 (2008)
    # 2579-2605 to be equal to:
    #
    # H = sum(Pj_i * np.log2(Pj_i))
    #
    # where Pj_i is the conditional similarity between points j and i.
    #
    # Input:
    #       D :          numpy ndarray for all datapoints X=(x1, x2, ..., xn)
    #                    in the n-dimensional data space Rn.
    #       sigma_sq :   numpy ndarray for all current variances.
    #
    P = np.exp(-D.copy() * sigma_sq)
    sumP = sum(P)
    H = np.log(sumP) + sigma_sq * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def get_similarity_matrix(datapoints, perplexity, tolerance, tries):
    #
    # Performs a binary search to obtain the similarity matrix P such that
    # the conditional Gaussian at each datapoint has the same perplexity.
    #
    # Input:
    #       datapoints : numpy ndarray for all datapoints X=(x1, x2, ..., xn)
    #                    in the n-dimensional data space Rn.
    #       perplexity : float value for the starting perplexity for the
    #                    binary search, defined as 2**(H(Pi)), where H(Pi)
    #                    is the Shannon Entropy of the probability distribution
    #                    Pi, defined as -sum(Pj_i*log2(Pj_i)) for all j.
    #       tolerance :  float value for the allowed absolute deviation from
    #                    the target value.
    #       tries :      integer value for the number of times the precision
    #                    should be altered if not within the given tolerance.
    #
    '''
        Type checks - all variables revert to default in the case of an
        incorrect input. In the case of incorrect input for the ndarray
        datapoints, a ValueError will be raised.
    '''
    try:
        datapoints = np.array(datapoints, dtype='float')
    except ValueError as error:
        raise error
        
    try:
        perplexity = float(perplexity)
    except ValueError as error:
        print('ValueError: ' + str(error))
        print('Default perplexity=30.0 used.')
        perplexity = 30.0
        
    try:
        tolerance = float(tolerance)
    except ValueError as error:
        print('ValueError: ' + str(error))
        print('Default tolerance=1e-5 used.')
        tolerance = 1e-5
    
    try:
        tries = int(tries)
    except ValueError as error:
        print('ValueError: ' + str(error))
        print('Default tries=50 used.')
        tries = 50

    (n, d) = datapoints.shape
    sum_dp = np.sum(np.square(datapoints), 1)
    D = np.add(np.add(-2 * np.dot(datapoints, datapoints.T), sum_dp).T, sum_dp)
    P = np.zeros((n,n))
    sigma_sq = np.ones((n, 1))
    log_perp = np.log(perplexity)
    
    for i in range(n):
        print(i)
        sigma_min = -np.inf
        sigma_max = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = shannon_entropy(Di, sigma_sq[i])
        
        difference = H - log_perp
        no_tries = 0
        while np.abs(difference) > tolerance and no_tries < tries:
            
            if difference > 0:
                sigma_min = sigma_sq[i].copy()
                if sigma_max == np.inf or sigma_max == -np.inf:
                    sigma_sq[i] = sigma_sq[i] * 2.
                else:
                    sigma_sq[i] = (sigma_sq[i] + sigma_max) / 2.
            else:
                sigma_max = sigma_sq[i].copy()
                if sigma_min == -np.inf or sigma_min == np.inf:
                    sigma_sq[i] = sigma_sq[i] / 2.
                else:
                    sigma_sq[i] = (sigma_sq[i] + sigma_min) / 2.
            
            H, thisP = shannon_entropy(Di, sigma_sq[i])
            difference = H - log_perp
            no_tries += 1
            
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
        
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / sigma_sq)))
    return P, sigma_sq

def get_map_similarity_matrix(mappoints):
    #
    # Returns the similarity matrix of the mappoints, given in
    # 'Visualising Data using t-SNE' by Laurens van der Maaten and
    # Geoffrey Hinton in Journal of Machine Learning Research 9 (2008)
    # 2579-2605 to be equal to:
    #
    # Q_ij = (1 + d**2)**(-1) / sum((1 + d**2)**(-1))
    #
    # where d is the euclidean distance between points i and j, and the sum
    # is taken across all values of k that are not equal to i.
    #
    # Input:
    #       mappoints : numpy ndarray for all mappoints M=(m1, m2, ..., mn)
    #                   in the y-dimensional map space Ry.
    #
    (n, d) = mappoints.shape
    sum_mp = np.sum(np.square(mappoints), 1)
    num = -2. * np.dot(mappoints, mappoints.T)
    num = 1. / (1. + np.add(np.add(num, sum_mp).T, sum_mp))
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    
    return Q, num

def kullback_leibler_divergence(P, Q):
    #
    # Returns the Kullback Leibler Divergence, given in 'Visualising Data using
    # t-SNE' by Laurens van der Maaten and Geoffrey Hinton in Journal of 
    # Machine Learning Research 9 (2008) 2579-2605 to be equal to:
    #
    # KL = sum(P_ij * log(P_ij / Q_ij))
    #
    # Where P_ij and Q_ij are the similarity matrices of the datapoints and the
    # mappoints respectively, and the sum is taken across all values of i and
    # j.
    #
    # Input:
    #       P : numpy ndarray for the similarity matrix of the datapoints.
    #       Q : numpy ndarray for the similarity matrix of the mappoints.
    #
    (n, d) = P.shape
    KL = np.sum(P * np.log(P / Q))
    
    return KL

def kullback_leibler_gradient(P, Q, mappoints, num):
    #
    # Returns the gradient of the Kullback Leibler Divergence, given in
    # 'Visualising Data using t-SNE' by Laurens van der Maaten and
    # Geoffrey Hinton in Journal of Machine Learning Research 9 (2008)
    # 2579-2605 to be equal to:
    #
    # grad = 4 * sum((P_ij - Q_ij) * (d) * (1 + d**2)**(-1))
    #
    # where P and Q are the similarity matrices of the datapoints and mappoints
    # respectively, d is the euclidean distance between mappoints i and j,
    # and the sum is taken across all values of j.
    #
    # Input:
    #       P :         numpy ndarray for the similarity matrix of the
    #                   datapoints.
    #       Q :         numpy ndarray for the similarity matrix of the
    #                   mappoints.
    #       mappoints : numpy ndarray for all mappoints Y=(y1, y2, ..., yn)
    #                    in the n-dimensional map space Ry.
    #       num :       numpy ndarray from output of get_map_similarity_matrix.
    #
    (n, d) = mappoints.shape
    KL_grad = np.zeros((n,d))
    PQ = P - Q
#    for i in range(n):
#        for j in range(n):
#            distance = euclidean_distance(mappoints[i], mappoints[j])
#            KL_grad[i,j] = (P[i,j] - Q[i,j])*(distance / (1 + (distance**2)))
    for i in range(n):
        KL_grad[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (d, 1)).T * (mappoints[i, :] - mappoints), 0)
    return KL_grad

def tsne(datapoints, perplexity, tolerance, tries, dimensions, learn_rate,
         max_iters, gamma):
    #
    # Implementation of the t-SNE algorithm, as described in 'Visualizing Data
    # using t-SNE' by Laurens van der Maaten and Geoffrey Hinton in Journal of
    # Machine Learning Research 9 (2008) 2579-2605.
    #
    # Input:
    #       datapoints : numpy ndarray for all datapoints X=(x1, x2, ..., xn)
    #                    in the n-dimensional data space Rn.
    #       perplexity : float value for the starting perplexity for the
    #                    binary search, defined as 2**(H(Pi)), where H(Pi)
    #                    is the Shannon Entropy of the probability distribution
    #                    Pi.
    #       tolerance :  float value for the allowed absolute deviation from
    #                    the target value.
    #       tries :      integer value for the number of times the precision
    #                    should be altered if not within the given tolerance.
    #       dimensions : integer value for the desired number of dimensions,
    #                    restricted to 2 or 3.
    #       learn_rate : integer value for the learning rate of the algorithm.
    #       max_iters :  integer value for the maximum number of iterations.
    #       gamma :      float value for the rate of change of the learning
    #                    rate, which should be equal to zero for a constant
    #                    learning rate. NOT CURRENTLY IN USE.
    #
    (n, d) = datapoints.shape
    P, sigma_sq = get_similarity_matrix(datapoints, perplexity, tolerance,
                                        tries)
    print(P)
    P = P + P.T
    P = 4. * P / np.sum(P)
    P = np.maximum(P, 1e-12)
    np.random.seed(0)
    q = np.random.randn(n, dimensions)
    grad = np.zeros((n, dimensions))
    update = np.zeros((n, dimensions))
    gains = np.ones((n, dimensions))
    Q, num = get_map_similarity_matrix(q)
    
    best_error = np.inf
    best_iteration = 0
    best_q = 0
    momentum = 0.5
    for i in range(max_iters):
        error = kullback_leibler_divergence(P, Q)
        grad = kullback_leibler_gradient(P, Q, q, num)
        print(grad, update)
        gains = (gains + 0.2) * ((grad > 0.) != (update > 0.)) + \
                (gains * 0.8) * ((grad > 0.) == (update > 0.))
        gains[gains < 0.01] = 0.01
        grad *= gains
        
        update = momentum * update - learn_rate * grad
        q += update
        Q, num = get_map_similarity_matrix(q)
        error = kullback_leibler_divergence(P, Q)
        if error < best_error:
            best_error = error
            best_iteration = i
            best_q = q

        if i == 20:
            momentum = 0.8
        if i == 100:
            P = P / 4.
    return best_q, best_error, best_iteration
