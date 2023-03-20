import os
import glob
import numpy as np

import sklearn.preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt


id2el = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'} #element to id mapping
el2id = {v: k for k, v in id2el.items()}    #id to element mapping

el = [1, 8]

NOFNODES = 20

NOFATOMS = {'H': 128, 'O': 64} #as a dict
#NOFATOMS_TEST = {'H': 4, 'O': 2} #as a dict

conv_energy = 5.5260279357571624        #check this value before implementing
mean_energy = -5.7452783952905939


def symmetryfunction(FUNCTION, SCALING):

    """
    Returns scaled symmetry function values

    Parameters:
    FUNCTION (str): path to function.cleaned.data
    SCALING (str): path to scaling.data

    Returns:
    scaledsf (dict): scaled symmetry function values
    el (np.ndarray): unique elements
    NOFSYM (list): number of symmetry functions for each element    
    """

    with open(FUNCTION) as f:
        lines = f.readlines()
        data = [line.split() for line in lines] #data is in memory now

    el = np.asarray(sorted(list({x[0] for x in data})), dtype = int)

    print(f' Number of unique elements: {len(el)} \n\n 1st element: {el[0]} --- {id2el[el[0]]} \n 2nd element: {el[1]} --- {id2el[el[1]]}\n\n')

    l1, l2 = [], []
    for i in data:
        if i[0] == str(el[0]):
            l1.append(i)
        if i[0] == str(el[1]):
            l2.append(i)

    l1 = np.asarray(l1, dtype = float)
    l2 = np.asarray(l2, dtype = float)

    NOFSYM = [l1.shape[1] - 1, l2.shape[1] - 1]
    print(f'Number of symmetry functions:\n\n for el:1 {NOFSYM[0]} \n for el:2 {NOFSYM[1]}\n\n')

    sf = {id2el[el[0]]: l1[:, 1:NOFSYM[0] + 1], id2el[el[1]]: l2[:, 1:NOFSYM[1] + 1]}
    
    scale = np.loadtxt(SCALING, usecols= (2, 3, 4, 5)) # Scaling of symmetry functions

    scalingmat = {id2el[el[0]]: scale[0:NOFSYM[0], :].T, id2el[el[1]]: scale[NOFSYM[0]:, :].T} #only for 2 elements have to generalize for more elements

    scaledsf = {id2el[i]: ((sf[id2el[i]] - scalingmat[id2el[i]][2]) / (scalingmat[id2el[i]][1] - scalingmat[id2el[i]][0])) for i in el} #scaled symmetry functions

    return scaledsf, el, NOFSYM

#To test the symmetryfunction function uncomment the following lines:

#scaledsf, el, NOFSYM = symmetryfunction(FUNCTION = '/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/testset/test-validationset/seed-1234567/nnp-train/function.cleaned.data', SCALING = '/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/testset/test-validationset/seed-1234567/scaling.data')
#print(f'elements: {el}\nNumber of symmetry functions: {NOFSYM}')

def weightmat(wlist, el, NOFSYM):

    """
    Returns weight matrices for each element
    
    Parameters:
    wlist (list): list of weight files
    el (np.ndarray): unique elements
    NOFSYM (list): number of symmetry functions for each element
    
    Returns:
    weights (dict): weight matrices for each element
    """
    
    weights = {}
    for i, nofsym, wlist in zip(el, NOFSYM, wlist):
        w = np.loadtxt(wlist, usecols = 0)

        w0 = w[:nofsym * NOFNODES].reshape(nofsym, NOFNODES) #can be much more generalized
        b0 = w[nofsym * NOFNODES:(nofsym * NOFNODES) + NOFNODES]
        w1 = w[(nofsym * NOFNODES) + NOFNODES: (nofsym * NOFNODES) + NOFNODES + (NOFNODES * NOFNODES)].reshape(NOFNODES, NOFNODES)
        b1 = w[(nofsym * NOFNODES) + NOFNODES + (NOFNODES * NOFNODES) : (nofsym * NOFNODES) + 2 * NOFNODES + (NOFNODES * NOFNODES)]
        w2 = w[(nofsym * NOFNODES) + 2 * NOFNODES + (NOFNODES * NOFNODES) : (nofsym * NOFNODES) + 2 * NOFNODES + (NOFNODES * NOFNODES) + NOFNODES]
        b2 = w[(nofsym * NOFNODES) + 2 * NOFNODES + (NOFNODES * NOFNODES) + NOFNODES : (nofsym * NOFNODES) + 2 * NOFNODES + (NOFNODES * NOFNODES) + NOFNODES + 1]

        weights[id2el[i]] = [w0, b0, w1, b1, w2, b2]

    return weights

#To test the weightmat function uncomment the following lines:
#weights = weightmat(wlist = sorted(glob.glob(os.path.join('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/seed-1234567/weight*'))), el = el, NOFSYM = NOFSYM)
#print(f"weights for element 1: {weights['H'][0].shape}")


def activation(x, act):
    if act == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif act == 'softplus':
        return np.log(1 + np.exp(x))
    elif act == 'ReLU':
        return np.maximum(0, x)
    elif act == 'tanh':
        return np.tanh(x)
    else:
        print('Invalid activation function')
        return None



def forwardpass(scaledsf, weightmat, el):
    
    """ performs the forward pass in a neural network

    Parameters:
    scaledsf (dict): scaled symmetry functions
    weightmat (dict): weight matrices for each element
    el (np.ndarray): unique elements

    Returns:
    featuresll (dict): features for each element
    energycontri (dict): energy contribution for each element
    """
    featuresll = {}
    #energycontri = {}

    for i in el:
        ll = []
        ec = []
        for j in range(scaledsf[id2el[i]].shape[0]):
            w0 = weightmat[id2el[i]][0]
            b0 = weightmat[id2el[i]][1]
            w1 = weightmat[id2el[i]][2]
            b1 = weightmat[id2el[i]][3]
            w2 = weightmat[id2el[i]][4]
            b2 = weightmat[id2el[i]][5]

            y1 = (w0.T @ scaledsf[id2el[i]][j]) + b0
            y1 = activation(y1, 'tanh')

            y2 = (w1.T @ y1) + b1
            y2 = activation(y2, 'tanh')

            ll.append(y2)
            y3 = (w2.T @ y2) + b2

            ec.append(y3)
        featuresll[id2el[i]] = np.array(ll).reshape(-1, NOFATOMS[id2el[i]] ,NOFNODES)
        featuresll[id2el[i]] = np.sum(featuresll[id2el[i]], axis = 1)
        print(featuresll[id2el[i]].shape)
        #energycontri[id2el[i]] = np.array(ec).reshape(-1, NOFATOMS[id2el[i]], 1)   #use when you want to calculate the energy contribution for each element

    features = np.concatenate([featuresll[id2el[i]] for i in el], axis = 1)

    return features


#features = forwardpass(scaledsf = scaledsf, weightmat = weights, el = el)

#print(features.shape)

#print(f"features for element 1: {featuresll['H'].shape}\nenergy contribution for element 1: {energycontri['H'].shape}")


##checking the forward pass by comparing the total energy

#eh = np.sum((energycontri['H']/conv_energy) + mean_energy, axis = 1)
#eo = np.sum((energycontri['O']/conv_energy) + mean_energy, axis = 1)

#totalenergy = eh + eo

#print(eh.shape, eo.shape)
#print(totalenergy.shape)
#print(totalenergy[:10])

#works perfectly fine



def feature_extractor(FUNCTION, SCALING):
    scaledsf, el, NOFSYM = symmetryfunction(FUNCTION, SCALING)
    weights = weightmat(wlist = sorted(glob.glob('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/trainset/seed-9812345/weight*')), el = el, NOFSYM = NOFSYM)
    features = forwardpass(scaledsf = scaledsf, weightmat = weights, el = el)
    return features


datasets = sorted(glob.glob(os.path.join('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/trainset/seed-9812345/trainset-80%-32/data-*')), key= lambda x: int(x.split('-')[-1]))

def committee():
    committee_model = {}
    for model, dataset in enumerate(datasets):
        os.chdir(dataset)
        features = feature_extractor(FUNCTION = 'function.cleaned.data', SCALING = 'scaling.data')
        true_energy = np.loadtxt('energy.data', usecols=0, dtype=float)
        committee_model['model-' + str(model)] = (features, true_energy)

        np.savetxt('dataset-' + str(model) + '.txt', np.concatenate((features, true_energy.reshape(-1, 1)), axis = 1), delimiter='\t', newline='\n', comments='# ', header=f'{2 * NOFNODES} features\tenergy', fmt = '%.8f')
    
    return committee_model

committee_model = committee()



def linear_regression(committee_model):
    lr_models = {}
    for key, values in enumerate(committee_model.values()):
        x = values[0]
        y = values[1]


        #x = sklearn.preprocessing.scale(values[0], axis = 1)
        #y = values[1]
        
        print(x.shape, y.shape)

        # print(f'{x.mean(axis = 0)}\t{x.std(axis = 0)}')
        # print(f'{x.mean(axis = 1)}\t{x.std(axis = 1)}')

        # print(f'{y.mean(axis = 0)}\t{y.std(axis = 0)}')

        lr = LinearRegression(fit_intercept= True).fit(x, y) #dic for LR for 1 model
        #lr = Ridge(alpha = 10000, fit_intercept= True, solver= 'auto').fit(x, y) #dic for LR for 1 model
        #lr = ARDRegression(fit_intercept= True, verbose= True).fit(x, y) #dic for LR for 1 model
        #lr = BayesianRidge(fit_intercept= True, verbose= True).fit(x, y) #dic for LR for 1 model

        lr_models['model-' + str(key)] = lr
        
    return lr_models

lr_models = linear_regression(committee_model)


#TESTING STAGE

feature_test = feature_extractor(FUNCTION = os.path.join('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/testset/test-validationset/NN7/function.cleaned.data'), SCALING = os.path.join('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/testset/test-validationset/NN7/scaling.data'))
energy_test = np.loadtxt(os.path.join('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/testset/test-validationset/NN7/energy.data'), usecols=0, dtype=float).reshape(-1, 1)
NOFFRAMES_TEST = feature_test.shape[0]


print(feature_test.shape)
print(energy_test.shape)


def multi_frame_prediction(lr_models, feature_test, energy_test):
    total_energy = {}
    for key, values in enumerate(lr_models.values()):  #model loop
        
        pred_energy = values.predict(feature_test)
        total_energy['model-' + str(key)] = pred_energy.reshape(-1, 1)
        
    return total_energy

total_energy = multi_frame_prediction(lr_models, feature_test, energy_test)

energy_mat = np.concatenate((list(total_energy.values())), axis = 1)
committee_mean = np.mean(energy_mat, axis = 1).reshape(-1, 1)
committee_std = np.std(energy_mat, axis = 1).reshape(-1, 1) / np.sqrt(energy_mat.shape[1] - 1)

n2p2_energy = np.loadtxt('/Users/shubhamdongriyal/my-drive/SabIA/HDNNP/H2O-Bing/MulRegression/testset/test-validationset/NN7/n2p2_energy_frames.out', usecols=0, dtype=float).reshape(-1, 1)
np.savetxt('/Users/shubhamdongriyal/Desktop/predicted_energy_trained-9812345.txt', np.concatenate((energy_test, np.concatenate((list(total_energy.values())), axis = 1)), axis =1), delimiter='\t', newline='\n', comments='# ', header=f'true_energy\tcommittee_energy_predictions', fmt = '%.8f')
np.savetxt('/Users/shubhamdongriyal/Desktop/committee_mean_std_trained-9812345.txt', np.concatenate((energy_test, n2p2_energy, committee_mean, committee_std), axis = 1), delimiter='\t', newline='\n', comments='# ', header=f'true_energy\tn2p2_energy\tcommittee_mean\tcommittee_std', fmt = '%.8f')
