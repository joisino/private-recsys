import os
import sys
import numpy as np
import json
import pickle
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix, save_npz


if sys.argv[1] == 'hetrec':
    sr = {}
    si = {}
    R = np.zeros((1892, 17632))
    with open('./hetrec/user_artists.dat') as f:
        f.readline()
        for r in f:
            s = r.strip().split()
            if s[0] not in sr:
                sr[s[0]] = len(sr)
            if s[1] not in si:
                si[s[1]] = len(si)
            R[sr[s[0]], si[s[1]]] = float(s[2])

    K = 10
    while(1):
        print('A')
        prv = R.copy()
        mask = R > 0
        print('reveiwer:', (mask.sum(1) > 0).sum(), 'item:', (mask.sum(0) > 0).sum())
        R[:, mask.sum(0) < K] = 0
        mask = R > 0
        R[mask.sum(1) < K, :] = 0
        if (R == prv).all():
            break

    cR = R[R.sum(1) > 0]
    cR = cR[:, cR.sum(0) > 0]
    np.save('hetrec.npy', cR)

if sys.argv[1] == 'home':
    sr = {}
    si = {}
    R = np.zeros((66519, 28237))
    T = np.zeros((66519, 28237))
    with open('./reviews_Home_and_Kitchen_5.json') as f:
        for r in f:
            s = json.loads(r)
            if s['reviewerID'] not in sr:
                sr[s['reviewerID']] = len(sr)
            if s['asin'] not in si:
                si[s['asin']] = len(si)
            R[sr[s['reviewerID']], si[s['asin']]] = s['overall']
            T[sr[s['reviewerID']], si[s['asin']]] = s['unixReviewTime']

    K = 10
    while(1):
        print('A')
        prv = R.copy()
        mask = R > 0
        print('reveiwer:', (mask.sum(1) > 0).sum(), 'item:', (mask.sum(0) > 0).sum())
        R[:, mask.sum(0) < K] = 0
        mask = R > 0
        R[mask.sum(1) < K, :] = 0
        if (R == prv).all():
            break

    cR = R[R.sum(1) > 0]
    cR = cR[:, R.sum(0) > 0]

    np.save('Home_and_Kitchen.npy', cR)

    cT = T[R.sum(1) > 0]
    cT = cT[:, R.sum(0) > 0]

    history = [[] for i in range(cT.shape[0])]
    for i in range(cT.shape[0]):
        for j in np.nonzero(cT[i] > 0)[0]:
            history[i].append((cT[i, j], j))

    with open('Home_and_Kitchen_history.pickle', 'wb') as f:
        pickle.dump(history, f)

if sys.argv[1] == 'adult':
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DCFR'))
    from dcfr.datasets.adult import AdultDataset
    data = AdultDataset()

    c = list(data.train.columns)
    c.remove('sex')
    c.remove('result')
    X = np.concatenate([np.array(data.train[c]), np.array(data.test[c])])
    y = np.concatenate([np.array(data.train['result']), np.array(data.test['result'])])
    is_sensitive = np.concatenate([np.array(data.train['sex']), np.array(data.test['sex'])])
    np.save('adult_X.npy', X)
    np.save('adult_y.npy', y)
    np.save('adult_a.npy', is_sensitive)

    m, d = X.shape
    K = 10

    weight = 1 / np.log2(np.arange(K)[::-1] + 2)
    weight /= weight.sum()

    R = -distance_matrix(X, X)
    R -= 1e9 * np.eye(m)

    A = np.zeros((m, m))
    rank = np.argsort(R, 1)[:, -K:]
    A[np.arange(m).repeat(K), rank.reshape(-1)] += weight.repeat(m).reshape(K, m).T.reshape(-1)

    At = csr_matrix(A.T)

    np.save('adult_R.npy', R)
    save_npz('adult_At.npz', At)
    np.save('adult_rank.npy', rank)
