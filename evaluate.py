import os
import numpy as np
import pickle
import argparse

from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix
from scipy.stats import entropy


np.random.seed(0)


def recall(li, gt):
    if gt in li:
        return 1
    return 0


def nDCG(li, gt):
    if gt in li:
        return 1 / np.log2(li.tolist().index(gt) + 2)
    return 0


def list_minimum_group(li, is_sensitive):
    return np.bincount(is_sensitive[li], minlength=2).min()


def list_entropy(li, is_sensitive):
    a = np.bincount(is_sensitive[li], minlength=2)
    return entropy(a / a.sum(), base=2)


def select_list(score, is_sensitive, used, K, b):
    assert(b * 2 <= K)
    score = score.copy()
    score[used] -= score.max() + 1
    li = []
    cnt = [0, 0]
    for x in score.argsort()[::-1]:
        cur_sensitive = int(is_sensitive[x])
        if cnt[1 - cur_sensitive] + K - len(li) <= b:
            continue
        cnt[cur_sensitive] += 1
        li.append(x)
        if len(li) == K:
            break
    return np.array(li)


parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['100k', '1m', 'home', 'hetrec'], default='100k')
parser.add_argument('--prov', choices=['cosine', 'bpr'], default='cosine')
parser.add_argument('--sensitive', choices=['popularity', 'old'], default='popularity', help='`old` is valid only for MovieLens')
parser.add_argument('--split', type=int, default=1, help='Total number of parallel execusion (only for parallel execusion, set 1 othePWise)')
parser.add_argument('--block', type=int, default=0, help='Id of the current execusion (only for parallel excecusion, set 0 othePWise)')
args = parser.parse_args()
assert(args.sensitive == 'popularity' or args.data in ['100k', '1m'])
assert(0 <= args.block and args.block < args.split)

#
# Load Data
#

if args.data == '100k':
    n = 943
    m = 1682
    filename = 'ml-100k/u.data'
    delimiter = '\t'
elif args.data == '1m':
    n = 6040
    m = 3952
    filename = 'ml-1m/ratings.dat'
    delimiter = '::'

K = 10

if args.data == '100k' or args.data == '1m':
    raw_R = np.zeros((n, m))
    history = [[] for i in range(n)]
    with open(filename) as f:
        for r in f:
            user, movie, r, t = map(int, r.split(delimiter))
            user -= 1
            movie -= 1
            raw_R[user, movie] = r
            history[user].append((t, movie))
elif args.data == 'hetrec':
    raw_R = np.log2(np.load('hetrec.npy') + 1)
    n, m = raw_R.shape
    history = [[] for i in range(n)]
    for i in range(n):
        for j in np.nonzero(raw_R[i] > 0)[0]:
            history[i].append((np.random.rand(), j))
elif args.data == 'home':
    raw_R = np.load('Home_and_Kitchen.npy')
    n, m = raw_R.shape
    with open('Home_and_Kitchen_history.pickle', 'br') as f:
        history = pickle.load(f)

if args.sensitive == 'popularity':
    mask = raw_R > 0
    if args.data == '100k':
        is_sensitive = mask.sum(0) < 50
    elif args.data == '1m':
        is_sensitive = mask.sum(0) < 300
    elif args.data == 'hetrec':
        is_sensitive = mask.sum(0) < 50
    elif args.data == 'home':
        is_sensitive = mask.sum(0) < 50
elif args.sensitive == 'old':
    is_sensitive = np.zeros(m, dtype='bool')
    if args.data == '100k':
        filename = 'ml-100k/u.item'
        delimiter = '|'
    elif args.data == '1m':
        filename = 'ml-1m/movies.dat'
        delimiter = '::'
    with open(filename, encoding='utf8', errors='ignore') as f:
        for r in f:
            li = r.strip().split(delimiter)
            if '(19' in li[1]:
                year = 1900 + int(li[1].split('(19')[1].split(')')[0])
            elif '(20' in li[1]:
                year = 2000 + int(li[1].split('(20')[1].split(')')[0])
            is_sensitive[int(li[0]) - 1] = year < 1990


#
# Data Loaded
#

damping_factor = 0.01
bs = [0, 1, 2, 3, 4, 5]  # minimum requirements

provider_recall = 0
provider_nDCG = 0
oracle_recall = [0 for j in bs]
oracle_nDCG = [0 for j in bs]
PR_recall = [0 for j in bs]
PR_nDCG = [0 for j in bs]
PW_recall = [0 for j in bs]
PW_nDCG = [0 for j in bs]

random_recall = [0 for j in bs]
random_nDCG = [0 for j in bs]


provider_minimum = 0
provider_entropy = 0
oracle_minimum = [0 for j in bs]
oracle_entropy = [0 for j in bs]
PR_minimum = [0 for j in bs]
PR_entropy = [0 for j in bs]
PW_minimum = [0 for j in bs]
PW_entropy = [0 for j in bs]

random_minimum = [0 for j in bs]
random_entropy = [0 for j in bs]

start_index = int(n * args.block / args.split)
end_index = int(n * (args.block + 1) / args.split)

for i in range(start_index, end_index):
    gt = sorted(history[i])[-1][1]
    source = sorted(history[i])[-2][1]
    used = [y for x, y in history[i] if y != gt]

    R = raw_R.copy()
    R[i, gt] = 0

    mask = R > 0

    if args.prov == 'bpr':
        model = BayesianPersonalizedRanking(num_threads=1, random_state=0)
    elif args.prov == 'cosine':
        model = CosineRecommender()

    sR = csr_matrix(mask.T)
    model.fit(sR, show_progress=False)
    if args.prov == 'bpr':
        score = model.item_factors @ model.item_factors.T
    else:
        score = np.zeros((m, m))
        for item in range(m):
            for j, v in model.similar_items(item, m):
                score[item, j] = v

    score_remove = score.copy()
    score_remove[:, used] -= 1e9
    score_remove -= np.eye(m) * 1e9

    list_provider = np.argsort(score_remove[source])[::-1][:K]

    # Service provider's recsys
    provider_recall += recall(list_provider, gt)
    provider_nDCG += nDCG(list_provider, gt)
    provider_minimum += list_minimum_group(list_provider, is_sensitive)
    provider_entropy += list_entropy(list_provider, is_sensitive)

    # Oracle recsys
    for k, b in enumerate(bs):
        oracle_list = []
        cnt = [0, 0]
        for j in np.argsort(score_remove[source])[::-1]:
            if j not in used + oracle_list and cnt[1 - int(is_sensitive[j])] + K - len(oracle_list) > b:
                oracle_list.append(j)
                cnt[int(is_sensitive[j])] += 1
            if len(oracle_list) == K:
                break
        oracle_list = np.array(oracle_list)
        oracle_recall[k] += recall(oracle_list, gt)
        oracle_nDCG[k] += nDCG(oracle_list, gt)
        oracle_minimum[k] += list_minimum_group(oracle_list, is_sensitive)
        oracle_entropy[k] += list_entropy(oracle_list, is_sensitive)

    # Construct the recsys graph
    A = np.zeros((m, m))
    rank = np.argsort(score_remove, 1)[:, -K:]
    weight = 1 / np.log2(np.arange(K)[::-1] + 2)
    weight /= weight.sum()
    A[np.arange(m).repeat(K), rank.reshape(-1)] += weight.repeat(m).reshape(K, m).T.reshape(-1)

    # PrivateRank
    psc = np.zeros(m)
    cur = np.zeros(m)
    cur[source] = 1
    for _ in range(11):
        psc += (1 - damping_factor) * cur
        cur = damping_factor * A.T @ cur
    for k, b in enumerate(bs):
        PR_list = select_list(psc, is_sensitive, used, K, b)
        PR_recall[k] += recall(PR_list, gt)
        PR_nDCG[k] += nDCG(PR_list, gt)
        PR_minimum[k] += list_minimum_group(PR_list, is_sensitive)
        PR_entropy[k] += list_entropy(PR_list, is_sensitive)

    # PrivateWalk
    for k, b in enumerate(bs):
        PW_list = []
        cnt = [0, 0]
        it = 0
        for lth in range(K):
            cur = source
            max_length = 100
            for _ in range(max_length):
                it += 1
                cur = np.random.choice(rank[cur], p=weight)
                cur_sensitive = int(is_sensitive[cur])
                if cur not in used + PW_list and cnt[1 - cur_sensitive] + K - lth > b:
                    break
            while cur in used + PW_list or cnt[1 - cur_sensitive] + K - lth <= b:
                cur = np.random.randint(m)
                cur_sensitive = int(is_sensitive[cur])
            PW_list.append(cur)
            cnt[cur_sensitive] += 1
        PW_list = np.array(PW_list)
        PW_recall[k] += recall(PW_list, gt)
        PW_nDCG[k] += nDCG(PW_list, gt)
        PW_minimum[k] += list_minimum_group(PW_list, is_sensitive)
        PW_entropy[k] += list_entropy(PW_list, is_sensitive)

    # Random
    random_score = np.random.rand(m)
    for k, b in enumerate(bs):
        random_list = select_list(random_score, is_sensitive, used, K, b)
        random_recall[k] += recall(random_list, gt)
        random_nDCG[k] += nDCG(random_list, gt)
        random_minimum[k] += list_minimum_group(random_list, is_sensitive)
        random_entropy[k] += list_entropy(random_list, is_sensitive)

    def print_single(method, li):
        print(method + ' |', end='')
        for k, b in enumerate(bs):
            print(' b={}: {:.2f} |'.format(b, li[k]), end='')
        print()

    print('#')
    print('# User {} - {}'.format(start_index, i))
    print('#')
    print('-' * 30)
    print('provider recall    | {:.2f}'.format(provider_recall))
    print_single('oracle recall     ', oracle_recall)
    print_single('PrivateRank recall', PR_recall)
    print_single('PrivateWalk recall', PW_recall)
    print_single('random recall     ', random_recall)
    print('-' * 30)
    print('provider nDCG    | {:.2f}'.format(provider_nDCG))
    print_single('oracle nDCG     ', oracle_nDCG)
    print_single('PrivateRank nDCG', PR_nDCG)
    print_single('PrivateWalk nDCG', PW_nDCG)
    print_single('random nDCG     ', random_nDCG)
    print('-' * 30)
    print('provider least count    | {:.2f}'.format(provider_minimum))
    print_single('oracle least count     ', oracle_minimum)
    print_single('PrivateRank least count', PR_minimum)
    print_single('PrivateWalk least count', PW_minimum)
    print_single('random least count     ', random_minimum)
    print('-' * 30)
    print('provider entropy    | {:.2f}'.format(provider_entropy))
    print_single('oracle entropy     ', oracle_entropy)
    print_single('PrivateRank entropy', PR_entropy)
    print_single('PrivateWalk entropy', PW_entropy)
    print_single('random entropy     ', random_entropy)
    print('-' * 30)

if not os.path.exists('out'):
    os.mkdir('out')

with open('out/config.txt', 'w') as f:
    print(len(bs), file=f)

with open('out/{}.txt'.format(args.block), 'w') as f:
    print(provider_recall, file=f)
    print(provider_nDCG, file=f)
    print(provider_minimum, file=f)
    print(provider_entropy, file=f)

    for k, p in enumerate(bs):
        print(oracle_recall[k], file=f)
        print(oracle_nDCG[k], file=f)
        print(oracle_minimum[k], file=f)
        print(oracle_entropy[k], file=f)

    for k, b in enumerate(bs):
        print(PR_recall[k], file=f)
        print(PR_nDCG[k], file=f)
        print(PR_minimum[k], file=f)
        print(PR_entropy[k], file=f)

    for k, p in enumerate(bs):
        print(PW_recall[k], file=f)
        print(PW_nDCG[k], file=f)
        print(PW_minimum[k], file=f)
        print(PW_entropy[k], file=f)

    for k, p in enumerate(bs):
        print(random_recall[k], file=f)
        print(random_nDCG[k], file=f)
        print(random_minimum[k], file=f)
        print(random_entropy[k], file=f)
