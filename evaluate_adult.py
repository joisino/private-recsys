import numpy as np
import argparse

from scipy.stats import entropy
from scipy.sparse import load_npz


np.random.seed(0)


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
parser.add_argument('--split', type=int, default=1, help='Total number of parallel execusion (only for parallel execusion, set 1 otherwise)')
parser.add_argument('--block', type=int, default=0, help='Id of the current execusion (only for parallel excecusion, set 0 otherwise)')
args = parser.parse_args()
assert(0 <= args.block and args.block < args.split)

#
# Load Data
#

X = np.load('adult_X.npy')
y = np.load('adult_y.npy')
is_sensitive = np.load('adult_a.npy')

m, d = X.shape
K = 10

weight = 1 / np.log2(np.arange(K)[::-1] + 2)
weight /= weight.sum()

R = np.load('adult_R.npy')
At = load_npz('adult_At.npz')
rank = np.load('adult_rank.npy')

#
# Data Loaded
#

damping_factor = 0.01
bs = [0, 1, 2, 3, 4, 5]

provider_accuracy = 0
oracle_accuracy = [0 for j in bs]
PR_accuracy = [0 for j in bs]
PW_accuracy = [0 for j in bs]
random_accuracy = [0 for j in bs]

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

start_index = int(m * args.block / args.split)
end_index = int(m * (args.block + 1) / args.split)

for i in range(start_index, end_index):
    source = i
    used = [source]

    # Service provider's recsys
    list_provider = rank[source]
    provider_accuracy += (y[list_provider] == y[source]).sum()
    provider_minimum += list_minimum_group(list_provider, is_sensitive)
    provider_entropy += list_entropy(list_provider, is_sensitive)

    # Oracle recsys
    for k, b in enumerate(bs):
        oracle_list = []
        cnt = [0, 0]
        for j in np.argsort(R[source])[::-1]:
            if j not in used + oracle_list and cnt[1 - int(is_sensitive[j])] + K - len(oracle_list) > b:
                oracle_list.append(j)
                cnt[int(is_sensitive[j])] += 1
            if len(oracle_list) == K:
                break
        oracle_list = np.array(oracle_list)
        oracle_accuracy[k] += (y[oracle_list] == y[source]).sum()
        oracle_minimum[k] += list_minimum_group(oracle_list, is_sensitive)
        oracle_entropy[k] += list_entropy(oracle_list, is_sensitive)

    # PrivateRank
    psc = np.zeros(m)
    cur = np.zeros(m)
    cur[source] = 1
    for _ in range(11):
        psc += (1 - damping_factor) * cur
        cur = damping_factor * At @ cur
    for k, b in enumerate(bs):
        PR_list = select_list(psc, is_sensitive, used, K, b)
        PR_accuracy[k] += (y[PR_list] == y[source]).sum()
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
        PW_accuracy[k] += (y[PW_list] == y[source]).sum()
        PW_minimum[k] += list_minimum_group(PW_list, is_sensitive)
        PW_entropy[k] += list_entropy(PW_list, is_sensitive)

    # Random
    random_score = np.random.rand(m)
    for k, b in enumerate(bs):
        random_list = select_list(random_score, is_sensitive, used, K, b)
        random_accuracy[k] += (y[random_list] == y[source]).sum()
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
    print('provider accuracy    | {:.2f}'.format(provider_accuracy))
    print_single('oracle accuracy     ', oracle_accuracy)
    print_single('PrivateRank accuracy', PR_accuracy)
    print_single('PrivateWalk accuracy', PW_accuracy)
    print_single('random accuracy     ', random_accuracy)
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

with open('out/config.txt', 'w') as f:
    print(len(bs), file=f)

with open('out/{}.txt'.format(args.block), 'w') as f:
    print(provider_accuracy, file=f)
    print(provider_minimum, file=f)
    print(provider_entropy, file=f)

    for k, p in enumerate(bs):
        print(oracle_accuracy[k], file=f)
        print(oracle_minimum[k], file=f)
        print(oracle_entropy[k], file=f)

    for k, b in enumerate(bs):
        print(PR_accuracy[k], file=f)
        print(PR_minimum[k], file=f)
        print(PR_entropy[k], file=f)

    for k, p in enumerate(bs):
        print(PW_accuracy[k], file=f)
        print(PW_minimum[k], file=f)
        print(PW_entropy[k], file=f)

    for k, p in enumerate(bs):
        print(random_accuracy[k], file=f)
        print(random_minimum[k], file=f)
        print(random_entropy[k], file=f)
