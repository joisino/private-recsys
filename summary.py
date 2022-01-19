import sys

n = int(sys.argv[1])

with open('out/config.txt') as f:
    L = int(f.readline())

provider_recall = 0
provider_nDCG = 0
oracle_recall = [0 for j in range(L)]
oracle_nDCG = [0 for j in range(L)]
PR_recall = [0 for j in range(L)]
PR_nDCG = [0 for j in range(L)]
PW_recall = [0 for j in range(L)]
PW_nDCG = [0 for j in range(L)]

random_recall = [0 for j in range(L)]
random_nDCG = [0 for j in range(L)]

provider_minimum = 0
provider_entropy = 0
oracle_minimum = [0 for j in range(L)]
oracle_entropy = [0 for j in range(L)]
PR_minimum = [0 for j in range(L)]
PR_entropy = [0 for j in range(L)]
PW_minimum = [0 for j in range(L)]
PW_entropy = [0 for j in range(L)]

random_minimum = [0 for j in range(L)]
random_entropy = [0 for j in range(L)]

for i in range(n):
    with open('out/{}.txt'.format(i)) as f:
        provider_recall += float(f.readline())
        provider_nDCG += float(f.readline())
        provider_minimum += float(f.readline())
        provider_entropy += float(f.readline())

        for k in range(L):
            oracle_recall[k] += float(f.readline())
            oracle_nDCG[k] += float(f.readline())
            oracle_minimum[k] += float(f.readline())
            oracle_entropy[k] += float(f.readline())

        for k in range(L):
            PR_recall[k] += float(f.readline())
            PR_nDCG[k] += float(f.readline())
            PR_minimum[k] += float(f.readline())
            PR_entropy[k] += float(f.readline())

        for k in range(L):
            PW_recall[k] += float(f.readline())
            PW_nDCG[k] += float(f.readline())
            PW_minimum[k] += float(f.readline())
            PW_entropy[k] += float(f.readline())

        for k in range(L):
            random_recall[k] += float(f.readline())
            random_nDCG[k] += float(f.readline())
            random_minimum[k] += float(f.readline())
            random_entropy[k] += float(f.readline())

print('provider recall:', provider_recall)
print('oracle recall:', oracle_recall)
print('PrivateRank recall:', PR_recall)
print('PrivateWalk recall:', PW_recall)
print('random recall:', random_recall)
print('provider nDCG:', provider_nDCG)
print('oracle nDCG:', oracle_nDCG)
print('PrivateRank nDCG:', PR_nDCG)
print('PrivateWalk nDCG:', PW_nDCG)
print('random nDCG:', random_nDCG)

print('provider least count:', provider_minimum)
print('oracle least count:', oracle_minimum)
print('PrivateRank least count:', PR_minimum)
print('PrivateWalk least count:', PW_minimum)
print('random least count:', random_minimum)

print('provider entropy:', provider_entropy)
print('oracle entropy:', oracle_entropy)
print('PrivateRank entropy:', PR_entropy)
print('PrivateWalk entropy:', PW_entropy)
print('random entropy:', random_entropy)
