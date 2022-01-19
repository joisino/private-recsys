import sys

n = int(sys.argv[1])

with open('out/config.txt') as f:
    L = int(f.readline())

provider_accuracy = 0
oracle_accuracy = [0 for j in range(L)]
PR_accuracy = [0 for j in range(L)]
PW_accuracy = [0 for j in range(L)]
random_accuracy = [0 for j in range(L)]

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

PW_evaluation = [0 for j in range(L)]
rwp_evaluation = [0 for j in range(L)]

for i in range(n):
    with open(f'out/{i}.txt') as f:
        provider_accuracy += float(f.readline())
        provider_minimum += float(f.readline())
        provider_entropy += float(f.readline())

        for k in range(L):
            oracle_accuracy[k] += float(f.readline())
            oracle_minimum[k] += float(f.readline())
            oracle_entropy[k] += float(f.readline())

        for k in range(L):
            PR_accuracy[k] += float(f.readline())
            PR_minimum[k] += float(f.readline())
            PR_entropy[k] += float(f.readline())

        for k in range(L):
            PW_accuracy[k] += float(f.readline())
            PW_minimum[k] += float(f.readline())
            PW_entropy[k] += float(f.readline())

        for k in range(L):
            random_accuracy[k] += float(f.readline())
            random_minimum[k] += float(f.readline())
            random_entropy[k] += float(f.readline())

print('provider accuracy:', provider_accuracy)
print('oracle accuracy:', oracle_accuracy)
print('PR accuracy:', PR_accuracy)
print('PW ccuracy:', PW_accuracy)
print('random accuracy:', random_accuracy)

print('provider minimum:', provider_minimum)
print('oracle minimum:', oracle_minimum)
print('PR minimum:', PR_minimum)
print('PW minimum:', PW_minimum)
print('random minimum:', random_minimum)

print('provider entropy:', provider_entropy)
print('oracle entropy:', oracle_entropy)
print('PR entropy:', PR_entropy)
print('PW entropy:', PW_entropy)
print('random entropy:', random_entropy)
