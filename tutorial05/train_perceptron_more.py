from collections import defaultdict

def CREATE_FEATURES(x):
    phi = defaultdict(int)
    words = x.split()
    for word in words:
        phi[word] += 1
    return phi

def PREDICT_ONE(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def UPDATE_WEIGHT(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y
