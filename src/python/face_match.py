import numpy as np


def dist(base_emb, known_embs, names):
    dist = {}
    for i in range(len(known_embs)):
        identity = names[i]
        embedding = known_embs[i]
        dist[identity] = np.linalg.norm(base_emb - embedding)

    return min(dist, key=dist.get)


