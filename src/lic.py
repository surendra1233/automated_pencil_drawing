import numpy as np


def label_regions(im, num_regions):
    """
    Builds an 8-connected graph of pixels, with edge weights being the
    l2 distance between pixel colors in whatever color space they are in.
    Then, finds the minimum-cost spanning forest with num_regions components.
    """

    H, W, C = im.shape
    J, I = np.meshgrid(np.arange(W), np.arange(H))

    edges = []
    indices = I * W + J

    h_cost = np.sum((im[:, :-1, :] - im[:, 1:, :]) ** 2, axis=2) ** 0.5
    h_edges = list(
        zip(h_cost.ravel(), indices[:, :-1].ravel(), indices[:, 1:].ravel()))
    edges += h_edges

    v_cost = np.sum((im[:-1, :, :] - im[1:, :, :]) ** 2, axis=2) ** 0.5
    v_edges = list(
        zip(v_cost.ravel(), indices[:-1, :].ravel(), indices[1:, :].ravel()))
    edges += v_edges

    dr_cost = np.sum((im[:-1, :-1, :] - im[1:, 1:, :]) ** 2, axis=2) ** 0.5
    dr_edges = list(
        zip(dr_cost.ravel(), indices[:-1, :-1].ravel(), indices[1:, 1:].ravel()))
    edges += dr_edges

    ur_cost = np.sum((im[1:, :-1, :] - im[:-1, 1:, :]) ** 2, axis=2) ** 0.5
    ur_edges = list(
        zip(ur_cost.ravel(), indices[1:, :-1].ravel(), indices[:-1, 1:].ravel()))
    edges += ur_edges

    edges.sort()

    p, rank, components = np.arange(H * W), np.zeros(H * W), H * W

    def parent(x):
        if p[x] == x:
            return x
        p[x] = parent(p[x])
        return p[x]

    for cost, x, y in edges:
        if components <= num_regions:
            break

        x = parent(x)
        y = parent(y)
        if x == y:
            continue
        if rank[x] > rank[y]:
            x, y = y, x
        if rank[x] == rank[y]:
            rank[y] += 1
        p[x] = y
        components -= 1

    component_labels = dict()
    labels = np.zeros((H, W), dtype=np.int32)
    label_counts = []
    for i in range(H):
        for j in range(W):
            pi = parent(i * W + j)
            if pi not in component_labels:
                component_labels[pi] = len(component_labels)
                label_counts.append(0)
            labels[i, j] = component_labels[pi]
            label_counts[labels[i, j]] += 1

    return labels, label_counts
