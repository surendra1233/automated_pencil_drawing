import numpy as np
import cv2 as cv

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

def extract_region_vector_field(im_gray, labels, label_counts,
                                preblur_sigma=2,
                                preblur_size=7,
                                var_threshold=0.5,
                                normalize=True):
    """
    @params
        im_gray: numpy_array grayscale image,
        labels: the labels caluclated by connected component analysis,
        label_counts: the corresponding label_counts,
        dmap: the dmap of the image,
        threshold: the threshold that needs to be used for dmap
        ...
    Generates the vector field based on the mean intensities of
    each of the regions and subtracting it locally then
    taking the max variance direction
    and assigns it as the vector for that region
    """

    # Blur, then compute image gradients
    im_blur = cv.GaussianBlur(
        im_gray, (preblur_size, preblur_size), preblur_sigma)
    gradX = cv.Sobel(im_blur, cv.CV_64F, 1, 0, ksize=5)
    gradY = cv.Sobel(im_blur, cv.CV_64F, 0, 1, ksize=5)

    # Rotate gradients by 90 degrees to align with stripe direction
    vec = np.dstack([gradX, -gradY]) / 255.0

    # Constrain gradients to face right
    vec = vec * (1 - 2 * (vec[:, :, 1] < 0))[:, :, np.newaxis]

    # Compute variance of each region
    H, W = im_gray.shape
    K = len(label_counts)
    mean_vec, var_vec = np.zeros((K, 2)), np.zeros(K)
    for i in range(H):
        for j in range(W):
            mean_vec[labels[i, j]] += vec[i, j, :]
    mean_vec /= np.array(label_counts)[:, np.newaxis]
    for i in range(H):
        for j in range(W):
            l = labels[i, j]
            var_vec[l] += np.sum((vec[i, j, :] - mean_vec[l]) ** 2)
    var_vec /= np.array(label_counts)

    # Set regions with high variance to their mean vector
    for i in range(H):
        for j in range(W):
            l = labels[i, j]
            if var_vec[l] > var_threshold:
                vec[i, j, :] = mean_vec[l]

    if normalize:
        vec /= (np.sum(vec ** 2, axis=2) + 1e-2)[:, :, np.newaxis]

    return vec

