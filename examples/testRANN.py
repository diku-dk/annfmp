import cv2
import copy
import numpy

from annfmp import ANNField


print("Computing ANN field with RANN ...")

modelRANN = ANNField(
    n_neighbors=1,
    psize=8,
    dim_reduced=40,
    height=9,
    n_subset=1000,
    algorithm="rann",
    propagate=True,
    select_best_nn=True,
    verbose=1,
    seed=0,
    n_jobs=16,
)

image_a = cv2.imread("../data/example/1200/5.jpg")
image_b = cv2.imread("../data/example/1200/6.jpg")

if image_a is None:
    raise RuntimeError("Could not read image_a")

if image_b is None:
    raise RuntimeError("Could not read image_b")

print("Shape of image A:", image_a.shape)
print("Shape of image B:", image_b.shape)

nn_indices_rann = modelRANN.fit(image_a, image_b)

print("Returned shape:", nn_indices_rann.shape)
print("First 20 indices:", nn_indices_rann[:20])

n_patches_a = modelRANN._model.patches_a.shape[0]
n_patches_b = modelRANN._model.patches_b.shape[0]

print("Number of patches A:", n_patches_a)
print("Number of patches B:", n_patches_b)

assert nn_indices_rann.shape[0] == n_patches_a
assert nn_indices_rann.min() >= 0
assert nn_indices_rann.max() < n_patches_b

print("RANN returned valid k-NN indices!")