import cv2
import copy
import numpy

from annfmp import ANNField

### Point these to your image pair ------------
image_a = cv2.imread("../data/example/1.jpg")
image_b = cv2.imread("../data/example/2.jpg")

print("Shape of image A: {}".format(image_a.shape))
print("Shape of image B: {}".format(image_b.shape))

# compute ANN field
print("Computing ANN field ...")
model = ANNField(
    n_neighbors=8,
    psize=8,
    dim_reduced=40,
    leaf_size=128,
    n_subset=1000,
    algorithm="openclopt",
    propagate=True,
    select_best_nn=True,
    verbose=1,
    seed=0,
    n_jobs=16,
)
nn_indices = model.fit(image_a, image_b)

# compute score
print("Computing overall score ...")
patches_a_reconst = model.patches_b[nn_indices]
diff = model.patches_a.astype(numpy.float32) - patches_a_reconst.astype(numpy.float32)
l2 = numpy.mean(numpy.linalg.norm(diff, axis=1))
print("Overall L2 score: {}".format(l2))
