import cv2
import copy
import numpy

from annfmp import ANNField

# compute ANN field with openclopt
print("Computing ANN field with openclopt ...")
modelOpencl = ANNField(
    n_neighbors=8,
    psize=8,
    dim_reduced=40,
    leaf_size=128,
    height=8,
    n_subset=1000,
    algorithm="openclopt",
    propagate=True,
    select_best_nn=True,
    verbose=1,
    seed=0,
    n_jobs=16,
)

# compute ANN field with openmp
print("\nComputing ANN field with openmp ...")
modelOpenmp = ANNField(
    n_neighbors=8,
    psize=8,
    dim_reduced=40,
    leaf_size=128,
    n_subset=1000,
    algorithm="openmp",
    propagate=True,
    select_best_nn=True,
    verbose=1,
    seed=0,
    n_jobs=16,
)

flags = []

### Point these to your image pair ------------
for x in range(1, 3, 2):
    print(f"\n Current x: {x}\n")
    image_a = cv2.imread(f"../data/example/500/{x}.jpg")
    image_b = cv2.imread(f"../data/example/500/{x+1}.jpg")

    print("Shape of image A: {}".format(image_a.shape))
    print("Shape of image B: {}".format(image_b.shape))

    nn_indices_opencl = modelOpencl.fit(image_a, image_b)
    nn_indices_openmp = modelOpenmp.fit(image_a, image_b)

    # compute score
    #print("Computing OpenCL overall score ...")
    patches_a_reconst = modelOpencl.patches_b[nn_indices_opencl]
    diff = modelOpencl.patches_a.astype(numpy.float32) - patches_a_reconst.astype(numpy.float32)
    l2Opencl = numpy.mean(numpy.linalg.norm(diff, axis=1))
    #print("Overall L2 score: {}".format(l2Opencl))

    # compute score
    #print("Computing OpenMP overall score ...")
    patches_a_reconst = modelOpenmp.patches_b[nn_indices_openmp]
    diff = modelOpenmp.patches_a.astype(numpy.float32) - patches_a_reconst.astype(numpy.float32)
    l2Openmp = numpy.mean(numpy.linalg.norm(diff, axis=1))
    #print("Overall L2 score: {}".format(l2Openmp))

    if (l2Opencl - l2Openmp > 0.5):
        flags.append([x, l2Opencl, l2Openmp])

for flag in flags:
    print("\nBad L2 score for opencl found!")
    print(f"Image A: {flag[0]} \nImage B: {flag[0]+1}")
    print(f"Opencl score: {flag[1]}, openmp score: {flag[2]}\n")
    print(f"Difference: {flag[1] - flag[2]}")