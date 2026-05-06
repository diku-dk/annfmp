import cv2
import copy
import numpy

from annfmp import ANNField

# compute ANN field with openclirreg
print("Computing ANN field with openclirreg ...")
modelOpenclirreg = ANNField(
    n_neighbors=8,
    psize=8,
    dim_reduced=40,
    height=9,
    n_subset=1000,
    algorithm="openclirreg",
    propagate=True,
    select_best_nn=True,
    verbose=1,
    seed=0,
    n_jobs=16,
)

# compute ANN field with openclopt
print("Computing ANN field with openclopt...")
modelOpenclopt = ANNField(
    n_neighbors=8,
    psize=8,
    dim_reduced=40,
    leaf_size=654,
    n_subset=1000,
    algorithm="openclopt",
    propagate=True,
    select_best_nn=True,
    verbose=1,
    seed=0,
    n_jobs=16,
)


# compute ANN field with RANN
print("Computing ANN field with RANN...")
modelRANN = ANNField(
    n_neighbors=8,
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

flags = []

### Point these to your image pair ------------
for x in range(5, 7, 2):
    print(f"\n Current x: {x}\n")
    image_a = cv2.imread(f"../data/example/1200/{x}.jpg")
    image_b = cv2.imread(f"../data/example/1200/{x+1}.jpg")

    print("Shape of image A: {}".format(image_a.shape))
    print("Shape of image B: {}".format(image_b.shape))

    nn_indices_openclirreg = modelOpenclirreg.fit(image_a, image_b)

    nn_indices_openclopt = modelOpenclopt.fit(image_a, image_b)

    nn_indices_rann = modelRANN.fit(image_a, image_b)

    # compute score
    print("Computing OpenCLIRREG overall score ...")
    patches_a_reconst = modelOpenclirreg.patches_b[nn_indices_openclirreg]
    diff = modelOpenclirreg.patches_a.astype(numpy.float32) - patches_a_reconst.astype(numpy.float32)
    l2Openclirreg = numpy.mean(numpy.linalg.norm(diff, axis=1))
    print("OpenCLIRREG L2 score: {}".format(l2Openclirreg))

    print("Computing OpenCLOPT overall score ...")
    patches_a_reconst = modelOpenclopt.patches_b[nn_indices_openclopt]
    diff = modelOpenclopt.patches_a.astype(numpy.float32) - patches_a_reconst.astype(numpy.float32)
    l2Openclopt = numpy.mean(numpy.linalg.norm(diff, axis=1))
    print("OpenCLOPT L2 score: {}".format(l2Openclopt))

    print("Computing RANN overall score ...")
    patches_a_reconst = modelRANN._model.patches_b[nn_indices_rann]
    diff = modelRANN._model.patches_a.astype(numpy.float32) - patches_a_reconst.astype(numpy.float32)
    l2RANN = numpy.mean(numpy.linalg.norm(diff, axis=1))
    print("RANN L2 score: {}".format(l2RANN))

    print("-----------------------------------------")
    print("L2 score summary")
    print("-----------------------------------------")
    print("OpenCLIRREG:\t{}".format(l2Openclirreg))
    print("OpenCLOPT:\t{}".format(l2Openclopt))
    print("RANN:\t\t{}".format(l2RANN))
    print("-----------------------------------------")
