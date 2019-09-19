from ctypes import *

import numpy as np
import imageio
import cv2
from matplotlib import pyplot as plt
import pRANSAC
from datetime import datetime
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import vtk
from numba import vectorize, cuda, jit
from multiprocessing import Pool
import os
import pickle
from scipy.optimize import linear_sum_assignment
from sys import exit


import show_vtk
path = None#"/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/0/img/"

class Util:
    def __init__(self):
        intrinsic = np.loadtxt(path + 'intrinsic/intrinsic_depth.txt')
        self.fx = intrinsic[0, 0] #577.590698
        self.fy = intrinsic[1, 1]#578.729797
        self.cx = intrinsic[0, 2]#318.905426
        self.cy = intrinsic[1, 2]#242.683609
        self.EGBIS_LIB = cdll.LoadLibrary('./libegbis.so')
        self.EGBIS_LIB.segmentByNormal.restype = py_object

    def point_cloud(self, depth, rgb):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.
        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0)
        z = np.where(valid, depth / 1000, np.nan)
        x = np.where(valid, z * (c - self.cx) / self.fx, np.nan)
        y = np.where(valid, z * (r - self.cy) / self.fy, np.nan)

        points = np.dstack((x, y, z)).astype(np.float32)
        #if depth.shape != rgb.shape:
        #    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint32)
        #color = rgb[:,:,2]<<16 + rgb[:,:,1]<<8 + rgb[:,:,0]
        #color = (rgb[:, :, 1] << 16 | rgb[:, :, 0] << 8 | rgb[:, :, 2])/(256*256*256)
        #color = np.ones(color.shape) * 3
        #xyzrgb = np.dstack((points, color))
        #cloud.from_3d_array(xyzrgb.astype(np.float32))
        return points

    def create_normal_image(self, normal, points):
        # normal = cloud.make_IntegralImageNormalEstimation()
        #normal.set_NormalSmoothingSize(20)
        #normal.set_NormalEstimation_Method_COVARIANCE_MATRIX()
        #ncloud = normal.compute()
        #nImage = ncloud.to_3d_array()
        nImage = np.zeros((points.shape[0], points.shape[1], 4)).astype(np.float32)
        self.EGBIS_LIB.computeNormal(points.astype(np.float32).ctypes.data, c_int(points.shape[0]), c_int(points.shape[1]), nImage.ctypes.data)
        nImage = (nImage + 1) / 2
        #nImage[np.isnan(nImage)] = 0

        nImage = nImage * 255
        return nImage[:, :, 0:3].astype(np.float32)

    def segment_image(self, points, sigma=1.5, k = 200, min_size = 200 ):
        segImage = np.zeros((points.shape[0], points.shape[1], 3)).astype(np.float32)
        normalImage = np.zeros((points.shape[0], points.shape[1], 4)).astype(np.float32)
        num_ccs = c_int(0)
        (sizes, ids, (xs, yx)) = self.EGBIS_LIB.segmentByNormal(points.astype(np.float32).ctypes.data, points.shape[0], points.shape[1], normalImage.ctypes.data, segImage.ctypes.data, c_float(sigma), c_float(k),
                              c_int(min_size), byref(num_ccs))

        normalImage = (normalImage + 1) * 127.5

        #plt.imshow(normalImage[:, :, 0:3].astype(np.uint8), interpolation='nearest')
        #plt.imshow(segImage.astype(np.uint8), interpolation='nearest')
        #plt.show()
        #exit(0)
        return normalImage[:, :, 0:3].astype(np.uint8), segImage, sizes, xs, yx, ids

util = None

#path = "/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/0/img/"


def extractPlanes(I, t0=[0,0,0]):
    depth = imageio.imread(path + 'depth/' + str(I) + '.png')
    T = np.loadtxt(path + 'pose/' + str(I) + '.txt')
    R = T[0:3, 0:3]
    t = T[0:3, 3] - t0

    # color = imageio.imread('color/0.jpg')

    points = util.point_cloud(depth, None)
    points = points.dot(R.T) + t
    normalImg, seg, sizes, xs, ys, ids = util.segment_image(points, sigma=0.9, k=200, min_size=1000)

    # plt.imshow(seg.astype(np.uint8) * 10, interpolation='nearest')
    # plt.show()

    imageio.imwrite(path + 'normal/' + str(I) + '.png', normalImg)
    imageio.imwrite(path + 'plane/' + str(I) + '.png', seg.astype(np.uint8))
    pointsArray = points[ys, xs]
    starts = np.cumsum(sizes) - sizes
    N = len(sizes)
    if N == 0:
        return
    threads_per_block = 256
    blocks = N
    seed = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 10000
    Ws = np.zeros(shape=(N, threads_per_block, 4), dtype=np.float32)
    percents = np.zeros(shape=(N, threads_per_block), dtype=np.float32)
    pRANSAC.cpu_RANSAC(Ws, percents, pointsArray, sizes, starts)
    p = np.argmax(percents, axis=1)
    i = np.arange(p.shape[0])
    W = Ws[i, p]
    P = percents[i, p]
    sizes = np.asanyarray(sizes)
    ids = np.asanyarray(ids)
    mask = (sizes > 1000) & (P > 0.70)
    Pin = pRANSAC.cpu_removeOutliers(W[mask], P[mask], pointsArray, sizes[mask], starts[mask])
    w = pRANSAC.fitBestPlane(Pin)

    print("Wrting frame", I)
    # np.save(path+'data/points'+str(I), pointsArray)
    np.save(path + 'data/starts' + str(I), starts[mask])
    np.save(path + 'data/sizes' + str(I), sizes[mask])
    np.save(path + 'data/W' + str(I), w)
    np.save(path + 'data/ids' + str(I), ids[mask])
    # np.save(path+'data/P'+str(I), p)
    # np.save(path+'data/Pall'+str(I), P)
    np.save(path + 'data/Pin' + str(I), Pin)

def extractAllPlanes():
    os.makedirs(path + 'normal', exist_ok=True)
    os.makedirs(path + 'plane', exist_ok=True)
    os.makedirs(path + 'data', exist_ok=True)
    #rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)
    #pRANSAC.my_RANSAC[N, threads_per_block](Ws, percents, pointsArray, sizes, starts, rng_states)
    #normal = pcl.IntegralImageNormalEstimation(util.cloud)

    T = np.loadtxt(path + 'pose/' + str(0) + '.txt')
    t0 = T[0:3, 3]
    p = Pool(8)
    frames = range(200, 210, 10)
    p.starmap(extractPlanes, list(zip(frames, [t0]*len(frames))))
    #for I in range(0, 1735, 10):
    #    extractPlanes(I, t0)
    return

    #plt.imshow(seg, interpolation='nearest')
    #plt.show()
    #visual = pcl.pcl_visualization.CloudViewing()
    # PointXYZ
    #visual.ShowColorCloud(cloud, b'cloud')
    #visual.ShowMonochromeCloud(cloud, b'cloud')
    # visual.ShowColorCloud(ptcloud_centred, b'cloud')
    # visual.ShowColorACloud(ptcloud_centred, b'cloud')

    #v = True
    #while v:
    #    v = not(visual.WasStopped())
    #showPlanes(Ws[i, p, 0:3], Ws[i, p, 3])

    #show_vtk.show_cloud(pointsArray)
    #show_vtk.show_planes(Pin, W)

global counter
counter = 0
M = {}
Dic = {}
size = (320, 240)

Images = {}

Dataset = 0
DIR = '/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/cop-test/'

def createPositiveMatch(n, m):
    nn = str(Dataset) + '-' + str(n)
    mm = str(Dataset) + '-' + str(m)
    rgb_a = None #np.load(path + "matchD-10/" + str(n) + '.rgb.npy')
    rgb_np = None #np.load(path + "matchD-10/" + str(m) + '.rgb.npy')


    if nn not in Images:
        rgb_a = cv2.resize(imageio.imread(path + 'color/' + str(n) + '.jpg'), dsize=size, interpolation=cv2.INTER_LINEAR)
        nor = cv2.resize(imageio.imread(path + 'normal/' + str(n) + '.png'), dsize=size,
                           interpolation=cv2.INTER_LINEAR)
        Images[nn] = nn
        np.save(DIR + 'images/' + nn + '.rgb', rgb_a)
        np.save(DIR + 'images/' + nn + '.n', nor)
    else:
        rgb_a = np.load(DIR + 'images/' + nn + '.rgb.npy')

    if mm not in Images:
        rgb_np = cv2.resize(imageio.imread(path + 'color/' + str(m) + '.jpg'), dsize=size,
                           interpolation=cv2.INTER_LINEAR)
        nor = cv2.resize(imageio.imread(path + 'normal/' + str(m) + '.png'), dsize=size,
                       interpolation=cv2.INTER_LINEAR)
        Images[mm] = mm
        np.save(DIR + 'images/' + mm + '.rgb', rgb_np)
        np.save(DIR + 'images/' + mm + '.n', nor)
    else:
        rgb_np = np.load(DIR + 'images/' + mm + '.rgb.npy')

    W1 = np.load(path + 'data/W' + str(n) + '.npy')
    Pin1 = np.load(path + 'data/Pin' + str(n) + '.npy', allow_pickle=True)
    ids1 = np.load(path + 'data/ids' + str(n) + '.npy')

    W2 = np.load(path + 'data/W' + str(m) + '.npy')
    Pin2 = np.load(path + 'data/Pin' + str(m) + '.npy', allow_pickle=True)
    ids2 = np.load(path + 'data/ids' + str(m) + '.npy')

    mask1 = imageio.imread(path + 'plane/' + str(n) + '.png')[:, :, 0]
    mask2 = imageio.imread(path + 'plane/' + str(m) + '.png')[:, :, 0]

    l1 = min(10, len(W1))
    l2 = min(20, len(W2))
    w1 = W1[0:l1, :]
    w2 = W2[0:l2, :]

    D = np.zeros((len(w1), len(w2)), np.float32)
    for i in range(len(w1)):
        for j in range(len(w2)):
            D[i, j] = 1000 * (np.linalg.norm(np.abs(Pin1[i].dot(w2[j, 0:3]) - w2[j, 3])) / len(Pin1[i]) +
                       np.linalg.norm(np.abs(Pin2[j].dot(w1[i, 0:3]) - w1[i, 3])) / len(Pin2[j]))

    S = np.zeros((len(w1), min(len(w2), len(w1))), np.float32) # Similarity
    for i in range(len(w1)):
        for j in range(S.shape[1]):
            S[i, j] = 1.0 / (np.linalg.norm(np.abs(Pin1[i].dot(w2[j, 0:3]) - w2[j, 3])) / len(Pin1[i]) +
                              np.linalg.norm(np.abs(Pin2[j].dot(w1[i, 0:3]) - w1[i, 3])) / len(Pin2[j]))

            #D2[j, i] = np.linalg.norm(np.abs(Pin2[j].dot(w1[i, 0:3]) - w1[i, 3])) / len(Pin2[j])

    r_macth, c_match = linear_sum_assignment(D)
    good = D[r_macth, c_match] < 0.5
    r_macth = r_macth[good]
    c_match = c_match[good]

    r_differ, c_differ = linear_sum_assignment(S)

    #m1, d1 = np.argmin(D1, axis=1), 1000 * np.min(D1, axis=1)
    #m2, d2 = np.argmin(D2, axis=1), 1000 * np.min(D2, axis=1)
    #match2 = ((m1[m2] == np.arange(0, l2)) & (d2 < 0.5))
    #match1 = (m2[m1] == np.arange(0, l1)) & (d1 < 0.5)

    #match1 = np.array(list(zip(np.arange(0, len(m1))[match1], m1[match1])))
    #match2 = np.array(list(zip(m2[match2], np.arange(0, len(m2))[match2])))

    #match = np.unique(np.append(match1, match2).reshape(-1, 2).astype(np.int), axis=0)
    #match = match[match[:,0] < 10]

    global counter
    print(counter, n, m)
    if len(r_differ) == 0:
        return

    #differ, maxDis = np.argmax(D1, axis=1), 1000 * np.max(D1, axis=1)
    # differ=np.arange(0, len(w2))
    # differ[match1[:, 1]] = -1
    # differ = differ[differ != -1]
    # np.random.shuffle(differ)
    # differ = np.append(differ, maxIndex)




    for (r, c) in zip(r_macth, c_match):
        m1 = ids1[r]
        m2 = ids2[c]
        s = ids2[np.argmin(S[r])]
        if r in r_differ:
            s = ids2[c_differ[np.where(r_differ == r)]][0]
        K = (str(Dataset) + '-' + str(n) + '-' + str(m1),
             str(Dataset) + '-' + str(m) + '-' + str(m2),
             str(Dataset) + '-' + str(m) + '-' + str(s))
        masks = [cv2.resize((mask1 == m1).astype(np.int8) * 255, dsize=size, interpolation=cv2.INTER_LINEAR),
                 cv2.resize((mask2 == m2).astype(np.int8) * 255, dsize=size, interpolation=cv2.INTER_LINEAR),
                 cv2.resize((mask2 == s).astype(np.int8) * 255, dsize=size, interpolation=cv2.INTER_LINEAR)]
        for k in range(len(K)):
            if K[k] not in M:
                M[K[k]] = 1
                np.save(DIR + 'planes/' + str(K[k]), masks[k])

        Dic[counter] = (K, (nn, mm, mm))

        a = (0.2 * np.stack((masks[0], np.zeros_like(masks[0]), np.zeros_like(masks[0])), axis=-1)).astype(np.uint8)
        pos = (0.2 * np.stack((masks[1], np.zeros_like(masks[0]), np.zeros_like(masks[0])), axis=-1)).astype(np.uint8)
        neg = (0.2 * np.stack((np.zeros_like(masks[0]), np.zeros_like(masks[0]), masks[2]), axis=-1)).astype(np.uint8)

        # imageio.imwrite('{0}/match/{1}-{2}-{3}-a.jpg'.format(path, counter, m1, n), rgb_a + a)
        # imageio.imwrite('{0}/match/{1}-{2}-{3}-p.jpg'.format(path, counter, m2, m), rgb_np + pos)
        # imageio.imwrite('{0}/match/{1}-{2}-{3}-n.jpg'.format(path, counter, s,  m), rgb_np + neg)
        img = np.zeros((2 * rgb_a.shape[0], 2 * rgb_a.shape[1], rgb_a.shape[2]), np.uint8)
        l = (np.int) (rgb_a.shape[0]/2)
        img[l: 3 * l, 0:rgb_a.shape[1]] = rgb_a + a
        img[0: rgb_a.shape[0] , rgb_a.shape[1]:] = rgb_np + pos
        img[rgb_a.shape[0] :  , rgb_a.shape[1]:] = rgb_np + neg

        imageio.imwrite('{0}/visual/{1}-{2}-{3}-{4}.jpg'.format(DIR, counter, Dataset, n, m), img)

        counter = counter + 1


if __name__ == "__main__":
    #extractAllPlanes()
    #exit(0)
    # import cProfile
    # cProfile.run('main()', sort='time')
    np.set_printoptions(precision=3)
    n = 0
    m = 100
    path = "/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/" + str(0) + "/img/"
    util = Util()
    # counter = 0
    # for d in [(0, 5575), (3, 1735), (5, 1446), (10, 1364)]:
    #     Dataset = d[0]
    #     path = "/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/" + str(Dataset) + "/img/"
    #     for l in [250]:
    #         for i in range(40, d[1] - l, 80):
    #             createPositiveMatch(i, i + l)
    # f = open(DIR+"/train.pkl", "wb")
    # pickle.dump(Dic, f)
    # exit(0)
    W1 = np.load(path + 'data/W' + str(n)+'.npy')
    Pin1 = np.load(path + 'data/Pin' + str(n)+'.npy', allow_pickle=True)
    ids1 = np.load(path + 'data/ids' + str(n) + '.npy')

    W2 = np.load(path + 'data/W' + str(m) + '.npy')
    Pin2 = np.load(path + 'data/Pin' + str(m) + '.npy', allow_pickle=True)
    ids2 = np.load(path + 'data/ids' + str(m) + '.npy')

    T1 = np.loadtxt(path + 'pose/'+str(n)+'.txt')
    T2 = np.loadtxt(path + 'pose/'+str(m)+'.txt')
    R1 = T1[0:3, 0:3]
    t1 = T1[0:3, 3]

    R2 = T2[0:3, 0:3]
    t2 = T2[0:3, 3] - t1
    depth = imageio.imread(path + 'depth/' + str(n) + '.png')
    mask1 = imageio.imread(path + 'plane/' + str(n) + '.png')[:,:,0]
    color1 = cv2.resize(imageio.imread(path + 'color/' + str(n) + '.jpg'), dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
    points1 = util.point_cloud(depth , None).reshape((-1, 3))

    depth = imageio.imread(path + 'depth/' + str(m) + '.png')
    mask2 = imageio.imread(path + 'plane/' + str(m) + '.png')[:,:,0]
    color2 = cv2.resize(imageio.imread(path + 'color/' + str(m) + '.jpg'), dsize=(depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_CUBIC)

    points2 = util.point_cloud(depth , None).reshape((-1, 3))



    l = 20
    w1 = W1[0:l, :]
    w2 = W2[0:l, :]



    #for p in Pin1:
    #    show_vtk.show_planes([p], np.random.rand(3))

    show_vtk.show_planes(Pin1, [0, 1, 0])
    show_vtk.show_planes(Pin2, [1, 0, 0])

    # show_vtk.show_cloud(Pin1[7], [0, 1, 0])
    # show_vtk.show_cloud(Pin2[4], [1, 0, 0])

    #show_vtk.show_cloud(points1.dot(R1.T) , [0, 1, 0])
    #show_vtk.show_cloud(points2.dot(R2.T) + t2, [1, 0, 1])

    show_vtk.showVTK()
    #
    #extractPlanes(0)
    exit(0)

