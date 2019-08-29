from ctypes import *

import numpy as np
import imageio
import cv2
from matplotlib import pyplot as plt
import pRANSAC
from datetime import datetime
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import vtk
import show_vtk

class Util:
    def __init__(self):
        self.fx = 577.590698
        self.fy = 578.729797
        self.cx = 318.905426
        self.cy = 242.683609
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
        valid = (depth > 0) & (depth < 255)
        z = np.where(valid, depth / 256.0, np.nan)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)

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
        (sizes, (xs, yx)) = self.EGBIS_LIB.segmentByNormal(points.astype(np.float32).ctypes.data, points.shape[0], points.shape[1], normalImage.ctypes.data, segImage.ctypes.data, c_float(sigma), c_float(k),
                              c_int(min_size), byref(num_ccs))

        normalImage = (normalImage + 1) * 127.5

        #plt.imshow(normalImage[:, :, 0:3].astype(np.uint8), interpolation='nearest')
        #plt.imshow(segImage.astype(np.uint8), interpolation='nearest')
        #plt.show()
        #exit(0)
        return normalImage[:, :, 0:3].astype(np.uint8), segImage, sizes, xs, yx

util = Util()

path = "/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/0/img/"

def extractPlanes(I):

    #depth = cv2.imread('/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/9/depth/0.png', 0)
    #plt.imshow(seg.astype(np.uint8), interpolation='nearest')
    #plt.show()

    #rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)
    #pRANSAC.my_RANSAC[N, threads_per_block](Ws, percents, pointsArray, sizes, starts, rng_states)
    #normal = pcl.IntegralImageNormalEstimation(util.cloud)
    for I in range(0, 5558):
        depth = imageio.imread('/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/0/img/depth/'+str(I) + '.png')
        #color = imageio.imread('color/0.jpg')

        points = util.point_cloud(depth/100.0, None)
        normalImg, seg, sizes, xs, ys = util.segment_image(points, sigma=0.9, k=200, min_size=1000)
        imageio.imwrite('/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/0/img/normal/'+str(I) + '.png', normalImg)
        imageio.imwrite('/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/0/img/plane/' + str(I) + '.png', seg)
        pointsArray = points[ys, xs]
        starts = np.cumsum(sizes) - sizes
        N = len(sizes)
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
        Pin, P = pRANSAC.cpu_removeOutliers(W, P, pointsArray, sizes, starts)

        np.save(path+'data/points'+str(I), pointsArray)
        np.save(path+'data/starts'+str(I), starts)
        np.save(path+'data/sizes'+str(I), sizes)
        np.save(path+'data/W'+str(I), W)
        np.save(path+'data/P'+str(I), p)
        np.save(path+'data/Pall'+str(I), P)
        np.save(path+'data/Pin'+str(I), Pin)
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


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    n = 0
    m = 200
    W1 = np.load(path + 'data/W' + str(n)+'.npy')
    pointsArray1 = np.load(path + 'data/points' + str(n)+'.npy')
    Pin1 = np.load(path + 'data/Pin' + str(n)+'.npy', allow_pickle=True)

    W2 = np.load(path + 'data/W' + str(m) + '.npy')
    pointsArray2 = np.load(path + 'data/points' + str(m) + '.npy')
    Pin2 = np.load(path + 'data/Pin' + str(m) + '.npy', allow_pickle=True)

    T1 = np.loadtxt(path + 'pose/'+str(n)+'.txt')
    T2 = np.loadtxt(path + 'pose/'+str(m)+'.txt')
    R1 = T1[0:3, 0:3]
    t1 = T1[0:3, 3]

    R2 = T2[0:3, 0:3]
    t2 = T2[0:3, 3] - t1


    print(W1.dot(R1.T))

    #show_vtk.show_cloud(pointsArray1.dot(R1.T) , [1, 0, 0])
    #show_vtk.show_cloud(pointsArray2.dot(R2.T) + t2/50, [0, 1, 0])
    show_vtk.show_planes([p.dot(R1.T) for p in Pin1], W1)
    show_vtk.show_planes([p.dot(R2.T) + t2/50 for p in Pin2], W2)
    show_vtk.showVTK()
    #
    #extractPlanes(0)
    exit(0)

