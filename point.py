from ctypes import *

import numpy as np
import pcl
#import pcl.pcl_visualization
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
        self.EGBIS_LIB.runEgbisOnMat.restype = py_object

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
        cloud = pcl.PointCloud()
        points = np.dstack((x, y, z)).astype(np.float32)
        if depth.shape != rgb.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint32)
        #color = rgb[:,:,2]<<16 + rgb[:,:,1]<<8 + rgb[:,:,0]
        color = (rgb[:, :, 1] << 16 | rgb[:, :, 0] << 8 | rgb[:, :, 2])/(256*256*256)
        color = np.ones(color.shape) * 3
        xyzrgb = np.dstack((points, color))
        #cloud.from_3d_array(xyzrgb.astype(np.float32))
        cloud.from_3d_array(points.astype(np.float32))
        return cloud, points

    def create_normal_image(self, normal):
        # normal = cloud.make_IntegralImageNormalEstimation()
        normal.set_NormalSmoothingSize(20)
        normal.set_NormalEstimation_Method_COVARIANCE_MATRIX()
        ncloud = normal.compute()
        nImage = ncloud.to_3d_array()
        nImage = (nImage + 1) / 2
        #nImage[np.isnan(nImage)] = 0
        nImage = nImage * 255
        return nImage[:, :, 0:3].astype(np.float32)

    def segment_image(self, image, sigma=1, k = 200, min_size = 200 ):
        segImage = image.copy()
        num_ccs = c_int(0)
        (sizes, (xs, yx)) = self.EGBIS_LIB.runEgbisOnMat(image.ctypes.data, image.shape[0], image.shape[1], segImage.ctypes.data, c_float(sigma), c_float(k),
                              c_int(min_size), byref(num_ccs))
        return segImage, sizes, xs, yx

def main():
    # cloud = pcl.load_XYZRGB(
    #     './examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')
    #cloud = pcl.load("table_scene_lms400.pcd")

    #depth = cv2.imread('/media/mahdi/4418B81419D11C10/media/private/dataset/scannet/9/depth/0.png', 0)
    depth = imageio.imread('0.png')
    color = imageio.imread('color/0.jpg')
    util = Util()
    cloud, points = util.point_cloud(depth/100.0, color)
    #cloud = pcl.load_XYZRGB('/home/mahdi/PycharmProjects/Pcl/python-pcl-0.3.0rc1/examples/pcldata/tutorials/table_scene_mug_stereo_textured.pcd')

    normal = cloud.make_IntegralImageNormalEstimation()

    normalImg = util.create_normal_image(normal)

    seg, sizes, xs, ys = util.segment_image(normalImg, sigma=0.9, k=200, min_size=1000)

    #plt.imshow(seg.astype(np.uint8), interpolation='nearest')

    #plt.show()

    pointsArray = points[ys, xs]
    starts = np.cumsum(sizes) - sizes

    N = len(sizes)
#    N = 5
    threads_per_block = 256
    blocks = N
    seed = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 10000
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)

    Ws = np.zeros(shape=(N, threads_per_block, 4), dtype=np.float32)
    percents = np.zeros(shape=(N, threads_per_block), dtype=np.float32)

    #pRANSAC.my_RANSAC[N, threads_per_block](Ws, percents, pointsArray, sizes, starts, rng_states)
    pRANSAC.cpu_RANSAC(Ws, percents, pointsArray, sizes, starts)
    p = np.argmax(percents, axis=1)
    #print(np.max(percents, axis=1))
    i = np.arange(p.shape[0])
    W = Ws[i, p]
    P = percents[i, p]
    print(W)
    print(P)

    Pin, P = pRANSAC.cpu_removeOutliers(W, P, pointsArray, sizes, starts)

    np.save('points', pointsArray)
    np.save('starts', starts)
    np.save('sizes', sizes)
    np.save('W', W)
    np.save('P', p)
    np.save('Pall', P)
    np.save('Pin', Pin)

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

    show_vtk.show_planes(Pin, W)

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
    exit(0)
    sizes = [0] * 20
    Ws = np.zeros((len(sizes), 4))
    percents = np.zeros((len(sizes), 1))

    threads_per_block = 40
    blocks = len(sizes)
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    out = np.zeros(threads_per_block * blocks, dtype=np.float32)

    my_RANSAC[blocks, threads_per_block, 2000](Ws, percents, [0], sizes, [0], rng_states)

