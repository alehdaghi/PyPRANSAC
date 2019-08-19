# bool runEgbisOnMat(const uint8_t* input, int h, int w, uint8_t* output, float sigma, float k, int min_size, int *numccs)
import numpy as np
import ctypes
from ctypes import *
import imageio
from matplotlib import pyplot as plt


testLib = ctypes.cdll.LoadLibrary('./libegbis.so')
testLib.runEgbisOnMat.restype = py_object
a = np.arange(150000).reshape(500, -1, 3).astype(np.uint8)
#s = testLib.mysum2(15, 10, a.ctypes.data)
#b= a.reshape(-1,5,3)
#s = testLib.convertArrayToNativeImage(b.ctypes.data, 10, 5)
img = imageio.imread('../images/chimps.jpg').astype(np.uint8)
seg = img.copy()
num_ccs = ctypes.c_int(0)
#print(img.ctypes.data)

segs = testLib.runEgbisOnMat(img.ctypes.data, img.shape[0], img.shape[1], seg.ctypes.data, c_float(1), c_float(500.), c_int(200 ), byref(num_ccs));

#imageio.imwrite( '../images/seg.jpg', seg)
print(segs[0])
#plt.imshow(seg, interpolation='nearest')
#plt.show()
print(num_ccs.value)
#testLib.foo.restype = py_object
#a = testLib.foo()


