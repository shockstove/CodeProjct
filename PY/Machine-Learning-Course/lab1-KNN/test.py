from skimage import io,data
from skimage import transform as tf
import matplotlib.pylab as plt
from skimage import transform as trans
import numpy as np

### 原图
img = data.camera()
io.imshow(img)
plt.show()

### 相似变换
tform = tf.SimilarityTransform(scale=1,rotation=np.deg2rad(10))
img1 = tf.warp(img,tform)
io.imshow(img1)
plt.show()