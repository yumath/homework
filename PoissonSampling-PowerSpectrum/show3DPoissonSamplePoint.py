from poisson3D import PoissonSampling3D

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure('Poisson Sample')
ax = fig.add_subplot(111, projection='3d')


if __name__ == '__main__':

    radius = 2
    length = 20
    width = 20
    height = 20
    k = 30
    points = PoissonSampling3D(radius=radius, length=length, width=width, height=height, n_k=k)

    x, y, z = [], [], []
    for i in range(len(points)):
        x.append(points[i].x)
        y.append(points[i].y)
        z.append(points[i].z)
    # 将数据点分成三部分画，在颜色上有区分度
    ax.scatter(x, y, z, c='r', marker='.')  # 绘制数据点

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    '''# 设置坐标轴刻度
    my_x_ticks = np.arange(-5, 5, 0.5)
    my_y_ticks = np.arange(-2, 2, 0.3)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)'''
    plt.show()
    '''
    # print(gray_max, gray_min)
    # print(fushitu)
    radius = 1
    length = 20
    width = 20
    height = 0
    #img = array(Image.open("/Users/yumath/Pictures/GitHub.jpg"))

    Power = PowerSpectrum(radius, length, width, height)
    fushitu = Draw(Power, length, width, height)

    #print(fushitu)
    for i in range(length):
        for j in range(width):

            if fushitu[i][j]<1:
                fushitu[i][j] = 0
            if fushitu[i][j]>=1:
                fushitu[i][j]=255

            fushitu[i][j] = int(fushitu[i][j])
    print(fushitu)
    img = Image.fromarray(fushitu)
    #img = Image.new("L", (fushitu.shape[0], fushitu.shape[1]))
            #img.putpixel([i, j], fushitu[i][j])
    #print(img.shape)
    img.show()
    '''
