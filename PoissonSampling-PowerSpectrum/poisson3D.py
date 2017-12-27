import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import random

class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, Point):
        return math.sqrt((Point.x - self.x)**2 + (Point.y - self.y)**2 + (Point.z - self.z)**2)

    def vectorize(self):
        return np.array([self.x, self.y, self.z])

def PoissonSampling3D(radius, length, width, height, n_k):
    grid_len = radius/math.sqrt(3)    #每个小方块的长度
    _len, _wid, _hei = math.ceil(length/grid_len), math.ceil(width/grid_len), math.ceil(height/grid_len) #将空间分割为方块
    grids = -np.ones(((_len, _wid, _hei)))

    # 维持活跃点
    active_list = []
    #打出来的所有点存在points里
    points = []

    # 随机产生一个点作为起点
    x0,y0,z0=random.uniform(-length/2, length/2), random.uniform(-width/2, width/2), random.uniform(-height/2, height/2)
    p0 = Point(x0, y0, z0)
    #将点放入active_list
    active_list.append(p0)
    points.append(p0)
    #计算第0个点所在格子的索引
    x0_index = math.floor((x0+length/2)/grid_len)
    y0_index = math.floor((y0+width/2)/grid_len)
    z0_index = math.floor((z0+height/2)/grid_len)
    # 将该点所在的格子标记为其索引
    grids[x0_index][y0_index][z0_index] = 0

    while len(active_list)>0:
        index = random.randint(0, len(active_list)-1)   #从active_list中随机选取一个点画圆环
        cur_point = active_list[index]  #取出该点

        is_success_paint_dot = False
        #随机打k个点
        for i in range(n_k):
            tmp_radius = random.uniform(radius, 2*radius)
            tmp_phi = random.uniform(0, 2*math.pi)
            tmp_theta = random.uniform(0, math.pi)
            #把球坐标转化为直角坐标
            tmp_x = tmp_radius * math.sin(tmp_theta) * math.cos(tmp_phi)
            tmp_y = tmp_radius * math.sin(tmp_theta) * math.sin(tmp_phi)
            tmp_z = tmp_radius * math.cos(tmp_theta)
            #按照上面随机产生的坐标打点
            tmp_point = Point(cur_point.x+tmp_x, cur_point.y+tmp_y, cur_point.z+tmp_z)
            if tmp_point.x>length/2 or tmp_point.x<-length/2 or tmp_point.y>width/2 or tmp_point.y<-width/2 or tmp_point.z>height/2 or tmp_point.z<-height/2:
                continue
            #找到与打出来的该点相交的方块
            x_index = math.floor((tmp_point.x + length / 2) / grid_len)
            y_index = math.floor((tmp_point.y + width  / 2) / grid_len)
            z_index = math.floor((tmp_point.z + height / 2) / grid_len)
            #确定要排查的格子索引界值
            x_up = y_up = z_up = 3
            if x_index == _len-1:
                x_up=1
            if x_index+1 == _len-1:
                x_up=2
            if y_index == _wid-1:
                y_up=1
            if y_index+1 == _wid-1:
                y_up=2
            if z_index == _hei-1:
                z_up=1
            if z_index+1 == _hei-1:
                z_up=2

            x_low = y_low = z_low = -2
            if x_index == 0:
                x_low=0
            if x_index == 1:
                x_low=-1
            if y_index == 0:
                y_low=0
            if y_index == 1:
                y_low=-1
            if z_index == 0:
                z_low=0
            if z_index == 1:
                z_low=-1
            #判断随机生成的点周围的方框内有没有交点
            is_intersect = False
            for j in range(x_low, x_up):
                if is_intersect == True:    #相交了就退出当前循环
                    break
                for k in range(y_low, y_up):
                    if is_intersect == True:    #相交就退出当前循环
                        break
                    for l in range(z_low, z_up):
                        if grids[x_index+j][y_index+k][z_index+l] == -1:
                            continue
                        else:
                            if tmp_point.distance(points[int(grids[x_index+j][y_index+k][z_index+l])]) < radius:
                                is_intersect = True
                                break
            if is_intersect == False:   #如果当前随机得到的点不和周围格子内的点相交，表示打点成功
                grids[x_index][y_index][z_index] = len(points)  #把该点的索引给grids
                points.append(tmp_point)    #把该点加入points
                active_list = np.append(active_list, tmp_point)
                #print(tmp_point.x, tmp_point.y, tmp_point.z)
                is_success_paint_dot = True
                break
            else:
                continue   #表示当前节点不满足要求，继续打
        if is_success_paint_dot == False:   #如果打的n_k个点都没有成功
            active_list = np.delete(active_list, index)   #将该点从active_list中抹掉
            #print('active_list还剩下：', len(active_list))

    print('半径为：', radius, '\t打的点数：', len(points))
    return points

def PowerSpectrum3D(points, radius):
    '''
    绘制三维功率谱
    :param radius: 泊松采样的半径
    :param length: 采样区域的长
    :param width: 宽
    :param height: 高
    :return: 三维空间中功率谱的值
    '''
    N = len(points) #采样点数
    # 黑圈的半径
    black_radius = 1 / radius
    scale = 4 * black_radius
    # 设置采样强度
    Density_x = 200
    Density_y = 200
    Density_z = 200
    # 初始化为0
    P = np.zeros(((Density_x, Density_y, Density_z)))
    #计算功率值
    for i in range(Density_x):
        for j in range(Density_y):
            for k in range(Density_z):
                #计算每个频谱上采样点的f坐标值
                f_x = -2 * black_radius + scale / Density_x * i
                f_y = -2 * black_radius + scale / Density_y * j
                f_z = -2 * black_radius + scale / Density_z * k
                # 设置扰动，去掉中间的十字架
                bias = random.uniform(0, scale / max(Density_x, Density_y, Density_z))
                # 计算得到频谱图中的f值
                f = np.array([f_x + bias, f_y - bias, f_z + bias])
                cos_sum, sin_sum = 0, 0
                for l in range(N):
                    s = points[l].vectorize()
                    cos_sum += math.cos(2*math.pi*np.dot(s,f))
                    sin_sum += math.sin(2*math.pi*np.dot(s,f))
                P[i][j][k]=(cos_sum**2 + sin_sum**2)/N
    return P

def DrawVerticalView(points, radius):
    '''
    绘制 俯视图
    :param points: poisson采样出的点
    :param radius: 采样半径
    :param length: 泊松采样的区域长
    :param width: 宽
    :return: 直接绘制图片
    '''
    N = len(points)  # 采样点数
    # 黑圈的半径
    black_radius = 1 / radius
    #设置显示范围
    scale = 4 * black_radius
    # 设置采样强度
    Density_x = 200
    Density_y = 200
    # 初始化为0
    P = np.zeros((Density_x, Density_y))
    # 计算功率值
    for i in range(Density_x):
        for j in range(Density_y):
            # 计算每个频谱上采样点的f坐标值
            f_x = -2 * black_radius + scale / Density_x * i
            f_y = -2 * black_radius + scale / Density_y * j
            # 设置扰动，去掉中间的十字架
            bias = random.uniform(0, scale / max(Density_x, Density_y))
            # 计算得到频谱图中的f值
            f = np.array([f_x + bias, f_y - bias, bias])
            cos_sum, sin_sum = 0, 0
            for l in range(N):
                s = points[l].vectorize()
                cos_sum += math.cos(2 * math.pi * np.dot(s, f))
                sin_sum += math.sin(2 * math.pi * np.dot(s, f))
            P[i][j] = (cos_sum ** 2 + sin_sum ** 2) / N
    #绘制
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i][j]>=1:
                P[i][j]=255
    #将矩阵mat_z以image的形式显示出来，双线性插值，灰度图，原点在下方，坐标范围定义为extent
    plt.imshow(P, interpolation='bilinear', cmap=matplotlib.cm.gray, origin='lower')
    plt.show()

if __name__ == '__main__':
    radius = 1
    length = 10
    width = 10
    height = 10
    k = 30
    #PowerSpectrum3D(radius, length, width, height)
    points = PoissonSampling3D(radius, length, width, height, k)
    DrawVerticalView(points, radius)
