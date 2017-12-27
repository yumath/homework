import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, Point):
        return math.sqrt((Point.x - self.x)**2 + (Point.y - self.y)**2)

    def vectorize(self):
        return np.array([self.x, self.y])

def PoissonSampling2D(radius, length, width, n_k):
    grid_len = radius/math.sqrt(2)    #每个小方块的长度
    _len, _wid= math.ceil(length/grid_len), math.ceil(width/grid_len)#将空间分割为方块
    grids = -np.ones((_len, _wid))

    # 维持活跃点
    active_list = []
    #打出来的所有点存在points里
    points = []

    # 随机产生一个点作为起点
    x0,y0=random.uniform(-length/2, length/2), random.uniform(-width/2, width/2)
    p0 = Point(x0, y0)
    #将点放入active_list
    active_list.append(p0)
    points.append(p0)
    #计算第0个点所在格子的索引
    x0_index = math.floor((x0+length/2)/grid_len)
    y0_index = math.floor((y0+width/2)/grid_len)
    # 将该点所在的格子标记为其索引
    grids[x0_index][y0_index] = 0

    while len(active_list)>0:
        index = random.randint(0, len(active_list)-1)   #从active_list中随机选取一个点画圆环
        cur_point = active_list[index]  #取出该点

        is_success_paint_dot = False
        #随机打k个点
        for i in range(n_k):
            tmp_radius = random.uniform(radius, 2*radius)
            tmp_phi = random.uniform(0, 2*math.pi)
            #把球坐标转化为直角坐标
            tmp_x = tmp_radius * math.cos(tmp_phi)
            tmp_y = tmp_radius * math.sin(tmp_phi)
            #按照上面随机产生的坐标打点
            tmp_point = Point(cur_point.x+tmp_x, cur_point.y+tmp_y)
            if tmp_point.x>length/2 or tmp_point.x<-length/2 or tmp_point.y>width/2 or tmp_point.y<-width/2:
                continue
            #找到与打出来的该点相交的方块
            x_index = math.floor((tmp_point.x + length/2) /grid_len)
            y_index = math.floor((tmp_point.y + width /2) /grid_len)
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

            x_low = y_low = z_low = -2
            if x_index == 0:
                x_low=0
            if x_index == 1:
                x_low=-1
            if y_index == 0:
                y_low=0
            if y_index == 1:
                y_low=-1
            #判断随机生成的点周围的方框内有没有交点
            is_intersect = False
            for j in range(x_low, x_up):
                if is_intersect == True:    #相交了就退出当前循环
                    break
                for k in range(y_low, y_up):
                    if is_intersect == True:    #相交就退出当前循环
                        break
                    for l in range(z_low, z_up):
                        if grids[x_index+j][y_index+k] == -1:
                            continue
                        else:
                            if tmp_point.distance(points[int(grids[x_index+j][y_index+k])]) <= radius:
                                is_intersect = True
                                break
            if is_intersect == False:   #如果当前随机得到的点不和周围格子内的点相交，表示打点成功
                grids[x_index][y_index] = len(points)  #把该点的索引给grids
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

def PowerSpectrum2D(points, radius):
    N = len(points) #采样点数
    #黑圈的半径
    black_radius = 1/radius
    scale = 4*black_radius
    #设置采样强度
    Density_x = 300
    Density_y = 300
    #初始化为0
    P = np.zeros((Density_x, Density_y))
    # 计算功率值
    for i in range(Density_x):
        for j in range(Density_y):
            #计算每个频谱上采样点的f坐标值
            f_x = -2*black_radius+scale/Density_x*i
            f_y = -2*black_radius+scale/Density_y*j
            #设置扰动，去掉中间的十字架
            bias = random.uniform(0, scale/max(Density_x,Density_y))
            #计算得到频谱图中的f值
            f = np.array([f_x+bias,f_y-bias])
            cos_sum, sin_sum = 0, 0
            for l in range(N):
                s = points[l].vectorize()
                cos_sum += math.cos(2*math.pi*np.dot(s,f))
                sin_sum += math.sin(2*math.pi*np.dot(s,f))
            P[i][j]=(cos_sum**2 + sin_sum**2)/N
    return P

if __name__ == '__main__':
    length = 20
    width = 20
    r = 1
    '''
    points = TwoDimPoissonSampling(r, length, width, 30)

    x, y = [], []
    for i in range(len(points)):
        x.append(points[i].x)
        y.append(points[i].y)
    # 将数据点分成三部分画，在颜色上有区分度
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, c='r', marker='8')
    plt.show()
    '''
    points = PoissonSampling2D(radius=1, length=20, width=20, n_k=30)
    Power = PowerSpectrum2D(points, r)
    #img = Image.fromarray(Power)
    #img.convert('L')
    #print(Power)
    #img = Image.new("L", (Power.shape[0], Power.shape[1]))
    #for i in range(Power.shape[0]):
    #    for j in range(Power.shape[1]):
    #        img.putpixel((i,j), int(Power[i][j]))   #putpixel的第二个参数必须是整数

    for i in range(Power.shape[0]):
        for j in range(Power.shape[1]):
            if Power[i][j]>=1:
                Power[i][j]=255
    #将矩阵mat_z以image的形式显示出来，双线性插值，灰度图，原点在下方，坐标范围定义为extent
    plt.imshow(Power, interpolation='bilinear', cmap=matplotlib.cm.gray, origin='lower')
    plt.show()
    #img.show()
