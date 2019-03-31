''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''

from __future__ import print_function

import pickle as pickle
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'models'))
from box_util import box3d_iou
from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


def rotate_pc_along_y(pc, rot_angle):              #将点集绕着 y轴旋转
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ          这个坐标系和参考相机坐标系差不多
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.    把连续的角度转化成离散的分类？

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at       把 2pi等分成N份，根据输入进行归类
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1                           剩余角，扣除分类后的角度，与余数的概念类似  -Y
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle%(2*np.pi)     #限制到 2pi范围内
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - \
        (class_id * angle_per_class + angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):    #根据分类把角度的class 转化成角度，上面那个函数的反函数
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle
        
def size2class(size, type_name):                
    ''' Convert 3D bounding box size to template class and residuals.   #将3D框框的大小转化成模板类和残差
    todo (rqi): support multiple size clusters per type.
 
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]      #由于python2和python3的字典取key的差异，会报错，
    size_residual = size - g_type_mean_size[type_name]   #根据model_util 每一个类别对应一个class(用int 表示)size class 和 type class的序号应该是对应的，
    return size_class, size_residual                      #并且在model中给出了每一个类别3D框框的平均大小，这个和之前看的有一篇论文的思路类似 -Y

def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud        这个参数的含义是否随机翻动点云数据
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation         是否旋转视锥
            overwritten_data_path: string, specify pickled file path.        .pikle文件所在的地址
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have      这个含义应该是只有 2D检测的结果(测试的时候应该是只提供2D检测结果，因为按照我的理解，这个网络是基于2D图像检测），并没有3D框框的groundtrue（类似于KITTI数据集里的label文件）只返回元素 -Y
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector        这个one hot vector是啥？ 
        '''
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join(ROOT_DIR,
                'kitti/frustum_carpedcyc_%s.pickle'%(split))

        self.from_rgb_detection = from_rgb_detection
        if from_rgb_detection:    #从预处理得到的.pilke中提取，具体的预处理代码 在 kitti/prepare_data.py line469左右 -Y
            with open(overwritten_data_path,'rb') as fp:
                self.id_list = pickle.load(fp,encoding='iso-8859-1')
                self.box2d_list = pickle.load(fp,encoding='iso-8859-1')
                self.input_list = pickle.load(fp,encoding='iso-8859-1')
                self.type_list = pickle.load(fp,encoding='iso-8859-1')
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp,encoding='iso-8859-1') 
                self.prob_list = pickle.load(fp,encoding='iso-8859-1')
        else:                        #从预处理得到的.pilke中提取，具体的预处理代码 在 kitti/prepare_data.py line300左右 -Y
            with open(overwritten_data_path,'rb') as fp:
                self.id_list = pickle.load(fp,encoding='iso-8859-1')      #python3和python2的差异必须加上encoding='bytes'
                self.box2d_list = pickle.load(fp,encoding='iso-8859-1')
                self.box3d_list = pickle.load(fp,encoding='iso-8859-1')
                self.input_list = pickle.load(fp,encoding='iso-8859-1')
                self.label_list = pickle.load(fp,encoding='iso-8859-1')
                self.type_list = pickle.load(fp,encoding='iso-8859-1')
                self.heading_list = pickle.load(fp,encoding='iso-8859-1')
                self.size_list = pickle.load(fp,encoding='iso-8859-1')
                # frustum_angle is clockwise angle from positive x-axis   视锥的角度是绕着x轴的正方向顺时针来定义的
                self.frustum_angle_list = pickle.load(fp,encoding='iso-8859-1') 

    def __len__(self):
            return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index) #这个rot_angle在原来的角度上增加了 0.5pi

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:   #如果旋转到中心的话，把视锥点集旋转到中心
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        # Resample
        # np.random.choice的用法   https://blog.csdn.net/wyx100/article/details/80639653
        '''
            import numpy as np

            # 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
            a1 = np.random.choice(a=5, size=3, replace=False, p=None)
            print(a1)
            # 非一致的分布，会以多少的概率提出来
            a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
            print(a2)
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是
            True的话， 有可能会出现重复的，因为前面的抽的放回去了。
            --------------------- 
            作者：qfpkzheng 
            来源：CSDN 
            原文：https://blog.csdn.net/qfpkzheng/article/details/79061601 
            版权声明：本文为博主原创文章，转载请附上博文链接！
        '''
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)   #这里应该是在在point_set中按照一致分布放回地选出 npoints个点出来
        point_set = point_set[choice, :]   #

        if self.from_rgb_detection:
            if self.one_hot:
                return point_set, rot_angle, self.prob_list[index], one_hot_vec
            else:
                return point_set, rot_angle, self.prob_list[index]
        
        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]   #如果数据是from_rgb_dection的话，那label文件是没有的，所以到前面的if就return,现在是针对train的操作，对label的说明，根据prepare.py
        seg = seg[choice]              #label是一个 n*1向量，（n的个数就是之前经过2d_box提取出的，在rect坐标系下能够投影到 2d_box点的数目），在这n个值中，在由3D框框内8个点构成的三角剖分网中的点标记为1其余标记为0 -Y
                                       #这里从label从选取由上一部choice得到的点
        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        print(type(self))
        size_class, size_residual = size2class(list(self.size_list)[index], #size_class的数据是一个 int ,而 size_residual是 size - mean_size剩下的两，mean_size在 model_util中有定义 -Y
            list(self.type_list)[index])

        # Data Augmentation  数据增广？
        if self.random_flip:    #做左右翻转，如果反转了，那么旋转角度的数据就会不正确
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random()>0.5: # 50% chance flipping
                point_set[:,0] *= -1  #根据文件输入的说明，点应该都是在 参考相机（rect坐标系下的）（应该！） 也就是 z轴 向前，y轴向下，x轴向右，这里有百分之50的概率做左右翻转
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle #注：这个heading_angle的含义有点不明确，根据prepare.data应该是kitti label文件里的rotation_y
        if self.random_shift:  #对点集做先后扰动
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            #这里先计算了扰动的shift
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            point_set[:,2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,    #angle_class的类型是一个int, angle_residual也是angel扣掉 mean_angle后的值
            NUM_HEADING_BIN)

        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual,\
                size_class, size_residual, rot_angle, one_hot_vec
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual,\
                size_class, size_residual, rot_angle

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]    #为什么要多加一个 0.5pi? 目前我的猜测是方便计算  -Y

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return box3d_center

#  作者在kitti_util中给出了3D框框的点的定义
'''
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
'''
    def get_center_view_box3d_center(self, index):   #旋转中心
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0,:] + \
            self.box3d_list[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), \    
            self.get_center_view_rot_angle(index)).squeeze()  #而 numpy.array.expand_dims的作用是扩展维度比如 原本是 x=[1,2] np.expand_dims(x,0)之后 结果变成了 [[1,2]].这里应该是先扩充成二维的矩阵，方便用旋转矩阵进行旋转操作
                                                              # numpy.array.squeze()函数，删除数组中的单维度的条目（应该可以理解成降维，eg reshape([1,1,10])中的两个1 
                                                              #参考解释 https://blog.csdn.net/tracy_leaf/article/details/79297121
        
    def get_center_view_box3d(self, index):   #旋转8个cornor
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
            self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):     #旋转整个点的集合
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])   #这里应该是复制点集
        return rotate_pc_along_y(point_set, \
            self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residuals[i,heading_class[i]] \
        for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residuals[i,size_class[i],:] \
        for i in range(batch_size)])

    iou2d_list = [] 
    iou3d_list = [] 
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
            heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
            heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
            heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label) 
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res,\
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    ty += h/2.0
    return h,w,l,tx,ty,tz,ry

if __name__=='__main__':
    import mayavi.mlab as mlab 
    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d
    median_list = []
    dataset = FrustumDataset(1024, split='val',
        rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
            'angle_class: ', data[3], 'angle_res:', data[4], \
            'size_class: ', data[5], 'size_residual:', data[6], \
            'real_size:', g_type_mean_size[g_class2type[data[5]]]+data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:,0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5],data[6]), class2angle(data[3], data[4],12), data[2])

        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:,0], ps[:,1], ps[:,2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1,0,0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))
