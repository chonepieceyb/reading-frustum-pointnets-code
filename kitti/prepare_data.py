''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
'''
跑完代码之后，的主要作用就是对kitti的数据机型预处理，其中在这个文件所在的文件夹中的kitti_object将数据都定义成了一个类
而一些操作放在了 kitti_util里了，比如一些旋转等操作 -Y

numpy的官方文档 https://docs.scipy.org/doc/numpy/reference/index.html
'''
from __future__ import print_function

import os
import sys
import numpy as np   #numpy是python的一个扩展的开源计算库，有很多使用的计算，比如矩阵乘法，需要好好研究一下 -Y
import cv2           #cv2是opencv -Y
from PIL import Image  #PIL：Python Imaging Library，已经是Python平台事实上的图像处理标准库了，但是PIL主要是在python2中的，python3可能不兼容，关于PIL可以看看廖雪峰的python2教程，在python3中用Pillow代替，是一个图像处理库-Y
BASE_DIR = os.path.dirname(os.path.abspath(__file__))    #os.path.abspath(__file__)获取当前脚本的路径就是prepare_data.py的路径(包含.py)，而os.path.dirname()是获取父目录（路径里应该不包含.py这个文件）-Y
ROOT_DIR = os.path.dirname(BASE_DIR)      #获取根目录，这里应该是指到 /frustum-pointnets 这一级，也就是项目的根目录-Y
sys.path.append(BASE_DIR) #sys.path返回的是一个列表，加上append方法就是把BASE_DIR加到这个列表中  对于模块和自己写的脚本不在同一个目录下，常在脚本开头加sys.path.append('xxx')：，只在运行时生效-Y
sys.path.append(os.path.join(ROOT_DIR, 'mayavi')) #os.path.join python的路径拼接函数，注意，这个路径拼接稍微有些复杂，要用的时候再看看 -Y
import kitti_util as utils      #kitti_util 自己定义的，里面装着很多操作 -Y
import cPickle as pickle   #CPickle是 python2的库 python3中是pikle，在跑程序的时候数据处理完毕，会在 kitti文件夹生成3个.pikle文件,
                           #pikkle是一个数据储存的模块,在机器学习中常常需要把训练好的模型储存起来，pikle 就能起到这个作用z` -Y
from kitti_object import *   #kitti_object是作者自己定义的，把整个数据集看成一个对象，kitti_object里面有数据规模的参数，还有很多读取文本照片的操作 -Y
import argparse        # argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数。它的使用也比较简单。 -Yang

'''
  Scipy是一个用于数学、科学、工程领域的常用软件包，可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题。它用于有效计算Numpy矩阵，使Numpy和Scipy协同工作，高效解决问题。
  Scipy是由针对特定任务的子模块组成,scripy.spatial就是其中一个子模块,应用领域是空间数据结构极其算法,可以参考https://www.jianshu.com/p/6c742912047f -Y
'''
def in_hull(p, hull):
    from scipy.spatial import Delaunay    #Delaunay 三角剖分算法，对数值分析（比如有限元分析）以及图形学来说，都是极为重要的一项预处理技术。
    if not isinstance(hull,Delaunay):     
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0  
    '''
     hull.find_simplex(p)
     作用 Find the simplices containing the given points.
    官方文档https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.find_simplex.html -Y
    
    经过翻阅官方文档的例子这段代码的意思是：
    1 如果 hull 不是一个Delaunay三角剖分网络（关于什么是Delaunay查百度百科）那么就通过点组hull(这里的hull应该是一个点组
    eg:((1,2),(3,4),(5,6))  )构建一个Delaunay三角剖分网络 ，这时候 hull是一个网络了
    2return 语句的含义，判断 点集p是否在这个三角剖分网中的一个三角形里 ，p应该是一个 numpy.array的数据结构，返回的也是一个numpy.array的数据结构 ，eg array([ True,  True,  True])
     说明： Delaunay.simplexs()方法 返回的形式类似于
     >>> tri.simplices
         array([[2, 3, 0],                 # may vary
         [3, 1, 0]], dtype=int32)
         其中 [2,3,0]表示 三角形的点 分别是点组中的第 2，3，0个点 （从0开始算）
     而     Delaunay.find_simplex(p),就返回一个 1维数组，数组中的数 表示 p(p是点组或者一个点)中对应的点，在 trimplices中的第几个三角形中
    如果没有就返回 -1
    
    总结：这个函数的作用就是判断，点集p中的点哪些在Delaunay三角剖分网中，哪些不在返回一类似于[true,false,true] 的东西-Y
    详情见 https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.Delaunay.html

    '''

def extract_pc_in_box3d(pc, box3d):   #,作用应该是返回pc中，包含在有box3d经过变换构成的Delaunay三角剖分网中的点 -Y -Y
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    '''
    由上面的分析可知box3d_roi_inds是一个有bool值构成的数组，所以对于下面的pc[box3d_roi_inds,:]操作的解释，就是采用掩模+切片取值，具体的解释参见
    https://blog.csdn.net/liujian20150808/article/details/81273289   -Y
    '''
    return pc[box3d_roi_inds,:], box3d_roi_inds   #box3d_roi_inds是一个有bool值构成的array

def extract_pc_in_box2d(pc, box2d):        #作用应该是返回pc中，包含在有box2d经过变换构成的Delaunay三角剖分网中的点 -Y
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''

    '''
    np.zeros:zeros(shape, dtype=float, order='C')返回：返回来一个给定形状和类型的用0填充的数组；参数：shape:形状 dtype:数据类型，可选参数，默认numpy.float64  -Y
    这里的作用是返回一个（应该） 4行2列的矩阵，元素全是0
    https://blog.csdn.net/qq_36621927/article/details/79763585
    '''
    box2d_corners = np.zeros((4,2))           
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds        #box2d_roi_inds是一个有bool值构成的array
     
def demo():        # 跑了代码之后，这是一个完整描述，这个网络的数据预处理过程的代码，先跳过了 -Y
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    #show_lidar_with_boxes(pc_velo, objects, calib)
    #raw_input()
    show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    print(' -------- LiDAR points projected to image plane --------')
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height) 
    raw_input()
    
    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()
    
    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin,ymin,xmax,ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

def random_shift_box2d(box2d, shift_ratio=0.1):      
    ''' Randomly shift box center, randomly scale width and height    
    '''
    #随意地改变box的中心，任意地缩放宽度和高度
    r = shift_ratio               #  r 应该是控制随机变换的范围，即 shift_ratio -Y
    xmin,ymin,xmax,ymax = box2d   #应该是2D图像框的 4个角的坐标     -Y                                                                                                              
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)     #np.random.random()：Return random floats in the half-open interval [0.0, 1.0). 有一个参数size,如果size=(2,3)就生成一个 2*3 的随机矩阵关于numpy库的随机函数详见 https://blog.csdn.net/kancy110/article/details/69665164 -Y
    cy2 = cy + h*r*(np.random.random()*2-1)    
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1  #这里的0.9大盘1.1是h乘的系数    
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1 
    #经过上面的操作，就由原来的box2d产生了一个随机的宽高经过变化的，中心改变的一个心得box,改变的程度由 shift_ratio决定 -Y
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
    #np.array()是 array是 numpy的基本的数据结构之一,这里返回了一个一维数组, 详情见官方文档  https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html?highlight=array#numpy.array -Y
 
def extract_frustum_data(idx_filename, split, output_filename, viz=False,
                       perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
    ''' Extract point clouds and corresponding annotations in frustums      #提取由2d框得到的三维立体视锥中的点云数据 ？-Y
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system      
        (as that in 3d box label files)
        
    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb（扰动？） the box2d
            (used for data augmentation(增广？) in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'), split)     #根据路径来建立一个kitti_object对象
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]    #string.retrip(char) 去掉字符串末尾的字符char（末尾有几个char就去除几个char)默认为空格
                                                                           #这句话的作用应该是把所有文件的编号放到一个列表里 eg['000000','000001',....],然后下面的操作都是根据文件的编号（就是kitti文件的文件名）来取出相应的文件
    #关于kitti数据集的坐标系的含义，以及坐标系的转换的公式的问题 可以参考 https://blog.csdn.net/KYJL888/article/details/82844823 -Y
    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord   ps:rect camera coord 应该是指参考相机坐标系，即PO（编号为0的相机）的坐标系-Y
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord   #把空间中三个向量扩充成了四个向量-Y
    label_list = [] # 1 for roi object, 0 for clutter        ps:1是指 roi(rigon of interest)区域，0是指干扰区域 -Y
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of          
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis      

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:       #
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix       #根据kitti_units和kitti_object判断这个操作是获取相应的data_idx对应的图片的calib文件，包括几个投影矩阵和相机的内外参是一个类，具体的内容见 kitti_unitl的line81和kitti_object的line60
        objects = dataset.get_label_objects(data_idx)                   #返回识别出来的物体，objects应该是一个list，里面存放这每张图片标注的物品的信息 -Y
        pc_velo = dataset.get_lidar(data_idx)          #获取对应图片的点云数据，格式是一个N*4的numpy array，以np.float32格式储存 -Y
        pc_rect = np.zeros_like(pc_velo)               #返回一个和pc_velo形状和数据类型相同的array -Y
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])    #将点从雷达坐标系投影到rect坐标系，输入是n*4输出是n*3 -Y
        pc_rect[:,3] = pc_velo[:,3]                  #最后一列（第四列）均赋值为1
        img = dataset.get_image(data_idx)             #获取图片,具体代码在 kitti_util的273行，用opencv的imread方法读取一张图片（应该还是彩色图片） -Y
        img_height, img_width, img_channel = img.shape               #img.shape返回 宽，高，颜色通道数 -Y
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],   # _是经过筛选的在能够投影到2D图像框上的点集，在雷达坐标系下 n*3，具体见函数-Y
            calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):       #object是单张图片里标定出来的物品集 eg car -Y
            if objects[obj_idx].type not in type_whitelist :continue        #如果不是所要识别的对象 -Y

            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d                      #获取box2d
            for _ in range(augmentX):                           #  _ 是啥？ augmentX是啥？
                # Augment data by box2d perturbation
                if perturb_box2d:                                       #对2D图像上物体的框进行随机扰动 -Y
                    xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                    print(box2d)                                             
                    print(xmin,ymin,xmax,ymax)
                else:
                    xmin,ymin,xmax,ymax = box2d
                #pc_image_coord, 所有的点云数据投影到2D图像坐标系下的点集,下面这段话是要进行过滤，让点云数据只能投影到识别出的物体的 2d_box内
                box_fov_inds = (pc_image_coord[:,0]<xmax) & \      
                    (pc_image_coord[:,0]>=xmin) & \
                    (pc_image_coord[:,1]<ymax) & \
                    (pc_image_coord[:,1]>=ymin)
                box_fov_inds = box_fov_inds & img_fov_inds       #将两个掩模进行与操作，     
                pc_in_box_fov = pc_rect[box_fov_inds,:]          #用掩模切片，取出在rect坐标系下，能够投影到所识别物体的 2d_box内的点，到这里应该也就初步完成了点云的过滤
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])          #计算2d_box的中心
                uvdepth = np.zeros((1,3))
                uvdepth[0,0:2] = box2d_center
                uvdepth[0,2] = 20 # some random depth                               #设置一个随机的值，现在uvdepth应该是一个三维的坐标，其中x,y是box的center而第三个值是图像坐标系下的一个随机的z
                box2d_center_rect = calib.project_image_to_rect(uvdepth)            #投影到 rect坐标系
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],         #box2d_center_rect(x,y,z)    这里计算z/x的反三角
                    box2d_center_rect[0,0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)    #经过计算， box3d_pts_2d(数据集给的3d框投影到平面图像image上的点的集合 8*2)   box3d_pts_3d（数据集给的3d框在rect坐标系下的点集)
                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)     #这个函数计算出，在rect坐标系下 过滤出pc_in_box_fov(n*3)中在 由box3d_pts_3d(8个corner)中的8个点构成的三角剖分网中的点并赋值给 _,将过滤得到的掩模赋值给inds
                label = np.zeros((pc_in_box_fov.shape[0]))               #构造出一个 n*1向量，（n的个数就是之前经过2d_box提取出的，在rect坐标系下能够投影到 2d_box点的数目）
                label[inds] = 1                                         #构造初一个label，pc_in_box_fov中在3d_box内的点都标记为1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax-ymin<25 or np.sum(label)==0:  
                    continue
               #下面是训练数据的排列方式？ -Y
                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
    
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]
        
    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
    
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i] 
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type=='DontCare':continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.type) 
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list

 
def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       viz=False,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    cache_id = -1
    cache = None
    
    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        print('det idx: %d/%d, data idx: %d' % \
            (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
            pc_rect[:,3] = pc_velo[:,3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(\
                pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib,pc_rect,pc_image_coord,img_fov_inds]
            cache_id = data_idx
        else:
            calib,pc_rect,pc_image_coord,img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected 
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & \
            (pc_image_coord[:,0]>=xmin) & \
            (pc_image_coord[:,1]<ymax) & \
            (pc_image_coord[:,1]>=ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
            box2d_center_rect[0,0])
        
        # Pass objects that are too small
        if ymax-ymin<img_height_threshold or \
            len(pc_in_box_fov)<lidar_point_threshold:
            continue
       
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)
    
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format. 
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {} 
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    args = parser.parse_args()

    if args.demo:
        demo()
        exit()

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'train.pickle'), 
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)

    if args.gen_val:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val.pickle'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(\
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=type_whitelist) 
