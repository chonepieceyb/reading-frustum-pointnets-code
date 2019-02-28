小组用来阅读和研究frustum-pointnets代码的仓库。
在阅读代码的同时请在旁边写上注释，并且在每一句注释之后加上 +姓的首字母，以便区分
---------------------
中文不一定准确，仅供参考(*^▽^*)

## Frustum PointNets for 3D Object Detection from RGB-D Data
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://www.cs.unc.edu/~wliu/" target="_black">Wei Liu</a>, <a href="http://www.cs.cornell.edu/~chenxiawu/" target="_blank">Chenxia Wu</a>, <a href="http://cseweb.ucsd.edu/~haosu/" target="_blank">Hao Su</a> and <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from <a href="http://www.stanford.edu" target="_blank">Stanford University</a> and <a href="http://nuro.ai" target="_blank">Nuro Inc.</a>

![teaser](https://github.com/charlesq34/frustum-pointnets/blob/master/doc/teaser.jpg)

## Introduction
This repository is code release for our CVPR 2018 paper (arXiv report [here](https://arxiv.org/abs/1711.08488)). In this work, we study 3D object detection from RGB-D data. We propose a novel detection pipeline that combines both mature 2D object detectors and the state-of-the-art 3D deep learning techniques. In our pipeline, we firstly build object proposals with a 2D detector running on RGB images, where each 2D bounding box defines a 3D frustum region. Then based on 3D point clouds in those frustum regions, we achieve 3D instance segmentation and amodal 3D bounding box estimation, using PointNet/PointNet++ networks (see references at bottom).

这个存储库是CVPR 2018论文(这里是arXiv报告)的代码版本。本文研究了基于RGB-D数据的三维物体检测方法。我们提出了一种结合了成熟的二维物体探测器和最先进的三维深度学习技术的新型检测管道。在我们的管道中，我们首先使用运行在RGB图像上的2D检测器构建对象提案，其中每个2D边框定义一个3D frustum区域。然后基于这些截锥体区域的三维点云，利用PointNet/PointNet++网络实现三维实例分割和amodal三维边界盒估计(见参考文献底部)。

By leveraging 2D object detectors, we greatly reduce 3D search space for object localization. The high resolution and rich texture information in images also enable high recalls for smaller objects like pedestrians or cyclists that are harder to localize by point clouds only. By adopting PointNet architectures, we are able to directly work on 3D point clouds, without the necessity to voxelize them to grids or to project them to image planes. Since we directly work on point clouds, we are able to fully respect and exploit the 3D geometry -- one example is the series of coordinate normalizations we apply, which help canocalizes the learning problem. Evaluated on KITTI and SUNRGBD benchmarks, our system significantly outperforms previous state of the art and is still in leading positions on current <a href="http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d">KITTI leaderboard</a>.

通过利用2D对象检测器，我们大大减少了用于对象定位的3D搜索空间。图像的高分辨率和丰富的纹理信息也使得像行人或骑自行车这样的小物体具有高的回忆能力，这些小物体仅靠点云很难定位。通过采用PointNet体系结构，我们可以直接在3D点云上工作，而不需要将它们体化为网格或投影到图像平面。因为我们直接在点云上工作，所以我们能够充分尊重和利用三维几何——一个例子是我们应用的一系列坐标标准化，这有助于解决学习问题。根据KITTI和SUNRGBD基准进行评估，我们的系统明显优于以前的技术水平，目前仍处于领先地位

For more details of our architecture, please refer to our paper or <a href="http://stanford.edu/~rqi/frustum-pointnets" target="_blank">project website</a>.

有关我们的架构的更多细节，请参阅我们的论文或<a href="http://stanford.edu/~rqi/frustum-pointnets" target="_blank">project website</a>.

## Citation
If you find our work useful in your research, please consider citing:

        @article{qi2017frustum,
          title={Frustum PointNets for 3D Object Detection from RGB-D Data},
          author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1711.08488},
          year={2017}
        }

## Installation
Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>.There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`, `mayavi`  etc. It's highly recommended that you have access to GPUs.

安装TensorFlow.对于一些用于数据处理和可视化的Python库，如' cv2 '、' mayavi '等，也存在一些依赖关系。强烈建议您访问gpu。

To use the Frustum PointNets v2 model, we need access to a few custom Tensorflow operators from PointNet++. The TF operators are included under `models/tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The compile script is written for TF1.4. There is also an option for TF1.2 in the script. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

要使用Frustum PointNets v2模型，我们需要访问几个来自pointnet++的自定义Tensorflow操作符。TF操作符包含在“models/tf_ops”下，您需要编译它们(请检查“tf_xxx_compile”)。sh ' under each ops子文件夹)first。如果需要，更新“nvcc”和“python”路径。编译脚本是为TF1.4编写的。脚本中还有一个用于TF1.2的选项。如果您使用的是早期版本，您可能需要在g++命令中删除' -D_GLIBCXX_USE_CXX11_ABI=0 '标志，以便正确编译。

If we want to evaluate 3D object detection AP (average precision), we need also to compile the evaluation code (by running `compile.sh` under `train/kitti_eval`). Check `train/kitti_eval/README.md` for details.

如果我们想评估3D对象检测AP(平均精度)，我们还需要编译评估代码(通过运行'compile.sh'下的`train/kitti_eval`)。检查`train/kitti_eval/README.md`的细节。

Some of the demos require `mayavi` library. We have provided a convenient script to install `mayavi` package in Python, a handy package for 3D point cloud visualization. You can check it at `mayavi/mayavi_install.sh`. If the installation succeeds, you should be able to run `mayavi/test_drawline.py` as a simple demo. Note: the library works for local machines and seems do not support remote access with `ssh` or `ssh -X`.

一些演示需要“mayavi”库。我们提供了一个方便的脚本，可以在Python中安装“mayavi”包，这是一个方便的3D点云可视化包。您可以在“mayavi/mayavi_install.sh”处检查它。如果安装成功，您应该能够运行“mayavi/test_drawline”。py '作为一个简单的演示。注意:该库适用于本地机器，似乎不支持使用“ssh”或“ssh -X”进行远程访问。

The code is tested under TF1.2 and TF1.4 (GPU version) and Python 2.7 (version 3 should also work) on Ubuntu 14.04 and Ubuntu 16.04 with NVIDIA GTX 1080 GPU. It is highly recommended to have GPUs on your machine and it is required to have at least 8GB available CPU memory.

该代码在使用NVIDIA GTX 1080 GPU的ubuntu14.04和ubuntu16.04上的TF1.2和TF1.4 (GPU版本)以及Python 2.7(版本3也可以)下测试。强烈建议在您的机器上使用gpu，并且要求至少有8GB可用CPU内存。

## Usage

Currently, we support training and testing of the Frustum PointNets models as well as evaluating 3D object detection results based on precomputed 2D detector outputs (under `kitti/rgb_detections`). You are welcomed to extend the code base to support your own 2D detectors or feed your own data for network training.

目前，我们支持Frustum PointNets模型的训练和测试，以及基于预先计算的2D检测器输出(在“kitti/rgb_detections”下)评估3D对象检测结果。欢迎您扩展代码库以支持您自己的2D检测器，或者为网络培训提供您自己的数据。

### Prepare Training Data
In this step we convert original KITTI data to organized formats for training our Frustum PointNets. <b>NEW:</b> You can also directly download the prepared data files <a href="https://shapenet.cs.stanford.edu/media/frustum_data.zip" target="_blank">HERE (960MB)</a> -- to support training and evaluation, just unzip the file and move the `*.pickle` files to the `kitti` folder.

在这一步中，我们将原始的KITTI数据转换为有组织的格式，以训练我们的Frustum切入点。NEW:你也可以直接下载准备好的数据文件HERE (960MB)——要支持培训和评估，只需解压文件并移动' *。将“文件”pickle到“kitti”文件夹。

Firstly, you need to download the <a href="http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d" target="_blank">KITTI 3D object detection dataset</a>, including left color images, Velodyne point clouds, camera calibration matrices, and training labels. Make sure the KITTI data is organized as required in `dataset/README.md`. You can run `python kitti/kitti_object.py` to see whether data is downloaded and stored properly. If everything is fine, you should see image and 3D point cloud visualizations of the data. 

首先，您需要下载KITTI 3d物体检测数据集，包括左侧彩色图像、Velodyne点云、相机校准矩阵和训练标签。确保KITTI数据按照“dataset/README.md”中的要求组织。你可以运行python kitti/kitti_object。查看数据是否被正确下载和存储。如果一切正常，您应该看到数据的图像和3D点云可视化。

Then to prepare the data, simply run: (warning: this step will generate around 4.7GB data as pickle files)

然后，要准备数据，只需运行:(警告:此步骤将生成4.7GB左右的数据作为pickle文件)

    sh scripts/command_prep_data.sh

Basically, during this process, we are extracting frustum point clouds along with ground truth labels from the original KITTI data, based on both ground truth 2D bounding boxes and boxes from a 2D object detector. We will do the extraction for the train (`kitti/image_sets/train.txt`) and validation set (`kitti/image_sets/val.txt`) using ground truth 2D boxes, and also extract data from validation set with predicted 2D boxes (`kitti/rgb_detections/rgb_detection_val.txt`).

基本上，在这个过程中，我们从原始的KITTI数据中提取了基于ground truth 2D边界框和2D对象检测器的frustum点云以及ground truth标签。我们将使用ground truth 2D box对列车(' kitti/image_sets/train.txt ')和验证集(' kitti/image_sets/ var .txt ')进行提取，并使用预测的2D box (' kitti/rgb_detections/ rgb_detection_var .txt ')从验证集提取数据。

You can check `kitti/prepare_data.py` for more details, and run `python kitti/prepare_data.py --demo` to visualize the steps in data preparation.

你可以检查“kitti/prepare_data”。有关详细信息，请运行“python kitti/prepare_data.py—demo”来可视化数据准备中的步骤。

After the command executes, you should see three newly generated data files under the `kitti` folder. You can run `python train/provider.py` to visualize the training data (frustum point clouds and 3D bounding box labels, in rect camera coordinate).

在执行该命令之后，您应该会在“kitti”文件夹下看到三个新生成的数据文件。您可以运行“python火车/提供商”。可视化训练数据(圆锥点云和3D边框标签，在矩形相机坐标)。

### Training Frustum PointNets

To start training (on GPU 0) the Frustum PointNets model, just run the following script:

要开始训练(在GPU 0上)Frustum PointNets模型，只需运行以下脚本:

CUDA_VISIBLE_DEVICES=0 sh scripts/command_train_v1.sh

You can run `scripts/command_train_v2.sh` to trian the v2 model as well. The training statiscs and checkpoints will be stored at `train/log_v1` (or `train/log_v2` if it is a v2 model). Run `python train/train.py -h` to see more options of training. 

您可以运行' scripts/command_train_v2。sh ' to trian the v2 model以及。训练静态数据和检查点将存储在“train/log_v1”(如果是v2模型，则存储在“train/log_v2”)中。运行“python train/train.py -h”查看更多培训选项。

<b>NEW:</b> We have also prepared some pretrained snapshots for both the v1 and v2 models. You can find them <a href="https://shapenet.cs.stanford.edu/media/frustum_pointnets_snapshots.zip" target="_blank">HERE (40MB)</a> -- to support evaluation script, you just need to unzip the file and move the `log_*` folders to the `train` folder.

我们还为v1和v2模型准备了一些经过预处理的快照。您可以在这里找到它们 (40MB)—要支持评估脚本，只需解压缩文件并将' log_* '文件夹移动到' train '文件夹。

### Evaluation
To evaluate a trained model (assuming you already finished the previous training step) on the validation set, just run:
要在验证集上评估一个经过训练的模型(假设您已经完成了前面的训练步骤)，只需运行:
    CUDA_VISIBLE_DEVICES=0 sh scripts/command_test_v1.sh

Similarly, you can run `scripts/command_test_v2.sh` to evaluate a trained v2 model. The script will automatically evaluate the Frustum PointNets on the validation set based on precomputed 2D bounding boxes from a 2D detector (not released here), and then run the KITTI offline evaluation scripts to compute precision recall and calcuate average precisions for 2D detection, bird's eye view detection and 3D detection.

类似地，您可以运行' scripts/command_test_v2。sh '来评估一个训练过的v2模型。脚本将自动评估验证集的截头PointNets根据预先计算的二维边界框从2 d检测器(这里没有公布),然后运行脚本KITTI离线评估计算精度回忆和calcuate平均精度为2 d检测,鸟瞰检测和三维检测。

Currently there is no script for evaluation on test set, yet it is possible to do it by yourself. To evaluate on the test set, you need to get outputs from a 2D detector on KITTI test set, store it as something in `kitti/rgb_detections`. Then, you need to prepare test set frustum point clouds for the test set, by modifying the code in `kitti/prepare_data.py`. Then you can modify test scripts in `scripts` by changing the data path, idx path and output file name. For our test set results reported, we used the entire `trainval` set for training.

目前还没有对测试集进行评估的脚本，但是您可以自己进行评估。要在测试集上求值，您需要从KITTI测试集上的2D检测器获得输出，并将其存储在“KITTI /rgb_detections”中。然后，您需要通过修改“kitti/prepare_data.py”中的代码，为测试集准备测试集frustum点云。然后，您可以通过更改数据路径、idx路径和输出文件名来在“scripts”中修改测试脚本。对于报告的测试集结果，我们使用整个“trainval”集进行培训。

## License
Our code is released under the Apache 2.0 license (see LICENSE file for details).

## References
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data: <a href="https://github.com/charlesq34/pointnet">here</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data: <a href="https://github.com/charlesq34/pointnet2">here</a>.

### Todo

- Add a demo script to run inference of Frustum PointNets based on raw input data.
- Add related scripts for SUNRGBD dataset

-添加一个演示脚本，以基于原始输入数据运行Frustum切入点的推断
-为SUNRGBD数据集添加相关脚本

