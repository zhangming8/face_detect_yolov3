github: https://github.com/zhangming8/face_detect_yolov3/tree/init
Note:
this darknet is for predicting (to train yolov3-tiny/yolov3 model please use origin darknet http://pjreddie.com/darknet)

changes from origin darkent:
1.darknet for test, add present_list (change src/yolov.c line 335)
2.use cv2 read image
3.one box only predict one class (use top1 in one box)

use:
python2.7 detect_all_img.py # set detect_method, for detect camera or local video or one local folder image

相关博客
https://blog.csdn.net/u010397980/article/details/86764637
人脸检测模型下载https://pan.baidu.com/s/109LU1GlCA-1o-l0CpZ0ZIw 提取码: mw5i





![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
