Note:
this darknet is for predicting (to train yolov3-tiny/yolov3 model please use origin darknet http://pjreddie.com/darknet)

changes from origin darkent:
1.darknet for test, add present_list (change src/yolov.c line 335)
2.use cv2 read image
3.one box only predict one class (use top1 in one box)

use:
1. sh test.sh #for detect local image with xml
2. python2.7 detect_all_img.py # set detect_method, for detect camera or local video or one local folder image
3. python save_detected_img.py  # save detected box img





![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
