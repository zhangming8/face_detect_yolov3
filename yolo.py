# coding:utf-8
import sys
from ctypes import *
import random
import cv2
import copy
import numpy as np


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("row", c_int),
                ("col", c_int),
                ("mask_num", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/lishundong/Desktop/cabinet_laika_v7/ai/cabinet-darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

float_to_image = lib.float_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

float_to_image = lib.float_transfer_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE


# use cv2.imread; one box predict one class
def detect(im, thresh=0.5, hier_thresh=.5, nms=.45):
    #import time
    #start_time = time.time()
    #im = cv2.imread(image)
    num = c_int(0)
    pnum = pointer(num)
    im = im.astype('float32')

    data = cast(im.ctypes.data, POINTER(c_float))
    im = float_to_image(im.shape[1], im.shape[0], 3, data)
    predict_image(net, im)
 
    dets = get_network_boxes(net, im.w, im.h,thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]

    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        cla_prob = []
        for ii in range(meta.classes):
            cla_prob.append(dets[j].prob[ii])
        if sum(cla_prob) != 0:
            i = np.argmax(cla_prob)
            b = dets[j].bbox
            x = b.x
            y = b.y
            w = b.w
            h = b.h

            left = int(x - w / 2)
            right = int(x + w / 2)
            top = int(y - h/2)
            bot = int(y + h/2)
            if py_version == 3:
                                                                   # (  x1,  y1,    x2,  y2)
                res.append((meta.names[i].decode('utf-8'), dets[j].prob[i], (left, top, right, bot)))
            else:
                res.append((meta.names[i], dets[j].prob[i], (left, top, right, bot)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)

    return res


def init_yolo(cfg="cfg/yolov3-tiny.cfg", weights="yolov3-tiny_140000.weights",
              voc_data="cfg/voc.data"):
    global net, meta, py_version
    #set_gpu(0)
    py_version = sys.version_info[0]
    if py_version == 3:
        print("using python3")
        cfg = cfg.encode("utf-8")
        weights = weights.encode("utf-8")
        voc_data = voc_data.encode("utf-8")
    else:
        print("using python2")
    net = load_net(cfg, weights, 0)
    meta = load_meta(voc_data)
    print("init yolov3 done...")


#init_yolo()


if __name__ == '__main__':
    import glob, os
    img_list = glob.glob("/media/lishundong/DATA2/Nobody/notpeople/labeled_notpeople/Test/*.jpg")
    for im in img_list[:2]:
        img = cv2.imread(im)
        result = detect(img)
        for res in result:
            cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]
            cv2.putText(img, str(cls) +": "+str(prob)[:5], (x1,y1), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0))
        print(im, result)
        cv2.imshow(os.path.basename(im), img)
    cv2.waitKey(0)
