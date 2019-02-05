#-*- coding:utf-8 -*-
import sys
import os, cv2, glob, shutil
from ctypes import *
import math
import random, json


from yolo import init_yolo
from yolo import detect


classes = ["face"]


def draw_text(img, result):
    cls, conf, x0, y0, x1, y1 = result[0], result[1], result[2][0], result[2][1], result[2][2], result[2][3]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(img, str(cls), (int(x0 + (x1 - x0) * 0.5), int(y0 + (y1 - y0) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体粗细
    cv2.putText(img, str(conf), (x0, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img


def get_imgpath(path, extend=".jpg"):
    img_list = []
    for fpath , dirs , fs in os.walk(path):
        for f in fs:
            img_path = os.path.join(fpath , f)
            if os.path.dirname(img_path) == os.getcwd():
                continue
            if not os.path.isfile(img_path):
                continue
            if os.path.basename(img_path)[-4:] == extend:
                img_list.append(img_path)
    return img_list


def mkdir(save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)


def detect_video_or_camera(cap, detect_method):
    index = 0
    while True:
        ret, im = cap.read()  # 读取一帧的图像
        if detect_method == 1:
            img = im
        elif detect_method == 2:
            img = im[:, :-580]  # get 710x720
        if 0xFF == ord('q') or type(img) == type(None):
            print("finish...")
            break
        results = detect(img, 0.5)
        index += 1
        if len(results) == 0:
            print("%d can't detect anything" % index)
           # cv2.imwrite(result_path + '/no_detect/' + str(index) + '.png', img)
        else:
            # plot predicted bounding box
            for result in results:
                print('{} detcted: {}'.format(index, result))
                img = draw_text(img, result)
            cv2.imwrite(save_path + '/'+ str(index) + '.jpg', img)
        cv2.imshow('img', img)
    cap.release()
    cv2.destroyAllWindows()


def detect_folder_img(img_list):
    for one_img_path in img_list:
        img = cv2.imread(one_img_path)
        results = detect(img, 0.5)
        print("----------------")
        if len(results) == 0:
            print("%s can't detect anything" % os.path.basename(one_img_path))
           # cv2.imwrite(result_path + '/no_detect/' + str(index) + '.png', img)
        else:
            # plot predicted bounding box
            for result in results:
                print('{} detcted: {}'.format(os.path.basename(one_img_path), result))
                img = draw_text(img, result)
            cv2.imwrite(save_path + '/'+ os.path.basename(one_img_path), img)


if __name__ == "__main__":
    init_yolo()

    save_path = "results"
    mkdir(save_path)
    
    detect_method = 3  # 1 detect local video, 2 detect camear 0 image, 3 detect one folder jpg image

    if detect_method == 1:
        viedo_name = "video/0.avi"
        cap = cv2.VideoCapture(viedo_name)

        detect_video_or_camera(cap, detect_method)
    elif detect_method == 2:
        camera = 0
        cap = cv2.VideoCapture(camera)
        cap.set(3, 1280)
        cap.set(4, 800)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        detect_video_or_camera(cap, detect_method)
    elif detect_method == 3:
        img_path = "."
        extend_name = ".jpg"
        img_list = get_imgpath(img_path, extend=".jpg")
        if len(img_list) == 0:
            print("cannot find image in folder %s" % img_path)
            os._exit(0)
        detect_folder_img(img_list)
    else:
        print("[INFO]: detect_method should be 1 or 2 or 3, you set {}".format(detect_method))
        os._exit(0)

