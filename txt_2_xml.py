#coding:utf-8
import os
import cv2

from xml_writer import PascalVocWriter 


#txt_label = "wider_face_split/wider_face_val_bbx_gt.txt"
#img_path = "WIDER_val/images"

txt_label = "wider_face_split/wider_face_train_bbx_gt.txt"
img_path = "WIDER_train/images"


def write_xml(img, res):
	img_dir = os.path.join(img_path, img)
	print img_dir
	img = cv2.imread(img_dir)
	shape = img.shape
	img_h, img_w = shape[0], shape[1]
	writer = PascalVocWriter("./", img_dir, (img_h, img_w, 3), localImgPath="./", usrname="wider_face")
	for r in res:
		r = r.strip().split(" ")[:4]
		print r
		x_min, y_min = int(r[0]), int(r[1])
		x_max, y_max = x_min + int(r[2]), y_min + int(r[3])
		writer.verified = True
		writer.addBndBox(x_min, y_min, x_max, y_max, 'face', 0)
		writer.save(targetFile = img_dir[:-4] + '.xml')


with open(txt_label, "r") as f:
	line_list = f.readlines()
	for index, line in enumerate(line_list):
		line = line.strip()
		if line[-4:] == ".jpg":
			print "----------------------"
			#print index, line
			label_number = int(line_list[index + 1].strip())
			print "label number:", label_number
			#print line_list[index: index + 2 + label_number]
			write_xml(line_list[index].strip(), line_list[index+2: index + 2 + label_number])
			


