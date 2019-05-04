import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio

from PIL import Image, ImageDraw
from pyramid import build_sfd
from layers import *
import cv2
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
torch.cuda.set_device(-1)


print('Loading model..')
ssd_net = build_sfd('test', 640, 2)
net = ssd_net
net.load_state_dict(torch.load('./weights/Res50_pyramid.pth'))
net = net.cuda()
net.eval()
print('Finished loading model!')
'''
if torch.cuda.device_count() > 1:
  net = nn.DataParallel(net,[0,1])

net.to(device)
'''
dirpath = '/home/data/FACE/vid_3/'
savepath = '/home/data/FACE/vid-3-face/'

def detect_face(image, shrink):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    ###***** shrink **********##########
    #print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123],dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    #x = Variable(x, volatile=True)
    x = Variable(x, volatile=True).cuda()

    net.priorbox = PriorBoxLayer(width,height)
    #the following part is very important,may be report memory error without them
    with torch.no_grad():
        y = net(x)

    detections = y.data

    scale = torch.Tensor([width, height, width, height])

    boxes=[]
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.01:
            score = detections[0,i,j,0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            boxes.append([pt[0],pt[1],pt[2],pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image,0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image,1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b



def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

def resize_image(image, height, width):
     top, bottom, left, right = (0, 0, 0, 0)

     h, w, _ = image.shape

     #对于长宽不相等的图片，找到最长的一边
     longest_edge = max(h, w)    

     #计算短边需要增加多上像素宽度使其与长边等长
     if h < longest_edge:
         dh = longest_edge - h
         top = dh // 2
         bottom = dh - top
     elif w < longest_edge:
         dw = longest_edge - w
         left = dw // 2
         right = dw - left
     else:
         pass 

     #RGB颜色
     BLACK = [0, 0, 0]

     #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
     constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)

     #调整图像大小并返回
     return cv2.resize(constant, (height, width))


def write_to_txt(det,image,num):
    n=0
    s=0
    j=0
    #f.write('{:s}\n'.format(str(num)+'.jpg'))
    #f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        if n < score:
            n = score
            s = score
            j = i
        #f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
        #        format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
    #this part to crop image ,need the pixes number is int type
    #print('j:',j)
    #hight=ymax-ymin
    #width=xmax-xmin
    #[ymin:ymin+hight,xmin:xmin+width]
    hight=det[j][3] - det[j][1] + 1
    width=det[j][2] - det[j][0] + 1
    x1=det[j][0]
    y1=det[j][1]
    if x1 <0:
        x1=0
    if y1 <0:
        y1=0
    #[ymin:ymin+hight,xmin:xmin+width]
    cropimg=image[int(y1):int(y1+hight),int(x1):int(x1+width)]
    cropimg=resize_image(cropimg, 224, 224) #use to vgg16
    #cropimg=image[:int(130.1+242.4),175:int(175.4+313.7)]
    #new_img=Image.fromarray(cropimg) #transfrom the array to image
    #new_img.show()
    cv2.imwrite(savepath + str(num.split('-')[0]) + '/' + str(num) + '.jpg',cropimg)


if __name__ == '__main__':
    '''
    subset = 'val' # val or test
    if subset is 'val':
        wider_face = sio.loadmat('/home/guoqiushan/share/workspace/caffe-ssd-s3fd/sfd_test_code/WIDER_FACE/wider_face_val.mat')    # Val set
    else:
        wider_face = sio.loadmat('/home/guoqiushan/share/workspace/caffe-ssd-s3fd/sfd_test_code/WIDER_FACE/wider_face_test.mat')   # Test set
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    Path = '/home/tmp_data_dir/zhaoyu/wider_face/WIDER_val/images/'
    save_path = '/home/guoqiushan/share/workspace/caffe-ssd-s3fd-focal/sfd_test_code/WIDER_FACE/eval_tools_old-version/tmp_haha' + '_' + subset + '/'

    
    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        if not os.path.exists(save_path + str(event[0][0].encode('utf-8'))[2:-1] ):
            os.makedirs(save_path + str(event[0][0].encode('utf-8'))[2:-1] )
        for num, file in enumerate(filelist):
            
            im_name = str(file[0][0].encode('utf-8'))[2:-1] 
            Image_Path = Path + str(event[0][0].encode('utf-8'))[2:-1] +'/'+im_name[:] + '.jpg'
            print(Image_Path)
    '''

    list = os.listdir(dirpath) #list the content and file
    n=0
    for i in range(0,len(list)):
        path = os.path.join(dirpath,list[i])
        #save_path = os.path.join(savepath,list[i])
        os.makedirs(savepath+list[i].split('.')[0], exist_ok=True)
        checkpath=savepath+list[i].split('.')[0]
        filenum=len([lists for lists in os.listdir(checkpath) if os.path.isfile(os.path.join(checkpath, lists))])
        if filenum == 32:
            continue
        # input image
        #image = cv2.imread(path,cv2.IMREAD_COLOR)
        # input video
        camera = cv2.VideoCapture(path)
        if not camera.isOpened():
            print("cannot open camear")
            exit(0)
        j=0
        while True:
            try:
                ret, frame = camera.read()
                #print('ret frame:',ret,frame.shape)
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
                #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
                #print('max_im_shrink:',max_im_shrink)
                max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
            
                shrink = max_im_shrink if max_im_shrink < 1 else 1

                det0 = detect_face(image, shrink)  # origin test
                det1 = flip_test(image, shrink)    # flip test
                [det2, det3] = multi_scale_test(image, max_im_shrink)#min(2,1400/min(image.shape[0],image.shape[1])))  #multi-scale test
                #print('image:',image.shape)
                det4 = multi_scale_test_pyramid(image, max_im_shrink)
                det = np.row_stack((det0, det1, det2, det3, det4))

                dets = bbox_vote(det)
                j=j+1
                #print('j:',j)
                #f = open(savepath + list[i].split('.')[0]+'-'+str(j)+ '.txt', 'w')
                #print('det:',dets)
                #write_to_txt(f,dets,image,list[i].split('.')[0]+'-'+str(j))
                write_to_txt(dets,image,list[i].split('.')[0]+'-'+str(j))
            except:
                fi = open('/home/ye/bugvid3'+ '.txt', 'w')
                fi.write('{:s}\n'.format(str(list[i])))
                fi.close()
                break
            if j == 32:
                #n+=1
                #print('finish-------------------')
                break
        n+=1
        print('n:',n)
            #print('event:%d num:%d' % (index + 1, num + 1))
