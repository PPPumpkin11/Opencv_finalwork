import cv2
import numpy as np
import os

# 处理图像，处理过程包括高斯滤波、canny算子提取边缘、闭运算
def process(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 用高斯滤波进行图像平滑
    img_canny=cv2.Canny(imgBlur,50,150)  # 提取图像边缘

    # 进行闭运算
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(img_canny, kernel, iterations=1)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    # cv2.imshow('imgthreshold',imgThreshold)
    # cv2.waitKey(0)
    return imgThreshold

# 进行图像矫正
def correct(img):
    imgThreshold=process(img)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # 针对有个别图像裁剪之后，大片的黑色区域里面还有一些白色孔洞，于是需要先去除孔洞
    if len(contours)>1:
        contours1=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if area>=8000:
                contours1.append(contour)
        rect = cv2.minAreaRect(contours1[0])
    else:
        rect = cv2.minAreaRect(contours[0])  # 得到最小外接矩形的中心，（宽，高），旋转角度

    # print(len(contours))
    if rect[2]>=45:
        angle=rect[2]-90
    elif rect[2]<45:
        angle=rect[2]

    h = img.shape[0]
    w = img.shape[1]
    rotate = cv2.getRotationMatrix2D((h * 0.5, w * 0.5), angle, 1)  # 根据旋转角度对图像进行旋转，将图像变为水平
    after_rotate = cv2.warpAffine(img, rotate, (w, h))  # 产生出一张倾斜矫正之后的图像
    # cv2.imshow('img', img)
    # cv2.imshow('after_rotate', after_rotate)
    # cv2.waitKey(0)
    return after_rotate

# 进行仿射变换
def transform(after_rotate):
    h1 = after_rotate.shape[0]
    w1 = after_rotate.shape[1]
    after_rotate_Threshold = process(after_rotate)  # 同样的操作，为了提取边框
    contours1, hierarchy1 = cv2.findContours(after_rotate_Threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours1))
    if len(contours1)>1:
        contours11=[]
        for contour in contours1:
            area=cv2.contourArea(contour)
            if area>=8000:
                contours11.append(contour)
        rect1 = cv2.minAreaRect(contours11[0])
    else:
        rect1=cv2.minAreaRect(contours1[0])
    box = cv2.boxPoints(rect1)
    # print(box)
    # 由于box四个位置对应的左上、右上、左下、右下的顺序不固定，所以要根据坐标大小重新找出
    box_x=np.zeros((4,1))
    box_y=np.zeros((4,1))
    for i in range(4):
        box_x[i]=box[i][0]
        box_y[i]=box[i][1]
    for j in range(4):
        if box_x[j]-box_x[np.argmin(box_x)]<=1:
            if box_y[j]<box_y[np.argmin(box_x)]:
                leftup=(box[j][0],box[j][1])
                leftdown=(box[np.argmin(box_x)][0],box[np.argmin(box_x)][1])
            else:
                leftup=(box[np.argmin(box_x)][0],box[np.argmin(box_x)][1])
                leftdown=(box[j][0],box[j][1])
    for k in range(4):
        if box_x[k]-box_x[np.argmax(box_x)]<=1:
            if box_y[k]<box_y[np.argmax(box_x)]:
                rightup=(box[k][0],box[k][1])
            else:
                rightup=(box[np.argmax(box_x)][0],box[np.argmax(box_x)][1])

    # 再进行仿射变换
    pts1 = np.float32([leftup, leftdown, rightup])
    pts2 = np.float32([[0, 0], [0, h1], [w1, 0]])  # 左上 左下 右上
    matrix = cv2.getAffineTransform(pts1, pts2)
    imgWarpColored = cv2.warpAffine(after_rotate, matrix, (w1, h1))
    # cv2.imshow('warp', imgWarpColored)
    # cv2.waitKey(0)
    return imgWarpColored

# 除了标准图像以外的其他图像需要先进行剪切处理
def process_big_image(img):
    img=img[2600:4400,:]
    img=cv2.resize(img,(int(img_original.shape[1]),int(img_original.shape[0])))
    return img

def process_img_orginal(img_orginal):
    after_rotate=correct(img_original)
    h1 = after_rotate.shape[0]
    w1 = after_rotate.shape[1]
    after_rotate_Threshold = process(after_rotate)  # 同样的操作，为了提取边框
    contours1, hierarchy1 = cv2.findContours(after_rotate_Threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect1 = cv2.minAreaRect(contours1[0])
    box = cv2.boxPoints(rect1)
    pts1 = np.float32([[box[0][0],box[0][1]],[box[3][0],box[3][1]],[box[1][0],box[1][1]]])
    pts2 = np.float32([[0, 0], [0, h1], [w1, 0]])  # 左上 左下 右上
    matrix = cv2.getAffineTransform(pts1, pts2)
    imgWarpColored = cv2.warpAffine(after_rotate, matrix, (w1, h1))
    # cv2.imshow('warp', imgWarpColored)
    # cv2.waitKey(0)
    return imgWarpColored

img_original=cv2.imread('./chemical_tube.png')  # 标准的图像
model=process_img_orginal(img_original)
model=cv2.cvtColor(model,cv2.COLOR_BGR2GRAY)
# 对用来测试的图像先对齐并且保存
daqipao_path = list(sorted(os.listdir('opencv_course_design_data/NG/daqipao/')))
jiaodai_path = list(sorted(os.listdir('opencv_course_design_data/NG/jiaodai/')))
OK_path = list(sorted(os.listdir('opencv_course_design_data/OK/')))
for i in range(len(daqipao_path)):
    img=cv2.imread('./opencv_course_design_data/NG/daqipao/'+daqipao_path[i])
    img1=process_big_image(img)
    after_rotate=correct(img1)
    after_transform=transform(after_rotate)
    cv2.imwrite(os.path.join('./opencv_course_design_data/aftertransform daqipao/'+daqipao_path[i]),after_transform)
    cv2.waitKey(0)

for i in range(len(jiaodai_path)):
    img=cv2.imread('./opencv_course_design_data/NG/jiaodai/'+jiaodai_path[i])
    img1=process_big_image(img)
    after_rotate=correct(img1)
    after_transform=transform(after_rotate)
    cv2.imwrite(os.path.join('./opencv_course_design_data/aftertransform jiaodai/'+jiaodai_path[i]),after_transform)
    cv2.waitKey(0)

for i in range(len(OK_path)):
    img=cv2.imread('./opencv_course_design_data/OK/'+OK_path[i])
    img1=process_big_image(img)
    after_rotate=correct(img1)
    after_transform=transform(after_rotate)
    cv2.imwrite(os.path.join('./opencv_course_design_data/aftertransform OK/'+OK_path[i]),after_transform)
    cv2.waitKey(0)

def jiaodai(img1,img2):
    ret1, thresh1 = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY)
    dif=thresh1-thresh2
    # cv2.imshow('dif',dif)
    kernel = np.ones((3, 3))
    dif=cv2.morphologyEx(dif,cv2.MORPH_OPEN,kernel,iterations=2)  # 两次开两次闭
    dif=cv2.morphologyEx(dif,cv2.MORPH_CLOSE,kernel,iterations=2)
    # cv2.imshow('dif', dif)
    # cv2.waitKey(0)
    contours,hierarchy=cv2.findContours(dif,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    if len(contours)>=24:
        return 'jiaodai'
    else:
        return 'OK'

# 产生用于缺陷检测的数据集，将ok数据集和jiaodai数据集放在一起
def dataset(path,path1):
    name=os.listdir(path)
    for i in range(len(name)):
        img=cv2.imread(path+'/'+name[i])
        cv2.imwrite(os.path.join(path1,name[i]), img)
        cv2.waitKey(0)
dataset(path='./opencv_course_design_data/aftertransform OK',path1='./opencv_course_design_data/ok_and_jiaodai/')
dataset(path='opencv_course_design_data/aftertransform jiaodai',path1='./opencv_course_design_data/ok_and_jiaodai/')
dataset(path='./opencv_course_design_data/aftertransform OK',path1='./opencv_course_design_data/ok_and_daqipao/')
dataset(path='opencv_course_design_data/aftertransform daqipao',path1='./opencv_course_design_data/ok_and_daqipao/')

name1=os.listdir('./opencv_course_design_data/ok_and_jiaodai')
num_correct=0
for i in range(len(name1)):
    img=cv2.imread('./opencv_course_design_data/ok_and_jiaodai/'+name1[i],cv2.IMREAD_GRAYSCALE)
    if jiaodai(img,model)[:3]==name1[i][:3]:
        num_correct+=1
precision_jiaodai=num_correct/len(name1)
correct_rate=precision_jiaodai/0.4 # 0.4是由于把jiaodai和OK数据集整合起来之后jiaodai占总数据集的0.4
print('检测胶带的正确率：',correct_rate)

# img_read=cv2.imread('./opencv_course_design_data/aftertransform OK/OK_0024.bmp',cv2.IMREAD_GRAYSCALE)
def daqipao(img1,img2):

    ret1, thresh1 = cv2.threshold(img1, 75, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img2,75, 255, cv2.THRESH_BINARY)
    # cv2.imshow('1',thresh1)
    # cv2.imshow('2',thresh2)
    dif=thresh1-thresh2
    # cv2.imshow('dif', dif)
    kernel = np.ones((3, 3))
    dif = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel, iterations=3)
    dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, kernel, iterations=3)
    # cv2.imshow('dif1',dif)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(dif, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    if len(contours) >= 20:
        return 'da_qipao'
    else:
        return 'OK'

name2=os.listdir('./opencv_course_design_data/ok_and_daqipao')
num_correct1=0
for i in range(len(name2)):
    img1=cv2.imread('./opencv_course_design_data/ok_and_daqipao/'+name2[i],cv2.IMREAD_GRAYSCALE)
    if daqipao(img1,model)[:3]==name2[i][:3]:
        num_correct1+=1

precision_daqipao=num_correct1/len(name2)
correct_rate1=precision_daqipao/0.4 # 0.4是由于把jiaodai和OK数据集整合起来之后jiaodai占总数据集的0.4
print('检测大气泡的正确率：',correct_rate1)
