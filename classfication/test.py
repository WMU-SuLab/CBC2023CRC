import albumentations as albu
import cv2
from fund_detect.nets import models
import torch
from convnext.code.part_with_full.util import get_fovea_point
#加载黄斑中心点定位模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "/Jane_TF_classification/convnext/code/fund_detect/models/model.pth"
print("Checkpoint file: {:s}".format(CHECKPOINT_FILE))
model_state_dict = torch.load(CHECKPOINT_FILE)
fd_model = models.resnet101(num_classes=2, pretrained=True)
fd_model.load_state_dict(model_state_dict)
fd_model.to(device)

data_transform = {
    'train.py':albu.Compose([
                          #albu.Resize(224,224),
                          albu.HorizontalFlip(p=0.5),
                          albu.VerticalFlip(p=0.5),
                          albu.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0),
                          ], keypoint_params=albu.KeypointParams(format='xy')),
    'val':albu.Compose([
                          #albu.Resize(224,224),
                          albu.Normalize([0.485, 0.456, 0.406],[ 0.229, 0.224,0.225],max_pixel_value=255.0,p=1.0),
                          ], keypoint_params=albu.KeypointParams(format='xy'))
}
img_path = '/data/home/yangjy/data/five_class/train/grade3/1901130035_OD.jpg'
image = cv2.imread(img_path)
cv2.imwrite('./image.jpg',image)

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
fovea = get_fovea_point(image_path=img_path,fovea_model=fd_model)
image_point = image.copy()
cv2.circle(image_point,(int(fovea[0]),int(fovea[1])),10,(0,0,0),thickness=-1)
cv2.imwrite('./image_point.jpg',image_point)

transformed = data_transform['train.py'](image = image,keypoints = [fovea]) #transform的时候都先不进行resize,
transformed_image = transformed['image']    # image此时是narray类型
transformed_point = transformed['keypoints']

# 根据transform之后进行roi的裁剪


trans_image_point = transformed_image.copy()*255.0
cv2.circle(trans_image_point,(int(transformed_point[0][0]),int(transformed_point[0][1])),10,(255,255,255),thickness=-1)
cv2.imwrite('./transformed_image_point.jpg',trans_image_point)