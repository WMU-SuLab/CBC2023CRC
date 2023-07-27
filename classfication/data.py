import os.path
from tqdm import tqdm
from torch.utils.data import Dataset
import pathlib
from torchvision import transforms
import numpy as np
import  torch
from PIL import Image,ImageFile
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.456, 0.485, 0.406], [0.224, 0.229, 0.225])])
#
# def find_classes(directory: str) :
#     """ÔÚÄ¿±êÄ¿Â¼ÖÐ²éÕÒÀàÎÄ¼þ¼ÐÃû³Æ¡£
#
#     ¼ÙÉèÄ¿±êÄ¿Â¼ÊÇÒÔ±ê×¼Í¼Ïñ·ÖÀà¸ñÊ½¡£
#
#     Args:
#         directory (str): ´ÓÖÐ¼ÓÔØÀàÃûµÄÄ¿±êÄ¿Â¼¡£
#
#     Returns:
#         Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
#
#     Example:
#         find_classes("food_images/train")
#         # >>> (["class_1", "class_2"], {"class_1": 0, ...})
#     """
#     # 1. Í¨¹ýÉ¨ÃèÄ¿±êÄ¿Â¼»ñÈ¡ÀàÃû
#     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
#     # 2. Èç¹ûÕÒ²»µ½ÀàÃûÔò Òý·¢´íÎó
#     if not classes:
#         raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
#     # 3. ´´½¨Ò»¸öË÷Òý±êÇ©×Öµä£¨¼ÆËã»ú ¸üÏ²»¶Êý×Ö±êÇ©¶ø²»ÊÇ ×Ö·û´®±êÇ©£©
#     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#     return classes, class_to_idx
# # 1. torch.utils.data.Dataset µÄ×ÓÀà
# class ImageFolderCustom(Dataset):
#     # 2. Ê¹ÓÃ targ_dir ºÍ transform (¿ÉÑ¡) ²ÎÊý³õÊ¼»¯
#     def __init__(self, targ_dir: str, target_folders,transform=None,) -> None:
#         # 3. ´´½¨ÀàÊôÐÔ
#         # »ñÈ¡ËùÓÐÍ¼Æ¬Â·¾¶
#         # Ö¸¶¨Òª¶ÁÈ¡µÄÎÄ¼þ¼ÐÁÐ±í
#
#         # ³õÊ¼»¯´æ´¢ÎÄ¼þÂ·¾¶µÄÁÐ±í
#         file_paths = []
#         # ±éÀúÃ¿¸öÄ¿±êÎÄ¼þ¼Ð£¬»ñÈ¡ÎÄ¼þÂ·¾¶
#         for folder in target_folders:
#             folder_path = pathlib.Path(targ_dir) / folder
#             file_paths.extend(list(folder_path.glob("*.png")))
#         self.paths = list(pathlib.Path(targ_dir).glob("*/*.png")) # note: you'd have to update this if you've got .png's or .jpeg's
#         # ÉèÖÃ transforms
#         self.transform = transform
#         # ÉèÖÃ classes ºÍ class_to_idx ÊôÐÔ
#         self.classes, self.class_to_idx = find_classes(targ_dir)
#     # 4. Ê¹ÓÃº¯Êý¼ÓÔØÍ¼Ïñ
#     def load_image(self, index: int) -> Image.Image:
#         "Opens an image via a path and returns it."
#         image_path = self.paths[index]
#         return Image.open(image_path)
#     # 5. ÖØÐ´ __len__() ·½·¨ (optional but recommended for subclasses of torch.utils.data.Dataset)
#     def __len__(self) -> int:
#         "Returns the total number of samples."
#         return len(self.paths)
#     # 6. ÖØÐ´ __getitem__() ·½·¨ (required for subclasses of torch.utils.data.Dataset)
#     def __getitem__(self, index: int) :
#         "Returns one sample of data, data and label (X, y)."
#         img = self.load_image(index)
#         class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
#         class_idx = self.class_to_idx[class_name]
#         # Transform if necessary
#         return self.transform(img), class_idx # return data, label (X, y)


import pathlib
import glob
from torch.utils.data import Dataset
class ImageFolderCustom(Dataset):

    # 2. Ê¹ÓÃ targ_dir ºÍ transform (¿ÉÑ¡) ²ÎÊý³õÊ¼»¯
    def __init__(self, targ_dir,exclude_files, transform=None):

        # 3. ´´½¨ÀàÊôÐÔ
        # »ñÈ¡ÏëÒªµÄÍ¼Æ¬Â·¾¶

        self.paths = [file for file in pathlib.Path(targ_dir).rglob('*/*') if
                      not any(exclude_file in str(file) for exclude_file in exclude_files)]
        #self.paths = [file for file in pathlib.Path(targ_dir).rglob('*/*.??[fg]') if not glob.fnmatch.fnmatch(file, '*label00*') ]
        # self.paths = [file for file in pathlib.Path(targ_dir).rglob('*/*.??[fg]') if not glob.fnmatch.fnmatch(file, '*label00*') ]
        #print(self.paths)
        # you've got .png's or .jpeg's
        # ÉèÖÃ transforms
        self.transform = transform
        # ÉèÖÃ classes ºÍ class_to_idx ÊôÐÔ
        self.class_to_idx = {'image00':0,'image11':1}

    # 4. Ê¹ÓÃº¯Êý¼ÓÔØÍ¼Ïñ
    def load_image(self, index: int) :
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. ÖØÐ´ __len__() ·½·¨ (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self):
        "Returns the total number of samples."
        return len(self.paths)

    # 6. ÖØÐ´ __getitem__() ·½·¨ (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index) :
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)



if __name__ == '__main__':

    data = ImageFolderCustom(r"C:\Users\12142\Desktop\test\data\train",['label00','label11'],transform = transform )
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=0)
    data_loader = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        img, class_idx = data
        print(class_idx)



    print(type(data))

    # print(a,b.shape,c.shape,d)  #b.shape,c.shape = (256, 256) (1444, 1620)

    # print(out.shape)

