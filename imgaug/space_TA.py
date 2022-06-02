import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import imgaug.augments as augments
import yaml
from PIL import Image

class TrivialAugment:
    def __init__(self, config_path):
        self.config_path = config_path
        self.augmentation = None
        self.parse_config()
        print('init config space')
        
    def parse_config(self):
        ALL_TRAN = []
        with open(self.config_path) as file:
            aug_file = yaml.load(file, Loader=yaml.FullLoader)
        for item in range(len(aug_file[0]['augmentations'])):
            for name_aug in aug_file[0]['augmentations'][item]:
            # print(name_aug)
                if aug_file[0]['augmentations'][item][name_aug] == 1:
                # print(name_aug)
                    aug = getattr(augments, name_aug)
                    ALL_TRAN.append(aug)
        augmenter = iaa.OneOf([])
        for aug in ALL_TRAN:
            augmenter.add(aug)
        print(f'Count of augments - {len(augmenter)}')
        self.augmentation = augmenter

    def __call__(self, image_or, bbox_or = None):
        # print('1',type(bbox_or))
        if isinstance(image_or, Image.Image):
            image_or = np.asarray(image_or)
        if bbox_or is not None:    
            # print('2',bbox_or)
            count_bbox = len(bbox_or)
            ia_bboxes = ia.BoundingBoxesOnImage.from_xyxy_array(bbox_or, shape=image_or.shape)
            # ia_bboxes = BoundingBoxesOnImage.from_xyxy_array(bbox_or, shape=image_or.shape)
            image_aug, aug_bbox = self.augmentation(image=image_or, bounding_boxes=ia_bboxes)
            aug_bbox_array = aug_bbox.to_xyxy_array()
            return image_aug, aug_bbox
        else:
            image_aug = self.augmentation(image=image)
            return image_aug
