from albumentations.augmentations.transforms import GaussNoise
import cv2
import os
import numpy as np
import os.path as osp
import albumentations as alb
# from torch._C import Ident
# from torch.nn.modules.linear import Identity


class IsotropicResize(alb.DualTransform):
    def __init__(self,
                 max_side,
                 interpolation_down=cv2.INTER_AREA,
                 interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False,
                 p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self,
              img,
              interpolation_down=cv2.INTER_AREA,
              interpolation_up=cv2.INTER_CUBIC,
              **params):
        return isotropically_resize_image(
            img,
            size=self.max_side,
            interpolation_down=interpolation_down,
            interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img,
                          interpolation_down=cv2.INTER_NEAREST,
                          interpolation_up=cv2.INTER_NEAREST,
                          **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Identity():
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class GroupTrainTransform():
    def __init__(self):
        self.ImageCompression = alb.ImageCompression(quality_lower=60,
                                                     quality_upper=100,
                                                     p=1),
        self.GaussNoise = alb.GaussNoise(p=1),
        self.GaussianBlur = alb.GaussianBlur(blur_limit=(3, 5), p=1),
        self.HorizontalFlip = alb.HorizontalFlip(p=1),
        self.LightChange = alb.OneOf([
            alb.RandomBrightnessContrast(),
            alb.FancyPCA(),
            alb.HueSaturationValue()
        ],
                                     p=1),
        self.ShiftRotate = alb.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=1),

    def _apply_aug(imgs, aug_method):
        for i, img in enumerate(imgs):
            imgs[i] = aug_method(image=img)['image']
        return imgs

    def __call__(self, imgs):
        # img compress
        if np.random.random() < 0.3:
            imgs = self._apply_aug(imgs, self.ImageCompression)
        # gauss noise
        if np.random.random() < 0.1:
            imgs = self._apply_aug(imgs, self.GaussNoise)
        # gauss blur
        if np.random.random() < 0.05:
            imgs = self._apply_aug(imgs, self.GaussianBlur)
        # flip
        if np.random.random() < 0.5:
            imgs = self._apply_aug(imgs, self.HorizontalFlip)
        # light
        if np.random.random() < 0.5:
            imgs = self._apply_aug(imgs, self.LightChange)
        # shift rotate
        if np.random.random() < 0.5:
            imgs = self._apply_aug(imgs, self.ShiftRotate)
        return imgs


class GroupTestTransform(Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_group_transform(mode):
    if mode == 'train':
        return GroupTrainTransform()
    elif mode == 'test':
        return GroupTestTransform()
    else:
        raise (NotImplementedError)


def isotropically_resize_image(img,
                               size,
                               interpolation_down=cv2.INTER_AREA,
                               interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def get_transform(mode, size):
    if mode == 'train':
        return get_train_transform(size)
    elif mode == 'test':
        return get_test_transform(size)
    else:
        raise (NotImplementedError)


def get_test_transform(size):
    return alb.Compose([
        IsotropicResize(max_side=size),
        alb.PadIfNeeded(min_height=size,
                        min_width=size,
                        border_mode=cv2.BORDER_CONSTANT),
    ])


def get_train_transform(size):
    return alb.Compose([
        # alb.GaussNoise(p=0.1),
        # alb.GaussianBlur(blur_limit=(3, 5), p=0.1),
        alb.HorizontalFlip(),
        alb.OneOf([
            IsotropicResize(max_side=size,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size,
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR),
        ],
                  p=1),
        alb.PadIfNeeded(min_height=size,
                        min_width=size,
                        border_mode=cv2.BORDER_CONSTANT),
        # alb.OneOf([
        #     alb.RandomBrightnessContrast(),
        #     alb.FancyPCA(),
        #     alb.HueSaturationValue()
        # ],
        #           p=0.5),
        # alb.ToGray(p=0.2),
        # alb.ShiftScaleRotate(shift_limit=0.1,
        #                      scale_limit=0.1,
        #                      rotate_limit=5,
        #                      border_mode=cv2.BORDER_CONSTANT,
        #                      p=0.5),
    ])



def scan_jpg_from_img_dir(img_dir):
    img_ps = [
        osp.join(img_dir, name)
        for name in sorted(os.listdir(img_dir),
                           key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if '.jpg' in name  # !! sort key
    ]
    return img_ps