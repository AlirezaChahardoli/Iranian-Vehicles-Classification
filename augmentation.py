import os 
import cv2 
import albumentations as A 
from tqdm import tqdm 


"""
augmentation.py
Data augmentation using Albumentations.
"""

def augment_and_save(input_dir, output_dir, augmentations_per_image=30): 
    transform = A.Compose([ A.RandomBrightnessContrast(brightness_limit=0.1,contrast_limit=0.1,p=0.3), 
                           A.Rotate(limit=20, p=0.5), 
                           #A.HorizontalFlip(p=0.5), 
                           A.GaussianBlur(blur_limit=3,p=0.1), 
                           A.RandomShadow(shadow_roi=(0,0.3,1,1),num_shadows_lower=1,num_shadows_upper=1,p=0.05), 
                           A.RandomRain(p=0.02), 
                           A.RandomFog(fog_coef_lower=0.05,fog_coef_upper=0.1,p=0.02), 
                           A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=3, p=0.3), 
                           A.Perspective(scale=(0.02, 0.05), p=0.2) ]) 
    os.makedirs(output_dir, exist_ok=True) 
    image_files = os.listdir(input_dir) 
    for image_file in tqdm(image_files): 
        image_path = os.path.join(input_dir, image_file) 
        image = cv2.imread(image_path) 
        for i in range(augmentations_per_image): 
            augmented = transform(image=image) 
            aug_img = augmented["image"] 
            output_path = os.path.join(output_dir, f"{image_file.split('.')[0]}_aug_{i}.jpg") 
            cv2.imwrite(output_path, aug_img) 