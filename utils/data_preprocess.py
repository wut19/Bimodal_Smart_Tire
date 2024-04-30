import cv2
import numpy as np
import os
 
def augment_image(image_dir, dest_dir, sigmas=[0,20,60,80], amounts=[0, 0.5, 0.1, 0.15,0.2], s_vs_ps = [0.5]):
    class_dirs = os.listdir(image_dir)
    for class_dir in class_dirs:
        count = 0
        files = os.listdir(os.path.join(image_dir, class_dir))
        os.makedirs(os.path.join(dest_dir, class_dir))
        for file in files:
            image = cv2.imread(os.path.join(image_dir, class_dir, file))
            height, width, channels = image.shape
            for sigma in sigmas:
                for amount in amounts:
                    for s_vs_p in s_vs_ps:
                        """ gaussian noise """
                        mean = 0
                        gauss = np.random.normal(mean,sigma,(height,width, channels))
                        noised_image = image + gauss

                        """ pepper salt noise """
                        num_salt = np.ceil(amount * image.size * s_vs_p)
                        coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
                        noised_image[coords[0],coords[1],:] = [255,255,255]
                        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
                        coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
                        noised_image[coords[0],coords[1],:] = [0,0,0]
                        noised_image = np.clip(noised_image, 0, 255)
                        save_path = os.path.join(dest_path, class_dir, f'{count}.png')
                        cv2.imwrite(save_path, noised_image)
                        count += 1

image_path = 'VisualTactileData_all/tactile/v'
dest_path = 'VisualTactileData_all/tactile/augmented_v'
augment_image(image_path, dest_path)
