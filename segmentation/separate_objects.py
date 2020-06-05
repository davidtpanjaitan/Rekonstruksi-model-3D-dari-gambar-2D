import os
import cv2
import numpy as np

# separate a list of images with and masks with multiple objects 
# into separate masks for each object
def separate_by_contour(images, masks):
    new_images, new_masks = [], []
    for image_path, mask_path in zip(images, masks):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)
        conts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(conts) > 1:
            #print('found more than 1 objects in ', conts)
            imgArea = img.shape[0] * img.shape[1]
            for i, cont in enumerate(conts):
                if cv2.contourArea(cont)/imgArea < 0.05:
                    continue
                topleft = np.min(cont, axis=0)[0]
                botright = np.max(cont, axis=0)[0]
                new_shape = (botright - topleft)
                print('new shape ', [new_shape[1], new_shape[0]])
                new_mask = np.zeros([new_shape[1], new_shape[1]])
                pad = new_shape[1] - new_shape[0]
                new_img = cv2.copyMakeBorder(
                    img[topleft[1]:botright[1], topleft[0]:botright[0]], 
                    top=0,
                    bottom=0,
                    left=0,
                    right=pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value = [128,128,128])
                cont = cont - topleft
                cv2.drawContours(new_mask, [cont], 0, 255, -1)
                new_mask_path = image_path[:-4] + '(%d)_mask.png' % i
                new_image_path = image_path[:-4] + '(%d).png' % i
                print('created new mask at ', new_mask_path)
                cv2.imwrite(new_mask_path, new_mask)
                cv2.imwrite(new_image_path, new_img)
                new_images.append(new_image_path)
                new_masks.append(new_mask_path)
        new_images.append(image_path)
        new_masks.append(mask_path)
    return new_images, new_masks
