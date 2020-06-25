import cv2
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from os import listdir
import skimage.io as io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join as pjoin
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.transform import rotate, AffineTransform, warp

warnings.filterwarnings('ignore')

def normalize(arr):
    ''' Function to scale an input array to [-1, 1]
    arr   : Array that is to be normalized
    return: Normalized array'''

    arr_min = arr.min()
    arr_max = arr.max()

    arr_range = arr_max - arr_min
    scaled = np.array((arr - arr_min) / float(arr_range), dtype='f')
    arr_new = -1 + (scaled * 2)

    return arr_new


def section_and_masks_to_npy(line, low, high, image_type="plt"):
    '''get section and mask from image folder and save it in npy format
       line       : "inlines"/"crosslines"
       low & high : section number
       image_type : "plt"/"PIL"
       Returns    : npy file'''

    sections = []
    masks = []
    for i, filename in enumerate(listdir(line)):
        if filename.split('.')[1] == 'tiff':
            line_num = int((filename.split('.')[0]).split('_')[1])

            if (line_num > low) and (line_num <= high):

                if image_type == "plt":
                    seismic_section = plt.imread(pjoin(line, filename))[:, :, 0]
                elif image_type == "PIL":
                    seismic_section = Image.open(pjoin(line, filename))
                else:
                    'Unknown image type! possible input: ["plt", "PIL"]'
                seismic_section = np.array(seismic_section)
                seismic_section = normalize(seismic_section)
                sections.append(seismic_section)

                mask_filename = filename.split('.')[0] + '_mask.png'
                seismic_facies = Image.open(pjoin('masks', mask_filename))
                seismic_facies = np.array(seismic_facies)
                masks.append(seismic_facies)

    npy_sections = np.asarray(sections)
    npy_masks = np.asarray(masks)

    print("Section Shape\t:{0}\nMask Shape\t:{1}".format(npy_sections.shape, npy_masks.shape))

    return npy_sections, npy_masks

def plot_section_mask(section, mask, vmin = None, vmax = None, figsize = (25, 8)):

    '''Plot section and corresponding mask, works for both section based and patch based
    section    : Seismic Sections in 3D array
    mask       : Corresponding Mask
    vmin, vmax : Normalize Section array between vmin, vmax value for visualization purpose'''

    idx = np.random.randint(0, mask.shape[0], (20))
    _, ax = plt.subplots(2, 20, figsize = figsize)
    for i in range(len(ax[0])):
        ax[0][i].imshow(mask[idx[i]], vmin = vmin, vmax = vmax)
        ax[0][i].set_yticks([])
        ax[0][i].set_xticks([])

        ax[1][i].imshow(section[idx[i]])
        ax[1][i].set_yticks([])
        ax[1][i].set_xticks([])
    plt.tight_layout()

def extract_patch(section, mask, stride = 50, patch = 99, padding = "VALID"):

    '''Extract patch from section and mask array using TesorFlow
       patch   : size of patch to be extracted
       Stride  : stride of patch window
       padding : Don't use "SAME" as it will pad with 0'''

    images = section[:,:,:]
    labels = mask[:,:,:]

    images = np.expand_dims(images, axis=3)
    labels = np.expand_dims(labels, axis=3)

    patch_images = tf.image.extract_patches(images, (1,patch,patch,1), (1,stride,stride,1), (1,1,1,1), padding = padding, name=None)
    patch_labels = tf.image.extract_patches(labels, (1,patch,patch,1), (1,stride,stride,1), (1,1,1,1), padding = padding, name=None)

    patch_images = tf.reshape(patch_images, (-1,patch,patch)).numpy()
    patch_labels = tf.reshape(patch_labels, (-1,patch,patch)).numpy()

    print("Patch Images Shape\t:{0}\nPatch Masks Shape\t:{1}".format(patch_images.shape, patch_labels.shape))

    return patch_images, patch_labels

def labels_conversion_canada_new(labels):

    '''Converts unwanted labels (0, 5, 6, 7) to 255 and renames (1, 2, 3, 4) to (0, 1, 2, 3) for Penobscot dataset'''

    labels = np.where(labels == 1, 255, labels)
    labels = np.where(labels == 0, 255, labels)
    labels = np.where(labels == 5, 255, labels)
    labels = np.where(labels == 6, 255, labels)
    labels = np.where(labels == 7, 255, labels)
    
    labels = np.where(labels == 2, 0, labels)
    labels = np.where(labels == 3, 1, labels)
    labels = np.where(labels == 4, 2, labels)

    return labels
    
def labels_conversion_canada_new_class_1_2_from_sections(labels):

    '''Converts unwanted labels (0, 5, 6, 7) to 255 and renames (1, 2, 3, 4) to (0, 1, 2, 3) for Penobscot dataset'''

    labels = np.where(labels == 1, 255, labels)
    labels = np.where(labels == 2, 255, labels)
    labels = np.where(labels == 0, 255, labels)
    labels = np.where(labels == 5, 255, labels)
    labels = np.where(labels == 6, 255, labels)
    labels = np.where(labels == 7, 255, labels)
    
    labels = np.where(labels == 3, 1, labels)
    labels = np.where(labels == 4, 2, labels)

    return labels

def labels_conversion_canada(labels):

    '''Converts unwanted labels (0, 5, 6, 7) to 255 and renames (1, 2, 3, 4) to (0, 1, 2, 3) for Penobscot dataset'''

    labels = np.where(labels == 0, 255, labels)
    labels = np.where(labels == 5, 255, labels)
    labels = np.where(labels == 6, 255, labels)
    labels = np.where(labels == 7, 255, labels)

    labels = np.where(labels == 1, 0, labels)
    labels = np.where(labels == 2, 1, labels)
    labels = np.where(labels == 3, 2, labels)
    labels = np.where(labels == 4, 3, labels)

    return labels

def labels_conversion_netherlands_new(labels):

    '''Converts unwanted labels (0, 1) to 255 and renames (0, 3, 4, 5) to (0, 1, 2, 3) for Netherlands F3 Block dataset'''

    labels = np.where(labels == 0, 255, labels)
    labels = np.where(labels == 1, 255, labels)
    labels = np.where(labels == 2, 255, labels)
    
    labels = np.where(labels == 3, 0, labels)
    labels = np.where(labels == 4, 1, labels)
    labels = np.where(labels == 5, 2, labels)

    return labels
    
def labels_conversion_netherlands(labels):

    '''Converts unwanted labels (0, 1) to 255 and renames (0, 3, 4, 5) to (0, 1, 2, 3) for Netherlands F3 Block dataset'''

    labels = np.where(labels == 1, 255, labels)
    labels = np.where(labels == 2, 255, labels)

    labels = np.where(labels == 0, 0, labels)
    labels = np.where(labels == 3, 1, labels)
    labels = np.where(labels == 4, 2, labels)
    labels = np.where(labels == 5, 3, labels)

    return labels

def filter_patches(images, labels, threshold = 0.70):

    '''Drops any patch with 255 if total pixel number for a particular patch exceeds threshold value
    returns : filtered patch (255 removed), based on threshold'''

    filtered_images = []
    filtered_labels = []
    count0 = 0
    total_pixel = sum(np.unique(labels[0], return_counts=True)[1])
    for i in range(images.shape[0]):
        unique = np.unique(labels[i], return_counts=True)
        if ((np.max(unique[1])/total_pixel) >= threshold): #checks if in a particular patch labels, any label is greater than given %age
            idx = np.argmax(unique[1]) #if above statement satisfies, then find out which label is that which statisfies above condition
            new_lbl = unique[0][idx] #if above statement satisfies, then find out which label is that which statisfies above condition
            if new_lbl == 255:
                continue #if that label is 255, don't save that patch
            else: #if that label  is anything but 255, save that patch
                filtered_images.append(images[i])
                filtered_labels.append(labels[i])
        else: #if first condition doesn't satisfy, then save all the patch
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])

    filtered_images = np.asarray(filtered_images)
    filtered_labels = np.asarray(filtered_labels)

    print("Filtered Patch Images Shape\t:{0}\nFiltered Patch Masks Shape\t:{1}".format(filtered_images.shape, filtered_labels.shape))

    return filtered_images, filtered_labels

def balance_class_dist(images, labels, class_to_be_balanced = 0, skipping_factor = 10):

    '''Skip class_to_be_balanced patch by a skipping factor to balance the dataset'''

    filtered_images = []
    filtered_labels = []
    count = 0
    for i in range(images.shape[0]):
        unique = np.unique(labels[i], return_counts=True)
        idx = np.argmax(unique[1]) #find out which label is most of the time present in a particular patch
        new_lbl = unique[0][idx] #find out which label is most of the time present in a particular patch
        if new_lbl == class_to_be_balanced: #if it's class 0, reduce the number of patches with a skipping factor
            if count % skipping_factor == 0:
                filtered_images.append(images[i])
                filtered_labels.append(labels[i])
            count+=1
        else: #if that label  is anything but 0, save that patch
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])

    filtered_images = np.asarray(filtered_images)
    filtered_labels = np.asarray(filtered_labels)

    print("Balanced Patch Images Shape\t:{0}\nBalanced Patch Masks Shape\t:{1}".format(filtered_images.shape, filtered_labels.shape))

    return filtered_images, filtered_labels

def balance_class_dist_for_classification(images, labels, class_to_be_balanced = 0, skipping_factor = 10):

    '''Skip class_to_be_balanced patch by a skipping factor to balance the dataset for classification based data'''

    filtered_images = []
    filtered_labels = []
    count = 0
    for i in range(images.shape[0]):
        if labels[i] == class_to_be_balanced: #if it's class 0, reduce the number of patches with a skipping factor
            if count % skipping_factor == 0:
                filtered_images.append(images[i])
                filtered_labels.append(labels[i])
            count+=1
        else: #if that label  is anything but 0, save that patch
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])

    filtered_images = np.asarray(filtered_images)
    filtered_labels = np.asarray(filtered_labels)

    print("Balanced Patch Images Shape\t:{0}\nBalanced Patch Masks Shape\t:{1}".format(filtered_images.shape, filtered_labels.shape))

    return filtered_images, filtered_labels

def classification_based_data_with_diff_threshold_for_a_particular_class(images, labels, class_threshold_be_dec = 2, dec_by = 0.1, threshold = 0.7):

    '''If within a patch, a class occurs more than the threshold value,
        accept that patch and reassign the labels as that class

        class_threshold_be_dec: class whose threshold to be decresed
        dec_by: threshold to be decreased by how much, for class_threshold_be_dec'''

    filtered_images = []
    filtered_labels = []

    total_pixel = sum(np.unique(labels[0], return_counts=True)[1])

    for i in range(images.shape[0]):
        unique = np.unique(labels[i], return_counts=True)
        if (np.max(unique[1])/total_pixel) >= (threshold - dec_by):
            idx = np.argmax(unique[1])
            new_lbl = unique[0][idx]
            if new_lbl == class_threshold_be_dec:
                filtered_images.append(images[i])
                filtered_labels.append(new_lbl)
            elif (np.max(unique[1])/total_pixel) >= threshold:
                idx = np.argmax(unique[1])
                new_lbl = unique[0][idx]
                if new_lbl == 255:
                    continue
                else:
                    filtered_images.append(images[i])
                    filtered_labels.append(new_lbl)
            else:
                continue

    filtered_images = np.asarray(filtered_images)
    filtered_labels = np.asarray(filtered_labels)

    print("Filtered Patch Images Shape\t:{0}\nFiltered Patch Labels Shape\t:{1}".format(filtered_images.shape, filtered_labels.shape))

    return filtered_images, filtered_labels

def classification_based_data(images, labels, threshold = 0.7):

    '''If within a patch, a class occurs more than the threshold value,
                                accept that patch and reassign the labels as that class'''

    filtered_images = []
    filtered_labels = []

    total_pixel = sum(np.unique(labels[0], return_counts=True)[1])

    for i in range(images.shape[0]):
        unique = np.unique(labels[i], return_counts=True)
        if (np.max(unique[1])/total_pixel) >= threshold:
            idx = np.argmax(unique[1])
            new_lbl = unique[0][idx]
            if new_lbl == 255:
                continue
            else:
                filtered_images.append(images[i])
                filtered_labels.append(new_lbl)
        else:
            continue

    filtered_images = np.asarray(filtered_images)
    filtered_labels = np.asarray(filtered_labels)

    print("Filtered Patch Images Shape\t:{0}\nFiltered Patch Labels Shape\t:{1}".format(filtered_images.shape, filtered_labels.shape))

    return filtered_images, filtered_labels


def class_dist_data(data, return_count=False):
    unique = np.unique(data, return_counts=True)
    if return_count == True:
        classes = unique
    else:
        classes = unique[0]
    classes_count = unique[1]
    total = sum(unique[1])
    percentage = np.round(100 * (classes_count / total), 1)

    print("Class Labels in given data: {0}\n%age of each class in data: {1}".format(classes, percentage))
    
def extract_particular_class(images, labels, desired_label):

    '''desired_label : Labels that needs to be extracted from a the dataset'''

    extracted_image = []
    
    for i in range(labels.shape[0]):
        lbl = labels[i]
        if lbl == desired_label:
            extracted_image.append(images[i])
        else:
            continue
        
    filtered_image = np.asarray(extracted_image)
    
    print("Extracted Patch Images Shape (For Class {0}) : {1}".format(desired_label, filtered_image.shape))

    return filtered_image
    
def shift_image(image):
    #apply shift operation
    transform = AffineTransform(translation=(25,25))
    wrapShift = warp(image,transform,mode='wrap')
   # io.imshow(wrapShift)
   # plt.title('Wrap Shift')
    return wrapShift
    
def flipLR(image):
    flipLR = np.fliplr(image)
    #io.imshow(flipLR)
    #plt.title('Left to Right Flipped')
    return flipLR
    
def addNoise(image, sigma):
    sigma=0.155
    noisyRandom = random_noise(image,var=sigma**2)
    #io.imshow(noisyRandom)
    #plt.title('Random Noise')
    return noisyRandom
    
def blurimage(image, sigma=1):
    #blur the image
    blurred = gaussian(image,sigma=1,multichannel=True)
    #plt.imshow(blurred)
    #plt.title('Blurred Image')
    return blurred
    
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  #io.imshow(result)
  return result
  
def apply_augmentation(imagesClassExtract, label):

    final_images = []
    final_labels = []
    for i in range(imagesClassExtract.shape[0]):
        final_images.append(imagesClassExtract[i])
        final_images.append(shift_image(imagesClassExtract[i]))
        final_images.append(np.fliplr(imagesClassExtract[i]))
        final_images.append(addNoise(imagesClassExtract[i], 0.155))
        final_images.append(random_noise(imagesClassExtract[i],var=0.2**2))
        final_images.append(blurimage(imagesClassExtract[i],1))
        final_images.append(rotate_image(imagesClassExtract[i], 10))

    augmented_images = np.array(final_images)
    augmented_labels = np.ones(augmented_images.shape[0]) * label

    print("Augmented Patch Images Shape\t:{0}\nAugmented Patch Labels Shape\t:{1}".format(augmented_images.shape, augmented_labels.shape))

    return augmented_images, augmented_labels