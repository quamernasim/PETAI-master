from os.path import join as pjoin
import json
import os
import imageio
import numpy as np
from sklearn.preprocessing import MinMaxScaler

images = []
labels = []
images_valid = []
labels_valid = []
images_test = []
labels_test = []
countTrain = 0
countTest = 0

for split in ['train', 'valid']:

    path = 'tiles_inlines'+ '/' +'splits'+ '/'+ 'patch_' + split + '.txt'

    rootdir = pjoin('tiles_inlines', split)

    with open(pjoin('tiles_inlines', split, 'labels.json'), 'r') as json_file:
        data = json.load(json_file)

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:

            path_name = str(os.path.join(subdir, file))
            file_name = path_name.split('/')[2]
            extension_type = file_name.split('.')[1]

            if extension_type == 'png':
                remove_file_name = file_name.split('_')[1].split('.')[0]
                leading_removed = [s.lstrip("0") for s in remove_file_name]
                if leading_removed.count('') == 8:
                    new_string = '0000'
                elif leading_removed.count('') == 7:
                    new_string = '000'+ ''.join(leading_removed)
                elif leading_removed.count('') == 6:
                    new_string = '00'+ ''.join(leading_removed)
                elif leading_removed.count('') == 5:
                    new_string = '0'+ ''.join(leading_removed)
                elif leading_removed.count('') <= 4:
                    new_string = ''.join(leading_removed)

                im = imageio.imread(path_name)
                img = np.asarray(im)

                new_image = "image_"+ new_string
                arr= np.zeros((40, 40))
                arr[:, :] = float(data[new_image])
                lbl = arr

                if lbl[1, 1] == 4:
                    continue
                elif lbl[1, 1] == 6:
                    continue
                elif lbl[1, 1] == 5:
                    continue
                else:
                    if split == 'train':
                        countTrain += 1
                        if countTrain%5 == 0:
                            labels_valid.append(lbl)
                            images_valid.append(img)
                        else:
                            labels.append(lbl)
                            images.append(img)
                    else:
                        countTest += 1
                        labels_test.append(lbl)
                        images_test.append(img)

images = np.asarray(images)
labels = np.asarray(labels).astype('uint8')
images_valid = np.asarray(images_valid)
labels_valid = np.asarray(labels_valid).astype('uint8')
images_test = np.asarray(images_test)
labels_test = np.asarray(labels_test).astype('uint8')

print('Total Number of examples in Training Folder (Desired Examples Only) is {0}, out of which Training Set is having {1} and Validation Set' 
            ' is having {2}, so split size is {3}%'.format(countTrain, images.shape[0], images_valid.shape[0], (np.asarray(images_valid.shape[0]/countTrain)*100).round(2)))
print('Total Number of examples in Test Folder (Desired Examples Only) is {0}'.format(countTest))

#Normalizing from -1 to 1
scaler = MinMaxScaler(feature_range = (-1, 1))
scaled = scaler.fit(images.reshape((images.shape[0], -1)))

images = scaled.transform(images.reshape((images.shape[0], -1))).reshape((images.shape[0], 40, 40))
images_valid = scaled.transform(images_valid.reshape((images_valid.shape[0], -1))).reshape((images_valid.shape[0], 40, 40))
images_test = scaled.transform(images_test.reshape((images_test.shape[0], -1))).reshape((images_test.shape[0], 40, 40))

print('Unique Elements in Training Set (Brfore Transformation) = {0}, each having {1} images'.format(np.unique(np.asarray(labels), return_counts = True)[0],
                                                                                                     np.unique(np.asarray(labels), return_counts = True)[1]/40/40))
print('Unique Elements in Validation Set (Brfore Transformation) = {0}, each having {1} images'.format(np.unique(np.asarray(labels_valid), return_counts = True)[0],
                                                                                                       np.unique(np.asarray(labels_valid), return_counts = True)[1]/40/40))
print('Unique Elements in Testing Set (Brfore Transformation) = {0}, each having {1} images'.format(np.unique(np.asarray(labels_test), return_counts = True)[0],
                                                                                                    np.unique(np.asarray(labels_test), return_counts = True)[1]/40/40))

labels = np.where(labels == 3, 5, labels)
labels = np.where(labels == 2, 4, labels)
labels = np.where(labels == 1, 3, labels)

labels_valid = np.where(labels_valid == 3, 5, labels_valid)
labels_valid = np.where(labels_valid == 2, 4, labels_valid)
labels_valid = np.where(labels_valid == 1, 3, labels_valid)

labels_test = np.where(labels_test == 3, 5, labels_test)
labels_test = np.where(labels_test == 2, 4, labels_test)
labels_test = np.where(labels_test == 1, 3, labels_test)

print('\nUnique Elements in Training Set (After Transformation) = {0}, each having {1} images'.format(np.unique(np.asarray(labels), return_counts = True)[0],
                                                                                                     np.unique(np.asarray(labels), return_counts = True)[1]/40/40))
print('Unique Elements in Validation Set (After Transformation) = {0}, each having {1} images'.format(np.unique(np.asarray(labels_valid), return_counts = True)[0],
                                                                                                       np.unique(np.asarray(labels_valid), return_counts = True)[1]/40/40))
print('Unique Elements in Testing Set (After Transformation) = {0}, each having {1} images'.format(np.unique(np.asarray(labels_test), return_counts = True)[0],
                                                                                                    np.unique(np.asarray(labels_test), return_counts = True)[1]/40/40))

print('Mean of Training images = {}'.format(np.mean(images)))
print('Mean of Validation images = {}'.format(np.mean(images_valid)))
print('Mean of Testing images = {}'.format(np.mean(images_test)))

np.save(pjoin('data_canada', 'train', 'images.npy'), images)
np.save(pjoin('data_canada', 'train', 'labels.npy'), labels)
np.save(pjoin('data_canada', 'valid', 'images.npy'), images_valid)
np.save(pjoin('data_canada', 'valid', 'labels.npy'), labels_valid)
np.save(pjoin('data_canada', 'test', 'images.npy'), images_test)
np.save(pjoin('data_canada', 'test', 'labels.npy'), labels_test)
print('Saved!')