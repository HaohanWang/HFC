__author__ = 'Haohan Wang'

import numpy as np
from scipy import signal

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0.5

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def distance2(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 2.0

def mask_radial2(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance2(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

if __name__ == '__main__':
    import sys
    version = sys.version_info
    import pickle
    def load_datafile(filename):
      with open(filename, 'rb') as fo:
          if version.major == 3:
              data_dict = pickle.load(fo, encoding='bytes')
          else:
              data_dict = pickle.load(fo)

          assert data_dict[b'data'].dtype == np.uint8
          image_data = data_dict[b'data']
          image_data = image_data.reshape(
              (10000, 3, 32, 32)).transpose(0, 2, 3, 1)
          return image_data, np.array(data_dict[b'labels'])

    eval_images, eval_labels = load_datafile('../data/CIFAR10/test_batch')

    np.save('../data/CIFAR10/test_data_regular', eval_images)

    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]

    train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
    train_labels = np.zeros(50000, dtype='int32')
    for ii, fname in enumerate(train_filenames):
        cur_images, cur_labels = load_datafile('../data/CIFAR10/'+fname)
        train_images[ii * 10000: (ii+1) * 10000, ...] = cur_images
        train_labels[ii * 10000: (ii+1) * 10000, ...] = cur_labels

    print train_images.shape, train_labels.shape


    np.save('../data/CIFAR10/train_images', train_images)
    np.save('../data/CIFAR10/train_label', train_labels)

    train_image_low_4, train_image_high_4 = generateDataWithDifferentFrequencies_3Channel(train_images, 4)
    np.save('../data/CIFAR10/train_data_low_4', train_image_low_4)
    np.save('../data/CIFAR10/train_data_high_4', train_image_high_4)

    train_image_low_8, train_image_high_8 = generateDataWithDifferentFrequencies_3Channel(train_images, 8)
    np.save('../data/CIFAR10/train_data_low_8', train_image_low_8)
    np.save('../data/CIFAR10/train_data_high_8', train_image_high_8)

    train_image_low_12, train_image_high_12 = generateDataWithDifferentFrequencies_3Channel(train_images, 12)
    np.save('../data/CIFAR10/train_data_low_12', train_image_low_12)
    np.save('../data/CIFAR10/train_data_high_12', train_image_high_12)

    train_image_low_16, train_image_high_16 = generateDataWithDifferentFrequencies_3Channel(train_images, 16)
    np.save('../data/CIFAR10/train_data_low_16', train_image_low_16)
    np.save('../data/CIFAR10/train_data_high_16', train_image_high_16)

    eval_image_low_4, eval_image_high_4 = generateDataWithDifferentFrequencies_3Channel(eval_images, 4)
    np.save('../data/CIFAR10/test_data_low_4', eval_image_low_4)
    np.save('../data/CIFAR10/test_data_high_4', eval_image_high_4)

    eval_image_low_8, eval_image_high_8 = generateDataWithDifferentFrequencies_3Channel(eval_images, 8)
    np.save('../data/CIFAR10/test_data_low_8', eval_image_low_8)
    np.save('../data/CIFAR10/test_data_high_8', eval_image_high_8)

    eval_image_low_12, eval_image_high_12 = generateDataWithDifferentFrequencies_3Channel(eval_images, 12)
    np.save('../data/CIFAR10/test_data_low_12', eval_image_low_12)
    np.save('../data/CIFAR10/test_data_high_12', eval_image_high_12)

    eval_image_low_16, eval_image_high_16 = generateDataWithDifferentFrequencies_3Channel(eval_images, 16)
    np.save('../data/CIFAR10/test_data_low_16', eval_image_low_16)
    np.save('../data/CIFAR10/test_data_high_16', eval_image_high_16)

    eval_image_low_20, eval_image_high_20 = generateDataWithDifferentFrequencies_3Channel(eval_images, 20)
    np.save('../data/CIFAR10/test_data_low_20', eval_image_low_20)
    np.save('../data/CIFAR10/test_data_high_20', eval_image_high_20)

    eval_image_low_24, eval_image_high_24 = generateDataWithDifferentFrequencies_3Channel(eval_images, 24)
    np.save('../data/CIFAR10/test_data_low_24', eval_image_low_24)
    np.save('../data/CIFAR10/test_data_high_24', eval_image_high_24)

    eval_image_low_28, eval_image_high_28 = generateDataWithDifferentFrequencies_3Channel(eval_images, 28)
    np.save('../data/CIFAR10/test_data_low_28', eval_image_low_28)
    np.save('../data/CIFAR10/test_data_high_28', eval_image_high_28)