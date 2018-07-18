import numpy as np
import random
from tensorlayer.layers import *
import os
from skimage.transform import resize
import tensorflow as tf
import random
from sklearn.preprocessing import normalize
import PIL as pil
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
from resizeimage import resizeimage
import tensorlayer as tl
from skimage.restoration import denoise_tv_chambolle
import regex as re
from numpy import inf
def get_unique_string(im_name):
    '''
    :param im_name: a string with the image file name
    :return: the unique identifier for that image
    '''
    im_parts = im_name.split("_")

    strings = []
    for parts in im_parts:
        if re.findall('\d+', parts):
            strings.append(re.findall('\d+', parts))

    unique_str = strings[0][0] + "_" + strings[1][0]
    return (unique_str)

def padding_with_zeros(im, orig_shape, new_shape):
    '''
    :param im:
    :param orig_shape:
    :param new_shape:
    :return:
    '''
#    im = im.reshape(orig_shape)
    result = np.zeros(new_shape)
    #print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((new_shape[0] - orig_shape[0])/2)  # 0 would be what you wanted
    y_offset = int((new_shape[1] - orig_shape[1])/2)  # 0 in your case
    #print(x_offset, y_offset)
    
    result[x_offset:im.shape[0]+x_offset,y_offset:im.shape[1]+y_offset] = im
    return(result)

def get_seg_data(im_dir, seg_dir, mask_dir, batch_size, pre_processed):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''

    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    mask_names = os.listdir(mask_dir)
    # random int for selecting images
    random_int = random.sample(range(0, len(im_names)), batch_size)
    # random_int = 1
    # set containers holding data
    images = []
    seg_maps = []
    masks = []
    im_id = []
    im_displayed = []
    # gather data
    k = 0
    for i in range(batch_size):
        im_name = im_names[0]#random_int[i]]
        unique_str = get_unique_string(im_name)
                # print("Just feeding same image")
        
        if batch_size > k:
            if "img" in im_name:
                k += 1
                #retrieve image
                train_im = np.loadtxt(im_dir + im_name)[0:160,0:400]
                # retrieve the corresponding seg_map
                y_path = [s for s in seg_names if unique_str in s]
                seg_im = np.loadtxt(seg_dir + y_path[0], dtype=np.int32)
                seg_im = seg_im[0:160, 0:400]

                # retrieve the corresponding masks
                mask_path = [s for s in mask_names if unique_str in s]
                mask_map = np.loadtxt(mask_dir + mask_path[0], dtype=np.float32)
                mask = 1 / mask_map * 100

                image_mean = np.mean(train_im)

                if (pre_processed == True):
                    im = Image.fromarray((train_im).astype('uint8'))
                    seg = Image.fromarray((seg_im))
                    mask_im = Image.fromarray((mask))

                    float_r = random.uniform(0.0, 1.0)
                    # 50 % chance that both im and seg is flipped
                    if float_r > 0.5:
                        im = ImageOps.mirror(im)
                        seg_im = np.asarray(ImageOps.mirror(seg))
                        mask = np.asarray(ImageOps.mirror(mask_im))

                    contrast = ImageEnhance.Contrast(im)
                    img_contr = contrast.enhance(3)
                    color = ImageEnhance.Color(img_contr)
                    img_contr = color.enhance(0.8)
                    brightness = ImageEnhance.Brightness(img_contr)
                    img_bright = brightness.enhance(2)
                    sharpness = ImageEnhance.Sharpness(img_bright)
                    img_sharp = sharpness.enhance(2)
                    train_im = np.asarray(img_sharp) / image_mean  # normalize
                    train_im = denoise_tv_chambolle(train_im, weight=0.5, multichannel=True).reshape(160,400)
                    #alpha = 34 and sigma =4 as in paper:  [Simard2003]
                    train_im = tl.prepro.elastic_transform(train_im, 34, 4)

                # append to list to return
                images.append(train_im)
                im_displayed.append(unique_str)
                seg_maps.append(seg_im)
                im_id.append(unique_str)
                masks.append(mask)

    # set shapes
    im_batch = np.reshape(np.asarray(images), (batch_size, 160, 400, 1))
    labels_batch = np.reshape(np.asarray(seg_maps), (batch_size, 160, 400, 1))
    masks_batch = np.reshape(np.asarray(masks), (batch_size, 160, 400, 1))

    # print("Number of images collected {}".format(k))
    return im_batch, labels_batch, im_displayed, masks_batch

def get_seg_data_gen_u_net(im_dir, seg_dir, mask_dir, batch_size, pre_processed=False):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''
    orig_shape = [160,400]
    new_shape = [320, 960]
    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    mask_names = os.listdir(mask_dir)
    # random int for selecting images
    random_int = random.sample(range(0, len(im_names)), batch_size)
    # random_int = 1
    # set containers holding data
    images = []
    seg_maps = []
    masks = []
    im_id = []
    im_displayed = []
    # gather data
    k = 0
    for i in range(batch_size):
        im_name = im_names[0]#random_int[i]]
        unique_str = get_unique_string(im_name)
        # print("Just feeding same image")

        if batch_size >= k:
            if "img" in im_name:
                k += 1
                # retrieve image
                train_im = np.loadtxt(im_dir + im_name)[0:160, 0:400]
                train_im = padding_with_zeros(train_im, orig_shape, new_shape)
                # retrieve the corresponding seg_map
                y_path = [s for s in seg_names if unique_str in s]
                seg_im = np.loadtxt(seg_dir + y_path[0], dtype=np.int32)
                seg_im = seg_im[0:160, 0:400]
                seg_im = padding_with_zeros(seg_im, orig_shape, new_shape)

                # retrieve the corresponding masks
                mask_path = [s for s in mask_names if unique_str in s]
                mask_map = np.loadtxt(mask_dir + mask_path[0], dtype=np.float32)
                mask_map = mask_map[0:160, 0:400]
                mask_map = padding_with_zeros(mask_map, orig_shape, new_shape)
                mask = 1 / mask_map * 100
                mask[mask == inf] = 0
                #scaling mask values
                masks_values = mask.reshape(-1)
                #derive values for normalization
                max_mask = max(masks_values)
                min_mask = min(masks_values)
                mean_mask = np.mean(masks_values)
                #scaling
                scaled = (masks_values - mean_mask) / (max_mask - min_mask)
                #setting one over to give higher values for border pixels

                image_mean = np.mean(train_im)

                if (pre_processed == True):
                    im = Image.fromarray((train_im).astype('uint8'))
                    seg = Image.fromarray((seg_im))
                    mask_im = Image.fromarray((mask))

                    float_r = random.uniform(0.0, 1.0)
                    # 50 % chance that both im and seg is flipped
                    if float_r > 0.5:
                        im = ImageOps.mirror(im)
                        seg_im = np.asarray(ImageOps.mirror(seg))
                        masks_values = np.asarray(ImageOps.mirror(mask_im))

                    contrast = ImageEnhance.Contrast(im)
                    img_contr = contrast.enhance(3)
                    color = ImageEnhance.Color(img_contr)
                    img_contr = color.enhance(0.8)
                    brightness = ImageEnhance.Brightness(img_contr)
                    img_bright = brightness.enhance(2)
                    sharpness = ImageEnhance.Sharpness(img_bright)
                    img_sharp = sharpness.enhance(2)
                    train_im = np.asarray(img_sharp) / image_mean  # normalize
                    train_im = denoise_tv_chambolle(train_im, weight=0.5, multichannel=True)#.reshape(160, 480)
                    # alpha = 34 and sigma =4 as in paper:  [Simard2003]
                    train_im = tl.prepro.elastic_transform(train_im, 34, 4)

                # append to list to return
                images.append(train_im)
                im_displayed.append(unique_str)
                seg_maps.append(seg_im)
                im_id.append(unique_str)
                masks.append(masks_values)

    # set shapes
    im_batch = np.reshape(np.asarray(images), (batch_size, 320, 960, 1))
    labels_batch = np.reshape(np.asarray(seg_maps), (batch_size, 320, 960,1))
    masks_batch = np.reshape(np.asarray(masks), (batch_size, 320, 960, 1))

    # print("Number of images collected {}".format(k))
    return im_batch, labels_batch, im_displayed, masks_batch


def get_seg_data_eval(im_dir, seg_dir, mask_dir, NUM_IMAGES, pre_processed):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''

    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    mask_names = os.listdir(mask_dir)
    # random int for selecting images
    # set containers holding data
    images = []
    seg_maps = []
    masks = []
    im_id = []
    im_displayed = []
    # gather data
    k = 0
    print(len(im_names))
    for im_name in im_names:
        unique_str = get_unique_string(im_name)
        if "img" in im_name:
            k += 1
            # retrieve image
            train_im = np.loadtxt(im_dir + im_name)[0:160, 0:400]
            # retrieve the corresponding seg_map
            y_path = [s for s in seg_names if unique_str in s]
            seg_im = np.loadtxt(seg_dir + y_path[0], dtype=np.int32)
            seg_im = seg_im[0:160, 0:400]

            # retrieve the corresponding masks
            mask_path = [s for s in mask_names if unique_str in s]
            mask_map = np.loadtxt(mask_dir + mask_path[0], dtype=np.float32)
            mask = 1 / mask_map * 100

            image_mean = np.mean(train_im)

            if (pre_processed == True):
                im = Image.fromarray((train_im).astype('uint8'))
                contrast = ImageEnhance.Contrast(im)
                img_contr = contrast.enhance(3)
                color = ImageEnhance.Color(img_contr)
                img_contr = color.enhance(0.8)
                brightness = ImageEnhance.Brightness(img_contr)
                img_bright = brightness.enhance(2)
                sharpness = ImageEnhance.Sharpness(img_bright)
                img_sharp = sharpness.enhance(2)
                train_im = np.asarray(img_sharp) / image_mean  # normalize
                train_im = denoise_tv_chambolle(train_im, weight=0.5, multichannel=True).reshape(160, 400)

            # append to list to return
            images.append(train_im)
            im_displayed.append(unique_str)
            seg_maps.append(seg_im)
            im_id.append(unique_str)
            masks.append(mask)
            print(np.asarray(images).shape)

    # set shapes
    im_batch = np.reshape(np.asarray(images), (NUM_IMAGES, 160, 400, 1))
    labels_batch = np.reshape(np.asarray(seg_maps), (NUM_IMAGES, 160, 400, 1))
    masks_batch = np.reshape(np.asarray(masks), (NUM_IMAGES, 160, 400, 1))

    # print("Number of images collected {}".format(k))
    return im_batch, labels_batch, im_displayed, masks_batch

def get_seg_data_gen_u_net_eval(im_dir, seg_dir, mask_dir, NUM_IMAGES
                                ,img_height,img_width,pre_processed=False):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''
    orig_shape = [160, 400]
    new_shape = [img_height, img_width]
    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    mask_names = os.listdir(mask_dir)
    # set containers holding data
    images = []
    seg_maps = []
    masks = []
    im_id = []
    im_displayed = []
    # gather data
    k = 0
    for im_name in im_names:
        if k>= NUM_IMAGES:
            print("Stop here")
            break
        print("load image {}".format(im_name))
        unique_str = get_unique_string(im_name)
        # print("Just feeding same image")
        if "img" in im_name:
            k += 1
            # retrieve image
            train_im = np.loadtxt(im_dir + im_name)[0:orig_shape[0], 0:orig_shape[1]]
            train_im = padding_with_zeros(train_im, orig_shape, new_shape)
            # retrieve the corresponding seg_map
            y_path = [s for s in seg_names if unique_str in s]
            seg_im = np.loadtxt(seg_dir + y_path[0], dtype=np.int32)
            seg_im = seg_im[0:160, 0:400]
            seg_im = padding_with_zeros(seg_im, orig_shape, new_shape)

            # retrieve the corresponding masks
            mask_path = [s for s in mask_names if unique_str in s]
            mask_map = np.loadtxt(mask_dir + mask_path[0], dtype=np.float32)
            mask_map = mask_map[0:160, 0:400]
            mask_map = padding_with_zeros(mask_map, orig_shape, new_shape)
            mask = (1 / mask_map )* 100
            mask[mask == inf] = 0
            # scaling mask values
            masks_values = mask.reshape(-1)
            # derive values for normalization
            max_mask = max(masks_values)
            min_mask = min(masks_values)
            mean_mask = np.mean(masks_values)
            # scaling
            masks_values = (masks_values - mean_mask) / (max_mask - min_mask)
            # setting one over to give higher values for border pixels

            image_mean = np.mean(train_im)

            if (pre_processed == True):
                im = Image.fromarray((train_im).astype('uint8'))

                contrast = ImageEnhance.Contrast(im)
                img_contr = contrast.enhance(3)
                color = ImageEnhance.Color(img_contr)
                img_contr = color.enhance(0.8)
                brightness = ImageEnhance.Brightness(img_contr)
                img_bright = brightness.enhance(2)
                sharpness = ImageEnhance.Sharpness(img_bright)
                img_sharp = sharpness.enhance(2)
                train_im = np.asarray(img_sharp) / image_mean  # normalize
                train_im = denoise_tv_chambolle(train_im, weight=0.5, multichannel=True)  # .reshape(160, 480)

            # append to list to return
            images.append(train_im)
            im_displayed.append(unique_str)
            seg_maps.append(seg_im)
            im_id.append(unique_str)
            masks.append(masks_values)

    # set shapes
    im_batch = np.reshape(np.asarray(images), (k, img_height, img_width, 1))
    labels_batch = np.reshape(np.asarray(seg_maps), (k, img_height, img_width, 1))
    # print("Number of images collected {}".format(k))
    return im_batch, labels_batch, im_displayed

def load_clinic_images(clinic_data_dir, img_width, img_height, pre_processed):
    '''
    :param clinic_data_dir: string: dir where clinic iamges are
    :param img_width:
    :param img_height:
    :return:
    '''
    clinic_images = os.listdir(clinic_data_dir)
    images = []
    images_padded = []
    image_names = []
    images_rezised = []
    size = [img_width, img_height]
    new_shape = [496, 512 * np.divide(img_width,img_height)] # 1280 = 2.5*512
    for clinic_image in clinic_images:
        #print(clinic_image)
        im = Image.open(clinic_data_dir + clinic_image).convert('L')
        im_array = np.array(im)
        orig_shape = [im_array.shape[0], im_array.shape[1]]
        new_shape = [im_array.shape[0], im_array.shape[1] * np.divide(img_width, img_height)]  # 1280 = 2.5*512
        im_padded = padding_with_zeros(im_array, orig_shape, new_shape)
        # im = np.array(im)
        im_resized = Image.fromarray(im_padded).resize(size, Image.ANTIALIAS)
        im_resized = np.array(im_resized)
        #print("size of image is {}".format(im_resized.shape))
        image_mean = np.mean(im_resized)
        #im_resized = denoise_tv_chambolle(im_resized, weight=3, multichannel=False).reshape(size)

        if (pre_processed == True):
            im = Image.fromarray(im_resized.astype('uint8'))
            contrast = ImageEnhance.Contrast(im)
            img_contr = contrast.enhance(3)
            color = ImageEnhance.Color(img_contr)
            img_contr = color.enhance(0.8)
            brightness = ImageEnhance.Brightness(img_contr)
            img_bright = brightness.enhance(2)
            sharpness = ImageEnhance.Sharpness(img_bright)
            img_sharp = sharpness.enhance(2)
            train_im = np.asarray(img_sharp) / image_mean  # normalize

        images.append(im_resized)
        image_names.append(clinic_image)

    return(images, image_names)

def load_clinic_images_gen_u_net(clinic_data_dir, img_width, img_height, pre_processed):
    '''
    :param clinic_data_dir: string: dir where clinic iamges are
    :param img_width:
    :param img_height:
    :return:
    '''
    orig_shape = [400, 160]
    new_shape = [img_width,img_height]
    clinic_images = os.listdir(clinic_data_dir)
    size = orig_shape
    images = []
    image_names = []
    for clinic_image in clinic_images:
        #print("The image loaded is {}".format(clinic_image))
        im = Image.open(clinic_data_dir + clinic_image).convert('L')
        # im = np.array(im)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized = np.array(im_resized)
        im_resized = padding_with_zeros(im_resized, orig_shape, new_shape)
        image_mean = np.mean(im_resized)
        if (pre_processed == True):
            im = Image.fromarray(im_resized.astype('uint8'))
            contrast = ImageEnhance.Contrast(im)
            img_contr = contrast.enhance(3)
            color = ImageEnhance.Color(img_contr)
            img_contr = color.enhance(0.8)
            brightness = ImageEnhance.Brightness(img_contr)
            img_bright = brightness.enhance(2)
            sharpness = ImageEnhance.Sharpness(img_bright)
            img_sharp = sharpness.enhance(2)
            train_im = np.asarray(img_sharp) / image_mean  # normalize
            im_resized = denoise_tv_chambolle(train_im, weight=0.5, multichannel=True).reshape(img_width,img_height)

        images.append(im_resized)
        image_names.append(clinic_image)

    return(images, image_names)


def get_clinic_train_data(im_dir, seg_dir, img_width, img_height,batch_size):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''

    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    # random int for selecting images
    random_int = np.random.choice(len(im_names),batch_size)
    # random_int = 1
    # set containers holding data
    images = []
    seg_maps = []
    im_id = []
    im_displayed = []
    # set sizes
    size = [img_width, img_height]
    # gather data
    k = 0
    for i in range(batch_size):
        im_name = im_names[random_int[i]]
        unique_str = get_unique_string(im_name)
        im_displayed.append(unique_str)
        # print("Just feeding same image")

        if batch_size > k:
            k += 1
            # retrieve image
            train_im = Image.open(im_dir + im_name).convert('L')
            train_im = np.array(train_im)
            orig_shape = [train_im.shape[0], train_im.shape[1]]
            new_shape = [train_im.shape[0], train_im.shape[1] * np.divide(img_width, img_height)]
            #print(new_shape, orig_shape)
            # print(train_im.shape)
            im_padded = padding_with_zeros(train_im, orig_shape, new_shape)
            # im = np.array(im)
            im_resized = Image.fromarray(im_padded).resize(size, Image.ANTIALIAS)

            # retrieve the labels
            print("The unique string is {}".format(unique_str))
            y_path = [s for s in seg_names if unique_str in s]
            seg_im = Image.open(seg_dir + y_path[0])  # .convert('L')
            seg_im = np.array(seg_im)
            seg_padded = padding_with_zeros(seg_im, orig_shape, new_shape)
            # im = np.array(im)
            seg_resized = Image.fromarray(seg_padded).resize(size)

            float_r = random.uniform(0.0, 1.0)
            # 50 % chance that both im and seg is flipped
            if float_r > 0.5:
                im_resized = ImageOps.mirror(im_resized)
                seg_resized = ImageOps.mirror(seg_resized)

            im_resized = np.array(im_resized)
            seg_resized = np.array(seg_resized)

            #invert the images with reversed colors
            if np.mean(im_resized) > 100:
                im_resized[np.where(im_resized < 100)] = 255
                im_resized[np.where(im_resized > 220)] = 0

            images.append(im_resized)
            seg_maps.append(seg_resized)

    im_batch = np.reshape(np.asarray(images), (batch_size, 160, 400, 1))
    labels_batch = np.reshape(np.asarray(seg_maps, dtype = np.int32), (batch_size, 160, 400, 1))

    return (im_batch, labels_batch, im_displayed)

def get_clinic_data_hardrive(im_dir, img_width, img_height):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''
    size = [img_width, img_height]
    train_im = Image.open(im_dir).convert('L')
    train_im = np.array(train_im)
    orig_shape = [train_im.shape[0], train_im.shape[1]]
    new_shape = [train_im.shape[0], train_im.shape[1] * np.divide(img_width, img_height)]
    # print(train_im.shape)
    im_padded = padding_with_zeros(train_im, orig_shape, new_shape)
    # im = np.array(im)
    im_resized = Image.fromarray(im_padded).resize(size, Image.ANTIALIAS)
    im_resized = np.array(im_resized)
    #invert the images with reversed colors
    if np.mean(im_resized) > 100:
        im_resized[np.where(im_resized < 100)] = 255
        im_resized[np.where(im_resized > 220)] = 0
    ###IMAGE READY FOR PREDICTION
    im_batch = np.reshape(im_resized, (1, 160, 400, 1))
    return im_batch, new_shape, orig_shape