from input_data import *
from model import *
import os
import numpy as np
import pandas as pd
import depth_vector as dv
import xml_info_retrieval as xir

img_height = 160
img_width = 400

logging_dir = '/home/olle/PycharmProjects/segmentation_OCT/logging_u_net_no_preprocc_dice/'


def unpadding_with_zeros(im, orig_shape, new_shape, batch_size):
    '''
    :param im:
    :param orig_shape:
    :param new_shape:
    :return:
    '''
    #    im = im.reshape(orig_shape)
    result = np.zeros(orig_shape)
    # print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((new_shape[0] - orig_shape[0]) / 2)  # 0 would be what you wanted
    y_offset = int((new_shape[1] - orig_shape[1]) / 2)  # 0 in your case
    # print(x_offset, y_offset)
    result = im[x_offset:im.shape[0] - x_offset, y_offset:im.shape[1] - y_offset]
    return (result)


# input placeholders
X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, None], name='X')
y = tf.placeholder(tf.float32, shape=[None, img_height, img_width, None], name='y')

logits = u_net(X)
print("logits shape is {}".format(logits.get_shape()))

# create session
session = tf.Session()
# init global vars
init = tf.global_variables_initializer()
# preidction
prediction = tf.argmax(tf.nn.softmax(logits), axis=1)
probability_map = tf.nn.softmax(logits)
# correct prediction
correct_prediction = tf.equal(prediction, tf.cast(tf.reshape(y, prediction.get_shape(), name=None), tf.int64))
# accuracy
accuracy_c = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Counter for total number of iterations performed so far.# Counte
total_iterations = 0
# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()
# Start the queue runners.
tf.train.start_queue_runners(sess=session)
# set logging to specific location
summary_writer = tf.summary.FileWriter(logging_dir, session.graph)
# Create a saver.
saver = tf.train.Saver(tf.global_variables())

step_start = 0
try:
    ####Trying to find last checkpoint file fore full final model exist###
    print("Trying to restore last checkpoint ...")
    save_dir = logging_dir
    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
    # get the step integer from restored path to start step from there
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    # create init op for the still unitilized variables
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    session.run(init_new_vars_op)
except:
    # If all the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore any checkpoints. Initializing variables instead.")
    session.run(init)
# bookkeep times
prediction_time = []
patient_time = []
data_loading_time = []
study_time = []
# path is the path retrieved from iterating the whole directors
path = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/augen_clinic_data/image_data/'
# set index in patient list to not re-predict on allready predicted depth maps
patients = os.listdir(path)
for patient in patients:
        starting_data_loading_patient = datetime.now()
        print("Pateint predict is now: {}".format(patient))
        patient_dir = os.path.join(path, patient)
        study_dates = os.listdir(patient_dir)
        study_start_time = datetime.now()
        for study_date in study_dates:
            study_date = ",".join(study_date).replace(",", " ")
            patient = study_date.split("/")[-2]
            patient_dir = "/".join(study_date.split("/")[0:-2])
            print("Pateint predict is now: {}".format(patient))
            # for study_date in study_dates:
            study_dir = os.path.join(patient_dir, study_date)
            '''
            Below script takes a path to an xml containg data of the left and right eye of on study.
            From the path and xml it produces a list of paths to all oct images that can be used to
            load and oct and predict segmentation from . These segmenattions can then be used to
            derive the depth vector which again in this script will be mapped to a HD depth map
            using the scale and dimension information contained from each study.
            '''
            # make sure study directory contains a Volume folder/set
            study_folders = os.listdir(study_dir)
            for folders in study_folders:
                if "Volume" in folders:
                    study_dir = os.path.join(patient_dir, study_date)
                    print(study_dir)
                    # add standard name for xml to path
                    study_path = os.path.join(study_dir, 'anom_explore_corrected_url.xml')
                    # get files in the study dir
                    study_files = os.listdir(study_dir)
                    for study_file in study_files:
                        # do not proceed in case no xml file exist
                        if "anom_explore_corrected_url" in study_file:
                            # check which lateralities are present
                            lateralities_path = os.path.join(study_dir, "Volume", "LOCALIZER")
                            # in xml are considered
                            Lateralities = os.listdir(lateralities_path)
                            for Laterality in Lateralities:
                                # make tree object and retrieve im_pd
                                OCT_pos_pixel, grid, im_pd, stop_function = xir.get_study_xml(study_path)
                                # check if stop function has been triggered
                                if stop_function == "yes":
                                    break
                                if OCT_pos_pixel["Laterality"].isin([Laterality]).any():
                                    # path to the fundus images
                                    fundus_path = os.path.join(lateralities_path, Laterality)
                                    # times of fundus photo
                                    fundus_times = pd.Series(os.listdir(os.path.join(lateralities_path, Laterality)))
                                    fundus_times = fundus_times[fundus_times.isin(im_pd.Image_aq_time)]
                                    # iterate through the fundus images for the current study and laterality
                                    for fundus_time in fundus_times:
                                        x_cord, y_cord, x_start, y_start, x_end, y_end, y_indices, x_indices = xir.get_xml_indices(
                                            Laterality, OCT_pos_pixel, fundus_time, im_pd)
                                        # derive paths to oct's and localizers
                                        oct_batch_path = os.path.join(study_dir, "Volume", "OCT", Laterality)
                                        localizer_path = os.path.join(study_dir, "Volume", "LOCALIZER", Laterality)
                                        localizer_timestamp = os.listdir(localizer_path)
                                        localizer_im_path = os.path.join(localizer_path, fundus_time)
                                        # retrieve all OCT_slice aquisition times
                                        oct_times = os.listdir(oct_batch_path)
                                        oct_times_pd = pd.Series(oct_times)
                                        # get fundus series id
                                        series_id = im_pd[im_pd.Image_aq_time == fundus_time].series.values[0]

                                        # get all oct slices present in folder and xml based on aquisition time and series
                                        oct_times_in_folder_and_xml = oct_times_pd[
                                            pd.Series(oct_times).isin(im_pd[im_pd.series == series_id].Image_aq_time)]
                                        # list to gather oct slices for one fundus
                                        oct_slice = []
                                        oct_names = []
                                        for oct_time in oct_times_in_folder_and_xml:
                                            image_time_path = os.path.join(oct_batch_path, oct_time)
                                            oct_images = os.listdir(image_time_path)
                                            # get all oct slice paths for this fundus pic
                                            oct_per_fundus = []
                                            for oct_image in oct_images:
                                                if "DV" not in oct_image and "seg" not in oct_image and "Thumb" not in oct_image:
                                                    oct_path = os.path.join(image_time_path, oct_image)
                                                    oct_per_fundus.append(oct_path)

                                            # set all pixels for depth vectors to 1
                                            # gather all Oct slices in oct_slice list
                                            start_data_loading = datetime.now()
                                            for oct_path in oct_per_fundus:
                                                oct_names.append(oct_path.split("/")[-1])
                                                im, new_shape, orig_shape = get_clinic_data_hardrive(oct_path,
                                                                                                     img_width,
                                                                                                     img_height)
                                                oct_slice.append(im)
                                            end_data_loading = datetime.now()
                                            data_loading_time.append(abs(
                                                (end_data_loading.microsecond - start_data_loading.microsecond)) / 1e6)
                                        # prepare all images for segmentation
                                        image_array = np.asarray(oct_slice).reshape(len(oct_slice), img_height,
                                                                                    img_width, 1)

                                        # list to gather all segmentations
                                        predictions = []
                                        start_prediction = datetime.now()
                                        for i in range(0, image_array.shape[0]):
                                            # segment every oct slice and gather segmentations in fundus
                                            feed_dict_train = {X: image_array[i].reshape(1, 160, 400, 1)}
                                            # predict
                                            pred, prob_map = session.run([prediction, probability_map],
                                                                         feed_dict=feed_dict_train)
                                            predictions.append(pred)
                                        # ensure correct shape and type of the segmentations
                                        end_prediction = datetime.now()
                                        prediction_time.append(
                                            abs((end_prediction.microsecond - start_prediction.microsecond)) / 1e6)
                                        pred_array = np.asarray(predictions).reshape(len(oct_slice), img_height,
                                                                                     img_width).astype(np.int32)
                                        # print(pred_array.shape[0])
                                        # reshape the segmentations to original shape
                                        im_batch = []
                                        # resize all images
                                        for i in range(0, pred_array.shape[0]):
                                            im_resized = np.asarray(Image.fromarray(pred_array[i]).resize(
                                                [int(new_shape[1]), int(new_shape[0])]))
                                            im_resized = unpadding_with_zeros(im_resized, orig_shape, new_shape,
                                                                              batch_size)
                                            im_batch.append(im_resized)
                                        starting_data_loading_patient = datetime.now()
                                        im_batch = np.asarray(im_batch, dtype=np.int32).reshape(len(oct_slice),
                                                                                                orig_shape[0],
                                                                                                orig_shape[1])

                                        for i in range(0, im_batch.shape[0]):
                                            d_v = dv.get_depth_vector(im_batch[i])
                                            # print(d_v.shape)
                                            # extract the id of the image for positional arguments in y_indices
                                            im_id = int(re.findall(r'\d+', oct_names[i])[0])
                                            # print(im_id)
                                            try:
                                                if y_cord == "iterable":
                                                    # assert indices are ints
                                                    x_start = int(x_start)
                                                    x_end = int(x_end)
                                                    # assert d_v has same width as x_end -x_start
                                                    if d_v.shape[0] > (x_end - x_start):
                                                        d_v = d_v[0:x_end - x_start]
                                                    if d_v.shape[0] < (x_end - x_start):
                                                        difference = (x_end - x_start) - d_v.shape[0]
                                                        d_v = np.append(d_v, np.zeros(int(difference)))
                                                    # shift indices when laterilty changes to "L"
                                                    if Laterality == "L":
                                                        grid[int(y_indices["starty_pos"].values[im_id - 1]),
                                                        x_start:x_end] = d_v
                                                    if Laterality == "R":
                                                        grid[int(y_indices["starty_pos"].values[im_id - 1]),
                                                        x_start:x_end] = d_v
                                                if x_cord == "iterable":
                                                    # assert indices are ints
                                                    y_start = int(y_start)
                                                    y_end = int(y_end)
                                                    # assert d_v has same width as x_end -x_start
                                                    if d_v.shape[0] > (y_end - y_start):
                                                        d_v = d_v[0:y_end - y_start]
                                                    if d_v.shape[0] < (y_end - y_start):
                                                        difference = (y_end - y_start) - d_v.shape[0]
                                                        d_v = np.append(d_v, np.zeros(difference))
                                                    # shift indices when laterilty changes to "L"
                                                    if Laterality == "L":
                                                        grid[y_start:
                                                        y_end, int(x_indices["startx_pos"].values[im_id - 1])] = d_v
                                                    if Laterality == "R":
                                                        grid[y_start:
                                                        y_end, int(x_indices["startx_pos"].values[im_id - 1])] = d_v
                                            except:
                                                print("COULD NOT CALCULATE GRID")
                                                print(im_id, d_v.shape, x_start, x_end, x_end - x_start)
                                                print("The study dir is {}".format(study_dir))
                                        # linearly interpolate missing values
                                        grid[grid == 0] = np.nan
                                        grid_pd_int = pd.DataFrame(grid).interpolate(limit_direction='both')
                                        grid_pd_int = grid_pd_int.fillna(0)

                                        np.save(localizer_im_path + "/HD_depth_map", grid_pd_int)

                        study_end_time = datetime.now()
                        study_time.append(abs((study_end_time.microsecond - study_start_time.microsecond)) / 1e6)

                        end_data_loading_patient = datetime.now()
                        patient_time.append(abs(
                            (end_data_loading_patient.microsecond - starting_data_loading_patient.microsecond)) / 1e6)
                        format_str = (
                        'Total processing time per patient is %.2f, data load = %.2f, prediction time = %.2f, study_time = %.2f')
                        # take averages of all the accuracies from the previous bathces
                        print(format_str % (np.mean(patient_time), np.mean(data_loading_time), np.mean(prediction_time),
                                            np.mean(study_time)))
