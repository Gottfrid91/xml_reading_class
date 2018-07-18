from model import *
import os
# path is the path retrieved from iterating the whole directors
path = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/augen_clinic_data/image_data/'
number_of_files = 0
number_of_HD_files = 0

patients = os.listdir(path)
for patient in patients:
    starting_data_loading_patient = datetime.now()
    patient_dir = os.path.join(path, patient)
    study_dates = os.listdir(patient_dir)
    study_start_time = datetime.now()

    for study_date in study_dates:
        study_dir = os.path.join(patient_dir, study_date)
        '''
        Below script takes a path to an xml containg data of the left and right eye of on study.
        From the path and xml it produces a list of paths to all oct images that can be used to
        load and oct and predict segmentation from . These segmenattions can then be used to
        derive the depth vector which again in this script will be mapped to a HD depth map
        using the scale and dimension information contained from each study.
        '''
        #make sure study directory contains a Volume folder/set
        study_folders = os.listdir(study_dir)
        for folders in study_folders:
            if "Volume" in folders:
                study_dir = os.path.join(patient_dir, study_date)
                # add standard name for xml to path
                lateralities_path = os.path.join(study_dir, "Volume", "LOCALIZER")
                Lateralities = os.listdir(lateralities_path)

                for Laterality in Lateralities:
                    localizer_path = os.path.join(study_dir, "Volume", "LOCALIZER", Laterality)
                    loc_dates = os.listdir(localizer_path)
                    for loc_date in loc_dates:
                        loc_date_path = os.path.join(localizer_path, loc_date)

                        loc_files = os.listdir(loc_date_path)
                        for loc_file in loc_files:
                            number_of_files = number_of_files + 1
                            if number_of_files % 10000 == 0:
                                print("number of files processed: {}".format(number_of_files))
                            if "HD" in loc_file:
                                number_of_HD_files = number_of_HD_files + 1
                                print("number_of_HD_files: {}".format(number_of_HD_files))
                                print("HD map predicted for: {}".format(study_dir))


