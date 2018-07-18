from model import *
import os
# path is the path retrieved from iterating the whole directors
path = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/augen_clinic_data/image_data/'
number_of_files = 0
number_of_HD_files = 0

patients = os.listdir(path)
print("THE LIST ORDER OF THE PATIENTS IS:")
print(patients)
