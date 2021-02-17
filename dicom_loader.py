import pydicom
from pydicom.filereader import read_dicomdir
import re
import os
import numpy as np


def dicom_from_dicomdir(dicom_dir):

    dataset_path = os.path.dirname(os.path.abspath(dicom_dir.filename))  # abs path without the final dicomdir
    for patient_record in dicom_dir.patient_records:
        studies = patient_record.children
        for study in studies:
            all_series = study.children
            for series in all_series:
                image_records = series.children
                # load data only from the series *301???.dcm or *3001???.dcm
                if bool(re.match("\w*30{1,2}1[0-9]{2,}\.dcm", image_records[0].ReferencedFileID)):
                    image_filenames = [
                        image_rec.ReferencedFileID for image_rec in image_records
                    ]
                    datasets = [pydicom.dcmread(os.path.join(dataset_path, basename)) for basename in image_filenames]
                    # raw data stacked together
                    volume = np.stack([images.pixel_array for images in datasets])
                    return image_filenames, datasets, volume
            raise Exception('No valid series found, abort!')
    raise Exception('no valid patient or study found in the path, abort!')