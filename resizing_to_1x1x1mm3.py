import SimpleITK as sitk
import numpy as np
import os
import glob
from collections import defaultdict

def resample_to_isotropic(image: sitk.Image, isotropic_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """
    Resample a SimpleITK image to isotropic voxel spacing (default: 1x1x1 mm).
    """
    original_size = np.array(image.GetSize(), dtype=np.int32)
    original_spacing = np.array(image.GetSpacing())

    new_size = (original_size * (original_spacing / np.array(isotropic_spacing))).astype(np.int32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(isotropic_spacing)
    resampler.SetSize([int(sz) for sz in new_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputPixelType(image.GetPixelID())

    return resampler.Execute(image)


if __name__ == '__main__':
    t2w_input = r'C:\DISSERTATION\Data for Project\Data_by_modality\t2w'
    other3modalities = r'C:\DISSERTATION\Data for Project\Data_by_modality\resampled_images'

    VOXEL_SIZED_FOLDER = r'C:\DISSERTATION\Data for Project\Data_by_modality\VOXEL_SIZED_FOLDER'
    os.makedirs(VOXEL_SIZED_FOLDER, exist_ok=True)

    modalities = ["t2w", "adc", "dwi", "lesion"]

    # Dictionary to store all files per patient and study
    patient_files = defaultdict(lambda: defaultdict(list))

    # ✅ FIRST: Collect **T2W** files from `t2w_input`
    for file_path in glob.glob(os.path.join(t2w_input, '*.nii.gz')):
        file_name = os.path.basename(file_path)
        name_parts = file_name.replace('.nii.gz', '').split('_')

        patient_id = name_parts[0]  # Example: "Patient001061633"
        study_number = name_parts[1] if "study" in name_parts[1] else "study_0"

        # Store T2W separately
        patient_files[patient_id][study_number].append(("t2w", "t2w", file_path))

    # ✅ SECOND: Collect **ADC, DWI, Lesion** files from `other3modalities`
    for file_path in glob.glob(os.path.join(other3modalities, '*.nii.gz')):
        file_name = os.path.basename(file_path)
        name_parts = file_name.replace('.nii.gz', '').split('_')

        patient_id = name_parts[0]
        study_number = name_parts[1] if "study" in name_parts[1] else "study_0"

        # Determine the modality (adc, dwi, lesion)
        modality = next((mod for mod in modalities if mod in file_name.lower()), None)
        sub_modality = "-".join([p for p in name_parts if modality in p]) if modality else ""

        if modality:
            patient_files[patient_id][study_number].append((modality, sub_modality, file_path))

    # ✅ Process each patient **ensuring T2W is processed before other modalities**
    for patient_id, studies in patient_files.items():
        for study_number, files in studies.items():
            # Sort files to **always process T2W first**
            files.sort(key=lambda x: (x[0] != "t2w", ["t2w", "adc", "dwi", "lesion"].index(x[0])))

            # ✅ Process **T2W first**
            for modality, sub_modality, file_path in files:
                print(f"Processing {modality.upper()} file: {file_path} for {patient_id} {study_number}")

                # Load image
                sitk_image = sitk.ReadImage(file_path)

                # Resample image
                resized_image = resample_to_isotropic(sitk_image, isotropic_spacing=(0.5, 0.5, 1.0))

                # Construct output filename
                output_file_name = f"{patient_id}_{study_number}_{sub_modality}_resized.nii.gz"
                output_path = os.path.join(VOXEL_SIZED_FOLDER, output_file_name)

                # Save the resampled image
                sitk.WriteImage(resized_image, output_path)
                print(f"Saved resized {modality.upper()} file to: {output_path}")
