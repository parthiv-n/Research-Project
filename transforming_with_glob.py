# import SimpleITK as sitk
# import numpy as np
# import os
# import glob
# from collections import defaultdict

# # Define functions (unchanged from your original script)
# def get_image_and_affine(image_path) -> list:
#     '''
#     Get the image and affine matrix from a SimpleITK image.
#     '''
#     sitk_image = sitk.ReadImage(image_path)
#     direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
#     origin = np.array(sitk_image.GetOrigin())
#     spacing = np.array(sitk_image.GetSpacing())

#     affine = np.identity(4)
#     affine[:3, :3] = direction * spacing[:, None]
#     affine[:3, 3] = origin

#     return sitk.GetArrayFromImage(sitk_image).transpose(1, 2, 0), affine

# def resample_to_isotropic(image: sitk.Image, isotropic_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
#     '''
#     Resample a SimpleITK image to isotropic voxel spacing (1x1x1 mm by default).
#     '''
#     original_size = np.array(image.GetSize(), dtype=np.int32)
#     original_spacing = np.array(image.GetSpacing())
    
#     new_size = (original_size * (original_spacing / np.array(isotropic_spacing))).astype(np.int32)
    
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(isotropic_spacing)
#     resampler.SetSize([int(sz) for sz in new_size])
#     resampler.SetOutputDirection(image.GetDirection())
#     resampler.SetOutputOrigin(image.GetOrigin())
#     resampler.SetOutputPixelType(image.GetPixelID())
    
#     return resampler.Execute(image)

# def resample_fromAtoB(target_image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
#     '''
#     Resample a SimpleITK image to a reference image. 
#     '''
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(reference_image.GetSpacing())
#     resampler.SetOutputDirection(reference_image.GetDirection())
#     resampler.SetOutputOrigin(reference_image.GetOrigin())
#     resampler.SetSize(reference_image.GetSize())
#     resampler.SetOutputPixelType(target_image.GetPixelID())
#     return resampler.Execute(target_image)

# def save_nifti(image, affine, output_path) -> None:
#     '''
#     Save a numpy array as a Nifti file.
#     '''
#     sitk_image = sitk.GetImageFromArray(image)
#     sitk_image.SetSpacing(affine[:3, :3].diagonal())
#     sitk_image.SetOrigin(affine[:3, 3])
#     sitk_image.SetDirection(affine[:3, :3].flatten())
#     sitk.WriteImage(sitk_image, output_path)

# def load_image(image_path):
#     """
#     Load a NIfTI image and return it as a NumPy array.
#     """
#     sitk_image = sitk.ReadImage(image_path)
#     return sitk.GetArrayFromImage(sitk_image)

# # Main processing script
# if __name__ == '__main__':
#     data_directory = r"C:\DISSERTATION\Data for Project\Data_by_modality"  # Replace with your directory path
#     output_directory = os.path.join(data_directory, "resampled_images")
#     os.makedirs(output_directory, exist_ok=True)

#     folders = ["lesion", "adc", "dwi", "t2w"]
#     files_dict = defaultdict(lambda: defaultdict(list))

#     # Collect all files per patient per modality
#     for folder in folders:
#         folder_path = os.path.join(data_directory, folder)
#         for file_path in glob.glob(os.path.join(folder_path, "*.nii.gz")):
#             file_name = os.path.basename(file_path)
#             prefix = file_name[:16]  # Extracting patient ID
#             files_dict[prefix][folder].append(file_path)  # Store all files per modality

#     # Process each patient
#     for prefix, files in files_dict.items():
#         if all(key in files for key in folders):
#             t2w_file_path = files["t2w"][0]  # Only one T2W file expected
            
#             # Load T2W as reference
#             t2w_originImage, t2w_affine = get_image_and_affine(t2w_file_path)
#             t2w_sitk = sitk.ReadImage(t2w_file_path)

#             # Resample all ADC files
#             for adc_file_path in files["adc"]:
#                 adc_originImage, adc_affine = get_image_and_affine(adc_file_path)
#                 adc_resampled = sitk.GetArrayFromImage(resample_fromAtoB(sitk.ReadImage(adc_file_path), t2w_sitk))
#                 save_nifti(adc_resampled.transpose(2, 0, 1), t2w_affine,
#                            os.path.join(output_directory, f"{os.path.basename(adc_file_path).replace('.nii.gz', '_resampled.nii.gz')}"))

#             # Resample all DWI files
#             for dwi_file_path in files["dwi"]:
#                 dwi_originImage, dwi_affine = get_image_and_affine(dwi_file_path)
#                 dwi_resampled = sitk.GetArrayFromImage(resample_fromAtoB(sitk.ReadImage(dwi_file_path), t2w_sitk))
#                 save_nifti(dwi_resampled.transpose(2, 0, 1), t2w_affine,
#                            os.path.join(output_directory, f"{os.path.basename(dwi_file_path).replace('.nii.gz', '_resampled.nii.gz')}"))

#             # Resample all Lesion files
#             for lesion_file_path in files["lesion"]:
#                 lesion_originImage, lesion_affine = get_image_and_affine(lesion_file_path)
#                 lesion_resampled = sitk.GetArrayFromImage(resample_fromAtoB(sitk.ReadImage(lesion_file_path), t2w_sitk))
#                 save_nifti(lesion_resampled.transpose(2, 0, 1), t2w_affine,
#                            os.path.join(output_directory, f"{os.path.basename(lesion_file_path).replace('.nii.gz', '_lesion_resampled.nii.gz')}"))

#             print(f"Processed all files for Patient {prefix}")

#     print("Processing complete.")


# # #How It Works
# # Manually set the file paths for T2W, ADC, DWI, and Lesion.
# # Load them with nibabel (.get_fdata() → NumPy array).
# # Identify all slices z in the lesion array that contain a nonzero voxel.
# # Build a figure with one row per z. Each row has 4 subplots: ADC, DWI, T2W, Lesion.
# # Save a single PNG with all those slices.

import SimpleITK as sitk
import os
import glob
from collections import defaultdict

def resample_to_isotropic(image: sitk.Image, isotropic_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """
    Resample a SimpleITK image to isotropic voxel spacing (1x1x1 mm by default).
    """
    original_size = list(image.GetSize())
    original_spacing = image.GetSpacing()
    # Compute new size using rounding
    new_size = [int(round(osz * osp / isp)) for osz, osp, isp in zip(original_size, original_spacing, isotropic_spacing)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(isotropic_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputPixelType(image.GetPixelID())
    
    return resampler.Execute(image)

def resample_fromAtoB(target_image: sitk.Image, reference_image: sitk.Image, interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    Resample a SimpleITK image (target_image) to the geometry of reference_image.
    The default interpolator is linear. For label images, pass interpolator=sitk.sitkNearestNeighbor.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputPixelType(target_image.GetPixelID())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(target_image)

def load_image(image_path):
    """
    Load a NIfTI image and return the SimpleITK image.
    """
    return sitk.ReadImage(image_path)

# Main processing script
if __name__ == '__main__':
    data_directory = r"C:\DISSERTATION\Data for Project\Data_by_modality"  # Replace with your directory path
    output_directory = os.path.join(data_directory, "resampled_images")
    os.makedirs(output_directory, exist_ok=True)

    folders = ["lesion", "adc", "dwi", "t2w"]
    files_dict = defaultdict(lambda: defaultdict(list))

    # Collect all files per patient per modality
    for folder in folders:
        folder_path = os.path.join(data_directory, folder)
        for file_path in glob.glob(os.path.join(folder_path, "*.nii.gz")):
            file_name = os.path.basename(file_path)
            prefix = file_name[:16]  # Extract patient ID (adjust if necessary)
            files_dict[prefix][folder].append(file_path)  # Store all files per modality

    # Process each patient
    for prefix, files in files_dict.items():
        # Check that all modalities are present
        if all(key in files for key in folders):
            # Use the T2 image as the reference geometry
            t2w_file_path = files["t2w"][0]  # Expect only one T2W file per patient
            t2w_image = sitk.ReadImage(t2w_file_path)

            # Resample ADC files using linear interpolation
            for adc_file_path in files["adc"]:
                adc_image = sitk.ReadImage(adc_file_path)
                adc_resampled = resample_fromAtoB(adc_image, t2w_image, interpolator=sitk.sitkLinear)
                adc_out_path = os.path.join(output_directory, 
                                f"{os.path.basename(adc_file_path).replace('.nii.gz', '_resampled.nii.gz')}")
                sitk.WriteImage(adc_resampled, adc_out_path)

            # Resample DWI files using linear interpolation
            for dwi_file_path in files["dwi"]:
                dwi_image = sitk.ReadImage(dwi_file_path)
                dwi_resampled = resample_fromAtoB(dwi_image, t2w_image, interpolator=sitk.sitkLinear)
                dwi_out_path = os.path.join(output_directory, 
                                f"{os.path.basename(dwi_file_path).replace('.nii.gz', '_resampled.nii.gz')}")
                sitk.WriteImage(dwi_resampled, dwi_out_path)

            # Resample Lesion files using nearest-neighbor interpolation
            for lesion_file_path in files["lesion"]:
                lesion_image = sitk.ReadImage(lesion_file_path)
                lesion_resampled = resample_fromAtoB(lesion_image, t2w_image, interpolator=sitk.sitkNearestNeighbor)
                lesion_out_path = os.path.join(output_directory, 
                                f"{os.path.basename(lesion_file_path).replace('.nii.gz', '_lesion_resampled.nii.gz')}")
                sitk.WriteImage(lesion_resampled, lesion_out_path)

            print(f"Processed all files for Patient {prefix}")

    print("Processing complete.")
