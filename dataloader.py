import torch
import torch.utils.data as tdata
import ujson
import numpy as np
import nibabel as nib
from pathlib import Path
import torchio as tio
import re
import matplotlib.pyplot as plt

# Define paths
data_dir = Path(r'C:\DISSERTATION\Data for Project\Data_by_modality\VOXEL_SIZED_FOLDER')
split_dir = Path(r'C:/DISSERTATION/Data for Project/Data_by_modality/split_patient_lists')

# Ensure split directory exists
split_dir.mkdir(parents=True, exist_ok=True)

train_json = split_dir / "train_patient_list.json"
val_json = split_dir / "val_patient_list.json"
test_json = split_dir / "test_patient_list.json"

class MyDataset(tdata.Dataset):
    def __init__(self, data_dir, phase):
        assert phase in ['train', 'val', 'test'], f"*** phase [{phase}] incorrect ***"
        self.phase = phase
        self.data_dir = Path(data_dir).expanduser().resolve()

        # Load patient IDs
        phase_json = {'train': train_json, 'val': val_json, 'test': test_json}[phase]
        print(f"Loading patient IDs from {phase_json}...")
        try:
            with open(phase_json, 'r') as f:
                self.patient_list = ujson.load(f)
            print(f"Loaded {len(self.patient_list)} patient IDs for {phase}.")
        except Exception as e:
            print(f"Error loading patient IDs: {e}")
            raise

        self.data_list = []
        for patient_id in self.patient_list:
            print(f"\nProcessing Patient {patient_id}")
            patient_files = sorted(self.data_dir.glob(f"Patient{patient_id}_study_*_resized.*"))
            print(f"Found files: {[str(f) for f in patient_files]}")

            # Store multiple files per modality
            modality_list = ['t2w', 'dwi', 'adc', 'lesion']
            studies = {m: [] for m in modality_list}  # Store all matching files

            for file in patient_files:
                for m in modality_list:
                    if m in file.stem.lower():
                        studies[m].append(file)  # Append all matching files

            # Debugging output for loaded studies
            print(f"Studies for Patient {patient_id}:")
            for m in modality_list:
                if studies[m]:
                    print(f"  {m}: {[f.name for f in studies[m]]}")
                else:
                    print(f"  {m}: None")

            # Ensure we have at least one valid file per modality
            if all(len(studies[m]) > 0 for m in modality_list):
                self.data_list.append((patient_id, studies))  # Store patient_id along with studies
                print("Patient added to data_list.")
            else:
                missing = [k for k, v in studies.items() if len(v) == 0]
                print(f"Skipping Patient {patient_id} - Missing: {missing}")

        if len(self.data_list) == 0:
            print(f"Error: No valid samples found in the {self.phase} set. Check your data paths.")
            raise ValueError(f"No valid samples found in the {self.phase} set.")

        print(f"\nTotal valid samples found in the {self.phase} set: {len(self.data_list)}")

    def load_nifti(self, file_path):
        img = nib.load(str(file_path)).get_fdata()
        img = np.nan_to_num(img)
        img = torch.tensor(img, dtype=torch.float32)
        return img

    def __getitem__(self, idx):
        _, entry = self.data_list[idx]
        

        # Select first available file from each list
        adc_img = self.load_nifti(entry['adc'][0])  
        dwi_img = self.load_nifti(entry['dwi'][0])  
        t2w_img = self.load_nifti(entry['t2w'][0])
        lesion_img = self.load_nifti(entry['lesion'][0])

        # Resize all images to (192, 192, 96)
        target_size = (192, 192, 96)
        resizer = tio.Resize(target_size)
        lesion_resizer = tio.Resize(target_size, image_interpolation='nearest')

        adc_img = resizer(adc_img.unsqueeze(0)).squeeze(0).unsqueeze(0)
        dwi_img = resizer(dwi_img.unsqueeze(0)).squeeze(0).unsqueeze(0)
        t2w_img = resizer(t2w_img.unsqueeze(0)).squeeze(0).unsqueeze(0)
        lesion_img = lesion_resizer(lesion_img.unsqueeze(0)).squeeze(0).unsqueeze(0)

        image_tensor = torch.cat([adc_img, dwi_img, t2w_img], dim=0)
        return image_tensor, lesion_img

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    train_dataset = MyDataset(data_dir, 'train')
    print(f"\nTraining samples: {len(train_dataset)}")


    # # Create a folder for debugging images
    # debugging_dir = Path(data_dir).parent / "debugging_images"
    # debugging_dir.mkdir(parents=True, exist_ok=True)

    # # Visualize first 595 patients
    # num_patients_to_visualize = 595
    # for patient_idx, (patient_id, image_tensor, label_tensor) in enumerate(train_dataset):
    #     if patient_idx >= num_patients_to_visualize:
    #         break

    #     adc_img = image_tensor[0]  
    #     dwi_img = image_tensor[1]  
    #     t2w_img = image_tensor[2]  
    #     lesion_img = label_tensor[0]  

    #     lesion_slices = [slice_idx for slice_idx in range(lesion_img.shape[2])
    #                      if torch.any(lesion_img[:, :, slice_idx] > 0)]
    #     num_slices = len(lesion_slices)
    #     if num_slices == 0:
    #         print(f"Skipping patient {patient_id} - No visible lesion in slices.")
    #         continue

    #     cols = 4  
    #     rows = num_slices
    #     fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    #     if rows == 1:
    #         axs = axs[np.newaxis, :]

    #     for row, slice_idx in enumerate(lesion_slices):
    #         axs[row, 0].imshow(adc_img[:, :, slice_idx], cmap='gray')
    #         axs[row, 0].set_title(f'ADC - Slice {slice_idx}')
    #         axs[row, 1].imshow(dwi_img[:, :, slice_idx], cmap='gray')
    #         axs[row, 1].set_title(f'DWI - Slice {slice_idx}')
    #         axs[row, 2].imshow(t2w_img[:, :, slice_idx], cmap='gray')
    #         axs[row, 2].set_title(f'T2W - Slice {slice_idx}')
    #         axs[row, 3].imshow(lesion_img[:, :, slice_idx], cmap='jet')
    #         axs[row, 3].set_title(f'Lesion Mask - Slice {slice_idx}')
    #         for ax in axs[row]:
    #             ax.axis('off')

    #     plt.tight_layout()
    #     patient_image_path = debugging_dir / f'patient_{patient_id}_lesion_slices.png'
    #     plt.savefig(str(patient_image_path), dpi=200)
    #     plt.close(fig)
    #     print(f'Saved visualization for patient {patient_id} at {patient_image_path}')
