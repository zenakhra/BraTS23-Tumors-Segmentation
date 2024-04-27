import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def get_direct_subfolders(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    return subfolders

def get_direct_subfiles(directory):
    subfiles = [f.path for f in os.scandir(directory) if f.is_file()]
    return subfiles

directory_path = r'D:\Brats21 Data\Dataset\training\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'  # Double backslashes in Windows path
subfolders = get_direct_subfolders(directory_path)
for subfolder in subfolders:
    print("Petient folder: ", subfolder)
    subfiles = get_direct_subfiles(subfolder)
    # paths:
    for nii_file_path in subfiles:
        # Check if the file ends with "seg.nii"
        if nii_file_path.endswith("seg.nii"):
            # Load the NIfTI image file
            nifti_img = nib.load(nii_file_path)
            # Get the image data as a NumPy array
            img_data = nifti_img.get_fdata()
            # Find the index of the frame with the most pixels above zero
            nonzero_counts = [np.count_nonzero(slice_img) for slice_img in img_data.transpose(2, 0, 1)]
            max_index = np.argmax(nonzero_counts)
            # Extract the frame with the most pixels above zero
            selected_frame = img_data[:, :, max_index]
            # Save the selected frame as a PNG image
            output_file = nii_file_path.replace('.nii', '_selected_frame.png')
            plt.imsave(output_file, selected_frame, cmap='gray')
            print(f"chosen saved frame {output_file}")
            # Display the selected frame
            print("Best slice max tumor surface: ")
            plt.imshow(selected_frame, cmap='gray')
            plt.title('Selected Frame')
            plt.axis('off')
            plt.show()
