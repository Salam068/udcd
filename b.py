import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

# Path to the folder containing .nii.gz label files
input_folder = '/media/salam/Salam/MSc/7262581/amos22/amos22/labelsTr/'

# Output folder to save PNG images
output_folder = '/media/salam/Salam/MSc/7262581/amos22/amos22/Data/Train/fold_1/masks/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Color palette for labels (from 0 to 15)
color_palette = {
    0: (0, 0, 0),        # background (black)
    1: (120, 70, 40),    # spleen (brownish)
    2: (80, 60, 100),    # right kidney (purplish)
    3: (130, 90, 60),    # left kidney (brownish)                                                                                        
    4: (80, 60, 150),    # gall bladder (purple-ish)
    5: (100, 80, 50),    # esophagus (brownish)
    6: (100, 70, 140),   # liver (purple-ish)
    7: (60, 100, 130),   # stomach (brownish)
    8: (100, 70, 160),   # aorta (purple-ish)
    9: (60, 80, 140),    # postcava (blue-ish)
    10: (80, 90, 110),   # pancreas (greyish purple)
    11: (100, 60, 80),   # right adrenal gland (dark purple)
    12: (80, 100, 120),  # left adrenal gland (greyish)
    13: (130, 70, 120),  # duodenum (pinkish)
    14: (90, 80, 100),   # bladder (brownish)
    15: (120, 100, 150), # prostate/uterus (dark purple)
}
from tqdm import tqdm

# Function to save slices as PNG (in PIL Palette format)
def save_slices_as_png(nii_file):
    # Load the .nii.gz file using SimpleITK
    img = sitk.ReadImage(nii_file)
    img_data = sitk.GetArrayFromImage(img)  # Get the image data as a numpy array (shape: [z, y, x])

    # Create a flat list of the RGB values from the color palette
    flat_palette = []
    for label, color in color_palette.items():
        flat_palette.extend(color)  # Add RGB components in sequence (R, G, B)

    # Iterate through each slice along the first dimension (axis 0) using tqdm for progress bar
    for slice_idx in tqdm(range(img_data.shape[0]), desc="Processing Slices", unit="slice"):
        # Extract the 2D slice (from the z dimension)
        slice_data = img_data[slice_idx, :, :]

        # Create a 2D image where pixel values represent labels
        label_image = Image.fromarray(slice_data.astype(np.uint8))

        # Convert the label image to 'P' mode (palette-based image)
        label_image = label_image.convert('P')

        # Set the custom color palette for this image
        label_image.putpalette(flat_palette)

        # Rotate and flip the image as per the requirement
        label_image = label_image.rotate(180, expand=True)
        label_image = label_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Define the filename for saving the slice
        base_filename = os.path.splitext(os.path.basename(nii_file))[0]  # Get filename without extension
        output_filename = f"{base_filename}_slice{slice_idx + 1}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Save the image as a PNG
        label_image.save(output_path)
        # print(f"Saved: {output_path}")


# Iterate through all .nii.gz files in the input folder
for nii_filename in os.listdir(input_folder):
    if nii_filename.endswith('.nii.gz'):
        nii_file_path = os.path.join(input_folder, nii_filename)
        print(f"Processing: {nii_file_path}")
        save_slices_as_png(nii_file_path)

# Print the color palette array
color_palette_array = np.array(list(color_palette.values()))
print("Color Palette (RGB):")
print(color_palette_array)

print("Processing completed!")
