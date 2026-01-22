import nibabel as nib 
from PIL import Image
import numpy as np 
import os
from pathlib import Path
from tqdm import tqdm

def convert_to_2d(nii_file, axis=2):
    """Convert a 3D NIfTI file to a series of 2D slices along the specified axis."""
    img = nib.load(nii_file)
    data = img.get_fdata()
    # Check if the specified axis is valid if axis < 0 or axis >= data.ndim: raise ValueError(f"Invalid axis {axis} for data with {data.ndim} dimensions.")
    # Generate 2D slices
    if data.ndim == 4:
        if data.shape[-1] == 4:
            data = data[:, :, :, 2]
        elif data.shape[-1] == 2:
            data = data[:, :, :, 0]
            
    slices = []
    for i in range(data.shape[axis]):
        if axis == 0:
            slice_2d = data[i, :, :]
        elif axis == 1:
            slice_2d = data[:, i, :]
        else:  # axis == 2
            slice_2d = data[:, :, i]
        slices.append(slice_2d)
    
    return slices

def normalize_image_slice(slice_2d):
    """Normalize image slice to 0-255 range."""
    # Handle different intensity ranges
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()
    
    if slice_max > slice_min:
        normalized = (slice_2d - slice_min) / (slice_max - slice_min) * 255
    else:
        normalized = np.zeros_like(slice_2d)
    
    return normalized.astype(np.uint8)

def scale_mask_for_visualization(mask_slice):
    """Scale mask values for better visualization while preserving class labels."""
    # For masks with small integer values (e.g., 0, 1, 2), scale them up
    # This makes them visible when saved as PNG
    unique_values = np.unique(mask_slice)
    max_label = unique_values.max()
    
    if max_label > 0 and max_label < 10:  # Small integer labels
        # Scale so each class is clearly visible
        # 0 stays 0, other values get mapped to visible intensities
        scaled = mask_slice.copy()
        scaled = (scaled * (255 // max_label)).astype(np.uint8)
        return scaled
    else:
        return mask_slice.astype(np.uint8)

def process_dataset_with_masks(image_dir, label_dir, output_image_dir, output_label_dir, axis=2):
    """
    Process NIfTI files and save only slices that have corresponding non-empty masks.
    Each sample is saved in its own folder.
    
    Args:
        image_dir: Directory containing image NIfTI files (e.g., imagesTr)
        label_dir: Directory containing label NIfTI files (e.g., labelsTr)
        output_image_dir: Directory to save image slices (each sample in its own folder)
        output_label_dir: Directory to save label slices (each sample in its own folder)
        axis: Axis along which to slice (0, 1, or 2)
    """
    # Create output directories
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get all .nii.gz files from image directory
    image_files = sorted(Path(image_dir).glob('*.nii.gz'))
    
    total_slices_saved = 0
    
    for image_file in tqdm(image_files, desc='Processing images'):
        # Construct corresponding label file path
        label_file = Path(label_dir) / image_file.name
        
        # Check if corresponding label file exists
        if not label_file.exists():
            print(f"\nWarning: No corresponding label found for {image_file.name}, skipping...")
            continue
        
        # Load both image and label
        image_slices = convert_to_2d(str(image_file), axis)
        label_slices = convert_to_2d(str(label_file), axis)
        
        # Check dimensions match
        if len(image_slices) != len(label_slices):
            print(f"\nWarning: Dimension mismatch for {image_file.name}, skipping...")
            continue
        
        # Get base filename (remove .nii.gz extension)
        base_filename = image_file.stem.replace('.nii', '')
        
        # Create sample-specific directories
        sample_image_dir = os.path.join(output_image_dir, base_filename)
        sample_label_dir = os.path.join(output_label_dir, base_filename)
        os.makedirs(sample_image_dir, exist_ok=True)
        os.makedirs(sample_label_dir, exist_ok=True)
        
        # Process each slice
        slices_saved_for_file = 0
        
        for idx, (img_slice, label_slice) in enumerate(zip(image_slices, label_slices)):
            # Check if label slice has any non-zero values (mask present)
            for cat in np.unique(label_slice):
                if cat == 0:
                    continue
                
                cat_label_slice = (label_slice == cat).astype(np.uint8)
                if np.any(cat_label_slice > 0):
                    # Normalize and save image slice
                    img_normalized = normalize_image_slice(img_slice)
                    img_pil = Image.fromarray(img_normalized)
                    img_output_path = os.path.join(sample_image_dir, f"slice_{idx:03d}_{cat}.png")
                    img_pil.save(img_output_path)
                    
                    # Scale and save label slice (make it visible in visualization)
                    label_scaled = scale_mask_for_visualization(cat_label_slice)
                    label_pil = Image.fromarray(label_scaled)
                    label_output_path = os.path.join(sample_label_dir, f"slice_{idx:03d}_{cat}.png")
                    label_pil.save(label_output_path)
                    
                    slices_saved_for_file += 1
        
        total_slices_saved += slices_saved_for_file
    
    print(f"\nTotal slices saved: {total_slices_saved}")
    
if __name__ == "__main__":
    # Sarcoma dataset paths
    base_dir = '/data/datasets/BTCV'
    output_base_dir = '/data/code/SANSA/data/BTCV'
    
    # for task in os.listdir(base_dir):
    #     # Process training data
    #     if task == ".":
    #         continue
    task = "BTCV"
    print("=" * 50)
    print(f"Processing Training Data {task}")
    print("=" * 50)
    process_dataset_with_masks(
        image_dir=os.path.join(base_dir, 'imagesTr'),
        label_dir=os.path.join(base_dir, 'labelsTr'),
        output_image_dir=os.path.join(output_base_dir, 'imagesTr'),
        output_label_dir=os.path.join(output_base_dir, 'labelsTr'),
        axis=2
    )
    
    # Process test data
    print("\n" + "=" * 50)
    print(f"Processing Test Data {task}")
    print("=" * 50)
    process_dataset_with_masks(
        image_dir=os.path.join(base_dir, 'imagesTs'),
        label_dir=os.path.join(base_dir, 'labelsTs'),
        output_image_dir=os.path.join(output_base_dir, 'imagesTs'),
        output_label_dir=os.path.join(output_base_dir, 'labelsTs'),
        axis=2
    )
