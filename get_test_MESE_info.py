import sys

import numpy as np
import SimpleITK as sitk

# Add project root to path if necessary, or run from root
# sys.path.insert(0, '/path/to/your/pymskt/project')
from pymskt.image.t2 import T2DataLoader

# Path to your anonymized data
dicom_dir = "data/MESE_T2_dicom_data_anon"

try:
    # Load the data using the class
    loader = T2DataLoader(source=dicom_dir, input_type="dicom")

    # Get the results
    tes = loader.get_tes()
    data = loader.get_data_array()
    ref_img = loader.get_reference_image()

    # Print the information needed for the test
    print(f"Expected TEs: {tes.tolist()}")  # Convert to list for easy copying
    print(f"Expected Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")
    print(f"Reference Image Size: {ref_img.GetSize()}")
    print(f"Reference Image Spacing: {ref_img.GetSpacing()}")
    print(f"Reference Image Origin: {ref_img.GetOrigin()}")
    print(f"Reference Image Direction: {ref_img.GetDirection()}")

except Exception as e:
    print(f"Error loading data: {e}")
    # If error, maybe print traceback?
    # import traceback
    # traceback.print_exc()
