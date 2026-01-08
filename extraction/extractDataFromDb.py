import torch
import numpy as np
import nibabel as nib
import pickle
import lmdb
import os
import re


def convert_to_uint16(data):
    """
    Converts float32 data to uint16 (0-65535)
    Preserves the full dynamic range
    """
    # Normalize data to 0-1
    data_min = data.min()
    data_max = data.max()

    if data_max > data_min:
        normalized = (data - 0) / (255 - 0)
    else:
        normalized = np.zeros_like(data)

    # Scale to uint16 range (0-65535)
    scaled = (normalized * 4095).astype(np.uint16)

    return scaled

def extract_all_dsa_sequences(lmdb_path, output_dir):
    """
    Extracts all DSA sequences from LMDB as 4D NIFTI files
    """
    os.makedirs(output_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, map_size=500 * 1024 * 1024 * 1024)  # 500GB

    total_sequences = 0
    successful_extractions = 0

    print("🔍 Starting full extraction of all DSA sequences...")

    with env.begin() as txn:
        cursor = txn.cursor()

        # Count total sequences
        cursor.first()
        for key, value in cursor:
            print(".", end="")
            total_sequences += 1

        print(f"📊 Found: {total_sequences} sequences")
        print("=" * 50)

        # Reset cursor and extract
        cursor.first()
        for i, (key, value) in enumerate(cursor, 1):
            key_str = key.decode('utf-8')

            # Show progress
            print(f"[{i:4d}/{total_sequences}] {key_str}")
            # Reset cursor und extract
            cursor.first()
            for i, (key, value) in enumerate(cursor, 1):
                key_str = key.decode('utf-8')

                # ignore any non-RAW sequences
                if not key_str.startswith('RAW'):
                    continue

                # Show progress
                print(f"[{i:4d}/{total_sequences}] {key_str}")

                try:
                    # Load Data
                    data = pickle.loads(value)

                    if isinstance(data, torch.Tensor):
                        numpy_data = data.detach().cpu().numpy()

                        # Remove batch dimension if available
                        if len(numpy_data.shape) == 5 and numpy_data.shape[0] == 1:
                            numpy_data = numpy_data.squeeze(0)  # (1,3,T,H,W) -> (3,T,H,W)

                        # make safe filename
                        safe_filename = make_safe_filename(key_str)

                        if len(numpy_data.shape) == 4:  # (C,T,H,W)
                            # Use channel 0 only
                            channel_0_data = numpy_data[0]  # (T,H,W)

                            # Convert to uint16
                            data_uint16 = convert_to_uint16(channel_0_data)

                            # Transpose: (T,H,W) -> (H,W,T)
                            data_nifti = np.transpose(data_uint16, (1, 2, 0))

                            # Create NIFTI
                            affine = np.eye(4)
                            affine[0, 0] = affine[1, 1] = 1.0  # 1mm Pixel
                            affine[2, 2] = 1.0  # Zeitauflösung

                            nii = nib.Nifti1Image(data_nifti, affine)

                            # Save
                            output_path = os.path.join(output_dir, f"{safe_filename}_ch0.nii.gz")
                            nib.save(nii, output_path)

                            print(
                                f"    ✅ {numpy_data.shape} -> Channel 0: {channel_0_data.shape} -> uint16 -> {output_path}")
                            successful_extractions += 1

                        elif len(numpy_data.shape) == 3:  # (T,H,W)
                            # Convert to uint16
                            data_uint16 = convert_to_uint16(numpy_data)

                            # Transpose: (T,H,W) -> (H,W,T)
                            data_nifti = np.transpose(data_uint16, (1, 2, 0))

                            affine = np.eye(4)
                            nii = nib.Nifti1Image(data_nifti, affine)

                            output_path = os.path.join(output_dir, f"{safe_filename}_3D.nii.gz")
                            nib.save(nii, output_path)

                            print(f"    ✅ {numpy_data.shape} -> uint16 -> {output_path}")
                            successful_extractions += 1

                        else:
                            print(f"    ⚠️  Unknown shape: {numpy_data.shape}")

                    elif isinstance(data, np.ndarray):
                        safe_filename = make_safe_filename(key_str)

                        if len(data.shape) == 3:
                            # Convert to uint16
                            data_uint16 = convert_to_uint16(data)
                            data_nifti = np.transpose(data_uint16, (1, 2, 0))
                            nii = nib.Nifti1Image(data_nifti, np.eye(4))

                            output_path = os.path.join(output_dir, f"{safe_filename}_3D.nii.gz")
                            nib.save(nii, output_path)

                            print(f"    ✅ {data.shape} -> uint16 -> {output_path}")
                            successful_extractions += 1

                    else:
                        print(f"    ❌ Unknown data type: {type(data)}")

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    continue
    print("=" * 50)
    print(f"🎯 Extraction completed:")
    print(f"   • Total: {total_sequences} sequences")
    print(f"   • Successful: {successful_extractions} sequences")
    print(f"   • Output directory: {output_dir}")


def make_safe_filename(filename):
    """
    Makes a filename safe for file systems
    """
    # Remove problematic characters
    safe = re.sub(r'[<>:"/\\|?*#]', '_', filename)
    safe = safe.replace('.dcm', '')
    safe = safe.replace('.', '_')

    # Truncate if too long
    if len(safe) > 100:
        safe = safe[:100]

    return safe


# Main function
if __name__ == "__main__":
    lmdb_path = r'D:\ThromboMap\Alipan'
    output_dir = r'D:\ThromboMap\2025-08-28-AmTICIS-Extractor\extracted_all\FirstChannel-CorrectRange-uint16'

    extract_all_dsa_sequences(lmdb_path, output_dir)