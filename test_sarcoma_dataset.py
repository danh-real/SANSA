#!/usr/bin/env python
"""Quick test to verify sarcoma dataset loads correctly."""
import argparse
import sys
sys.path.insert(0, '/mnt/disk1/aiotlab/huync/SANSA')

from datasets import build_dataset
import opts

# Create minimal args
parser = argparse.ArgumentParser(parents=[opts.get_args_parser()])
args = parser.parse_args([
    '--dataset_file', 'sarcoma',
    '--data_root', 'data',
    '--fold', '0',
    '--shots', '1',
    '--img_size', '256'
])

print("Building dataset...")
try:
    ds = build_dataset('sarcoma', image_set='val', args=args)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Number of classes: {ds.nclass}")
    print(f"   Class IDs: {list(ds.class_ids)}")
    print(f"   Split: {ds.split}")
    print(f"   Shot: {ds.shot}")
    
    # Try to load one sample
    print("\nTrying to load first sample...")
    sample = ds[0]
    print(f"✅ Sample loaded successfully!")
    print(f"   Query image shape: {sample['query_img'].shape}")
    print(f"   Query mask shape: {sample['query_mask'].shape}")
    print(f"   Support images shape: {sample['support_imgs'].shape}")
    print(f"   Support masks shape: {sample['support_masks'].shape}")
    print(f"   Class ID: {sample['class_id']}")
    
    print("\n✅ All tests passed! Dataset is ready for inference.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


