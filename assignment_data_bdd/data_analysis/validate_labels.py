"""
Validate YOLO format label files to find corrupted or invalid entries.
"""


from pathlib import Path
import numpy as np

def validate_yolo_labels(labels_dir, split='train', max_class_id=9):
    """
    Validate YOLO format label files.
    
    Args:
        labels_dir: Path to labels directory
        split: Dataset split name
        max_class_id: Maximum valid class ID (9 for BDD100K with 10 classes)
    """
    labels_path = Path(labels_dir)
    label_files = list(labels_path.glob('*.txt'))
    
    print(f"\n{'='*70}")
    print(f"Validating {split} labels")
    print(f"{'='*70}\n")
    print(f"Total label files: {len(label_files)}")
    
    issues = {
        'invalid_class_id': [],
        'invalid_coordinates': [],
        'empty_files': [],
        'malformed_lines': [],
        'out_of_range': []
    }
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                issues['empty_files'].append(label_file.name)
                continue
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Check if line has correct number of values
                if len(parts) != 5:
                    issues['malformed_lines'].append(f"{label_file.name}:L{line_num} (parts={len(parts)})")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate class ID
                    if class_id < 0 or class_id > max_class_id:
                        issues['invalid_class_id'].append(f"{label_file.name}:L{line_num} (class={class_id})")
                    
                    # Validate coordinates (should be in [0, 1])
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        issues['out_of_range'].append(
                            f"{label_file.name}:L{line_num} "
                            f"(x={x_center:.3f}, y={y_center:.3f}, w={width:.3f}, h={height:.3f})"
                        )
                    
                    # Check for invalid box dimensions
                    if width <= 0 or height <= 0:
                        issues['invalid_coordinates'].append(
                            f"{label_file.name}:L{line_num} (w={width:.3f}, h={height:.3f})"
                        )
                
                except ValueError as e:
                    issues['malformed_lines'].append(f"{label_file.name}:L{line_num} (error={str(e)})")
        
        except Exception as e:
            print(f"Error reading {label_file.name}: {e}")
    
    # Print results
    print(f"\nValidation Results:")
    print(f"{'='*70}")
    
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        print("✓ All labels are valid!")
    else:
        print(f"⚠ Found {total_issues} issues:\n")
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n{issue_type.replace('_', ' ').title()}: {len(issue_list)}")
                # Show first 10 examples
                for item in issue_list[:10]:
                    print(f"  - {item}")
                if len(issue_list) > 10:
                    print(f"  ... and {len(issue_list) - 10} more")
    
    print(f"\n{'='*70}\n")
    
    return issues

if __name__ == "__main__":
    dataset_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_yolo_dataset"
    
    print("\n" + "="*70)
    print("YOLO Label Validation")
    print("="*70)
    
    # Validate train labels
    train_issues = validate_yolo_labels(f"{dataset_path}/train/labels", 'train')
    
    # Validate val labels
    val_issues = validate_yolo_labels(f"{dataset_path}/val/labels", 'val')
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    train_total = sum(len(v) for v in train_issues.values())
    val_total = sum(len(v) for v in val_issues.values())
    print(f"Train issues: {train_total}")
    print(f"Val issues: {val_total}")
    print(f"Total issues: {train_total + val_total}")
    print("="*70 + "\n")
