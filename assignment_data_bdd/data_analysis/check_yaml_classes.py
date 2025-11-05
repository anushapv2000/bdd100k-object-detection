"""
Verify YAML configuration against actual label files
"""

from pathlib import Path
from collections import Counter
import yaml

def check_yaml_classes(yaml_path, labels_dir):
    """Check if YAML classes match the actual classes in label files"""
    
    # Load YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("YAML Configuration Check")
    print("="*70)
    
    # Check YAML structure
    print("\n1. YAML Structure:")
    print(f"   Path: {config.get('path', 'MISSING')}")
    print(f"   Train: {config.get('train', 'MISSING')}")
    print(f"   Val: {config.get('val', 'MISSING')}")
    print(f"   Number of classes (nc): {config.get('nc', 'MISSING')}")
    
    # Check class names
    yaml_classes = config.get('names', {})
    print(f"\n2. Classes in YAML ({len(yaml_classes)} classes):")
    for class_id, class_name in sorted(yaml_classes.items()):
        print(f"   {class_id}: {class_name}")
    
    # Check actual labels
    labels_path = Path(labels_dir)
    all_class_ids = set()
    class_counts = Counter()
    
    label_files = list(labels_path.glob('*.txt'))
    print(f"\n3. Scanning {len(label_files)} label files...")
    
    for label_file in label_files[:1000]:  # Sample first 1000
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            all_class_ids.add(class_id)
                            class_counts[class_id] += 1
        except Exception as e:
            print(f"   Error reading {label_file.name}: {e}")
    
    print(f"\n4. Classes found in labels:")
    for class_id in sorted(all_class_ids):
        yaml_name = yaml_classes.get(class_id, "NOT IN YAML!")
        print(f"   {class_id}: {yaml_name} ({class_counts[class_id]} instances)")
    
    # Validation
    print(f"\n5. Validation:")
    issues = []
    
    # Check if nc matches
    if config.get('nc') != len(yaml_classes):
        issues.append(f"   ⚠ nc={config.get('nc')} but {len(yaml_classes)} classes defined")
    
    # Check if all class IDs are in YAML
    for class_id in all_class_ids:
        if class_id not in yaml_classes:
            issues.append(f"   ⚠ Class ID {class_id} found in labels but not in YAML")
        if class_id < 0 or class_id >= config.get('nc', 0):
            issues.append(f"   ⚠ Class ID {class_id} is out of range [0, {config.get('nc', 0)-1}]")
    
    # Check if YAML has classes not in labels
    for class_id in yaml_classes:
        if class_id not in all_class_ids:
            issues.append(f"   ⚠ Class ID {class_id} ({yaml_classes[class_id]}) in YAML but not found in labels (sampled)")
    
    if not issues:
        print("   ✓ All validations passed!")
    else:
        print("   Issues found:")
        for issue in issues:
            print(issue)
    
    # Check paths
    print(f"\n6. Path Validation:")
    base_path = Path(config.get('path', ''))
    train_path = base_path / config.get('train', '')
    val_path = base_path / config.get('val', '')
    
    print(f"   Base path exists: {base_path.exists()}")
    print(f"   Train path: {train_path}")
    print(f"   Train exists: {train_path.exists()}")
    if train_path.exists():
        print(f"   Train/images exists: {(train_path / 'images').exists()}")
        print(f"   Train/labels exists: {(train_path / 'labels').exists()}")
    print(f"   Val path: {val_path}")
    print(f"   Val exists: {val_path.exists()}")
    if val_path.exists():
        print(f"   Val/images exists: {(val_path / 'images').exists()}")
        print(f"   Val/labels exists: {(val_path / 'labels').exists()}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    yaml_path = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/configs/bdd100k.yaml"
    labels_dir = "/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/data_analysis/data/bdd100k_yolo_dataset/train/labels"
    
    check_yaml_classes(yaml_path, labels_dir)
