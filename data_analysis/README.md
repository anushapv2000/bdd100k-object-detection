# BDD100k Dataset Analysis - Phase 1

## Overview
This directory contains a comprehensive analysis of the BDD100k dataset for object detection tasks. The analysis focuses exclusively on the 10 object detection classes with bounding box annotations, excluding semantic segmentation data such as drivable areas and lane markings.

## Project Structure
```
data_analysis/
├── Dockerfile                      # Docker configuration for containerization
├── requirements.txt                # Python dependencies
├── data_analysis.py               # Main analysis script
├── dashboard.py                   # Interactive dashboard application
├── analysis_documentation.md      # Detailed analysis findings
├── README.md                      # This file
├── VISUALIZATION_GUIDE_FOR_BEGINNERS.md  # Beginner-friendly visualization guide
└── data/                          # Dataset directory (mounted as volume)
    ├── bdd100k_images_100k/
    └── bdd100k_labels_release/
```

## Features

### Analysis Components
✅ **Class Distribution Analysis**: Distribution of object detection classes across train/val splits  
✅ **Train-Validation Split Comparison**: Detailed comparison with percentage calculations  
✅ **Anomaly Detection**: Identification of underrepresented classes (<1% threshold)  
✅ **Bounding Box Size Analysis**: Statistical analysis of bbox dimensions by class  
✅ **Objects Per Image Analysis**: Distribution of object counts per image  
✅ **Unique Sample Identification**: Detection and visualization of interesting samples  
✅ **Interactive Dashboard**: Web-based visualization dashboard  
✅ **Actual Image Visualization**: Sample images with bounding box overlays  

### Unique Sample Categories (Enhanced)

The analysis now identifies and visualizes **10 different categories** of interesting/unique samples:

#### 1. **Single Object Samples** 🎯
- Images containing exactly one labeled object
- Useful for understanding individual object appearance
- **Output**: `single_object_0.jpg` to `single_object_4.jpg`

#### 2. **Many Objects Samples** 📦
- Images with more than 15 labeled objects
- Represents complex urban scenes
- **Output**: `many_objects_0.jpg` to `many_objects_4.jpg`

#### 3. **Extremely Dense Samples** 🏙️
- Images with 60-70 objects (maximum complexity)
- Rush hour traffic, busy intersections
- **Output**: `extremely_dense_60_70_objects_0.jpg` to `extremely_dense_60_70_objects_9.jpg`

#### 4. **Small Objects** 🔍
- Images containing tiny bounding boxes (<1000 px²)
- Distant objects, challenging for detection
- **Output**: `small_objects_0.jpg` to `small_objects_4.jpg`

#### 5. **Large Objects** 🚛
- Images with large bounding boxes (>100,000 px²)
- Close-up vehicles, large trucks
- **Output**: `large_objects_0.jpg` to `large_objects_4.jpg`

#### 6. **Tiny Bounding Boxes** 🐜
- Extremely small bounding boxes (<100 px²)
- Edge case for model training
- **Output**: `tiny_bbox_0.jpg` to `tiny_bbox_4.jpg`

#### 7. **Huge Bounding Boxes** 🦕
- Extremely large bounding boxes (>200,000 px²)
- Very close objects filling the frame
- **Output**: `huge_bbox_0.jpg` to `huge_bbox_4.jpg`

#### 8. **Class-Specific Representatives** 🏷️
- One prominent sample per object class
- Shows what each of the 10 classes looks like
- **Output**: `class_car_0.jpg`, `class_person_0.jpg`, etc. (one per class)

#### 9. **Diverse Class Samples** 🌈
- Images containing 6+ different object classes
- High diversity scenes showing multiple object types
- **Output**: `diverse_classes_0.jpg` to `diverse_classes_4.jpg`

#### 10. **Occlusion/Overlap Samples** 🔲
- Images with significant object overlap (5+ overlapping pairs)
- Objects blocking each other (occlusion challenge)
- **Output**: `occlusion_overlap_0.jpg` to `occlusion_overlap_4.jpg`

#### 11. **Class Co-occurrence Patterns** 🤝
- Person + Traffic Light (pedestrian crossing scenarios)
- Car + Traffic Sign (driving scenarios)
- Person + Car (safety-critical scenarios)
- Bike + Rider (cyclist detection)
- **Output**: `cooccurrence_person_traffic_light_0.jpg`, etc.

### Why These Categories Matter for Interview

**Shows Deep Understanding:**
- Demonstrates awareness of edge cases and challenging scenarios
- Shows consideration for model training difficulties
- Proves thorough dataset exploration

**Practical Value:**
- **Occlusion samples**: Help train models to handle overlapping objects
- **Extreme bbox sizes**: Ensure model works across scale variations
- **Diverse class samples**: Test multi-object detection capability
- **Co-occurrence patterns**: Understand real-world object relationships
- **Class-specific samples**: Quick reference for each object type

**Interview Discussion Points:**
- "I identified images with 60-70 objects to understand maximum scene complexity"
- "Tiny bounding boxes (<100 px²) represent distant objects that are harder to detect"
- "Co-occurrence patterns like person+traffic light reveal pedestrian crossing scenarios"
- "Occlusion detection helps prepare for overlapping objects in crowded scenes"

### Code Quality
- **PEP8 Compliant**: All code formatted using Black
- **Comprehensive Docstrings**: Complete documentation for all functions
- **Type Hints**: Clear function signatures
- **Error Handling**: Robust error handling for file operations

## Prerequisites

### System Requirements
- Docker installed and running
- Minimum 8GB RAM recommended
- 10GB free disk space for dataset and analysis outputs

### Dataset Setup
Download the BDD100k dataset:
- **Images**: 100k Images (5.3GB) - https://www.bdd100k.com/
- **Labels**: Labels (107MB) - Training and Validation labels

Place the dataset in the following structure:
```
data_analysis/
└── data/
    ├── bdd100k_images_100k/
    │   └── bdd100k/
    │       └── images/
    │           └── 100k/
    │               ├── train/
    │               └── val/
    └── bdd100k_labels_release/
        └── bdd100k/
            └── labels/
                ├── bdd100k_labels_images_train.json
                └── bdd100k_labels_images_val.json
```

## Installation and Usage

### Option 1: Docker Container (Recommended for Interview Submission)

#### 1. Build the Docker Image
Navigate to the `data_analysis` directory and build the image:
```bash
cd data_analysis
docker build -t bdd-data-analysis .
```

**Build Time**: Approximately 2-5 minutes depending on internet speed.

#### 2. Run the Container
Run the container with the data folder mounted as a volume:
```bash
docker run -p 8050:8050 -v $(pwd)/data:/app/data bdd-data-analysis
```

**What Happens on Startup:**
The dashboard will automatically:
1. Load and analyze training/validation labels
2. Generate all 11 categories of unique sample visualizations
3. Save visualized images to `output_samples/` directory
4. Start the web dashboard on port 8050

**Expected Output:**
```
Loading datasets for dashboard...
============================================================
Searching for images with 60-70 objects...
Found X images with 60-70 objects!
...
Searching for class-specific representative samples...
Found representative samples for 10 classes:
  - car
  - person
  - truck
  ...
✓ Saved: class_car_0.jpg (X boxes)
✓ Saved: diverse_classes_0.jpg (X boxes)
...
Starting dashboard server...
Dashboard will be available at http://0.0.0.0:8050
```

**Parameters Explained:**
- `-p 8050:8050`: Maps port 8050 from container to host
- `-v $(pwd)/data:/app/data`: Mounts local data directory to container
- `bdd-data-analysis`: Image name

#### 3. Access the Dashboard
Open your web browser and navigate to:
```
http://localhost:8050
```

The dashboard will display:
- Summary statistics
- Class distribution charts (linear and log scale)
- Train vs validation comparisons
- Percentage distribution
- Anomaly detection results
- Objects per image histogram

#### 4. View Generated Visualizations
Check the `output_samples/` directory for all generated images:
```bash
ls -la output_samples/
```

You should see approximately **60-70+ images** including:
- Single object samples (5 images)
- Many objects samples (5 images)
- Extremely dense samples (10 images)
- Small/large objects (10 images)
- Tiny/huge bbox samples (10 images)
- Class-specific samples (10 images, one per class)
- Diverse class samples (5 images)
- Occlusion samples (5 images)
- Co-occurrence patterns (12+ images)

#### 5. Run Analysis Script Separately (Optional)
To run the full analysis script with all visualizations:
```bash
docker run -v $(pwd)/data:/app/data bdd-data-analysis python data_analysis.py
```

This provides detailed console output with statistics for each category.

### Option 2: Local Python Environment (For Development)

#### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Analysis
```bash
python data_analysis.py
```

#### 4. Run Dashboard
```bash
python dashboard.py
```
Then access at http://localhost:8050

## Output Files

### Generated Visualizations (60-70+ Images)

All visualizations are saved in the `output_samples/` directory:

**Basic Categories:**
1. `single_object_*.jpg` - Images with 1 object (5 images)
2. `many_objects_*.jpg` - Images with 15+ objects (5 images)
3. `small_objects_*.jpg` - Small bounding boxes (5 images)
4. `large_objects_*.jpg` - Large bounding boxes (5 images)

**Advanced Categories:**
5. `extremely_dense_60_70_objects_*.jpg` - Maximum complexity scenes (10 images)
6. `tiny_bbox_*.jpg` - Extremely small boxes <100 px² (5 images)
7. `huge_bbox_*.jpg` - Extremely large boxes >200k px² (5 images)
8. `class_[classname]_*.jpg` - One sample per class (10 images)
9. `diverse_classes_*.jpg` - Multi-class diversity (5 images)
10. `occlusion_overlap_*.jpg` - Overlapping objects (5 images)

**Co-occurrence Patterns:**
11. `cooccurrence_person_traffic_light_*.jpg` (3 images)
12. `cooccurrence_car_traffic_sign_*.jpg` (3 images)
13. `cooccurrence_person_car_*.jpg` (3 images)
14. `cooccurrence_bike_person_*.jpg` (3 images)

### Console Output
The analysis script provides detailed console output including:
- Total object counts and class distribution
- Train vs validation split statistics
- Anomaly detection results
- Bounding box size statistics
- Objects per image statistics
- Unique sample counts for all 11 categories
- Success/failure messages for each visualization

## Analysis Documentation

For detailed findings, methodology, and recommendations, see:
- **[analysis_documentation.md](analysis_documentation.md)**: Comprehensive analysis report
- **[VISUALIZATION_GUIDE_FOR_BEGINNERS.md](VISUALIZATION_GUIDE_FOR_BEGINNERS.md)**: Beginner-friendly guide to understanding visualizations

Key sections include:
- Data parsing methodology
- Class distribution findings
- Anomaly detection results
- Bounding box size analysis
- Pattern identification
- Recommendations for model training

## Code Quality Standards

### PEP8 Compliance
All code follows PEP8 standards. To verify:
```bash
# Format code with Black
black data_analysis.py dashboard.py

# Check with Pylint
pylint data_analysis.py dashboard.py
```

### Documentation Standards
- All functions have comprehensive docstrings
- Type hints provided where applicable
- Inline comments for complex logic
- Module-level documentation

## Troubleshooting

### Common Issues

**1. Docker Build Fails with Timeout**
```bash
# Increase timeout
docker build --build-arg PIP_DEFAULT_TIMEOUT=1000 -t bdd-data-analysis .
```

**2. Port 8050 Already in Use**
```bash
# Use different port
docker run -p 8051:8050 -v $(pwd)/data:/app/data bdd-data-analysis
# Access at http://localhost:8051
```

**3. Data Not Found Error**
- Verify data directory structure matches expected format
- Check volume mount path: `$(pwd)/data` should point to your data directory
- On Windows, use absolute path: `-v C:\path\to\data:/app/data`

**4. Memory Issues**
- Increase Docker memory limit in Docker Desktop settings
- Minimum 4GB recommended, 8GB optimal

**5. Dashboard Not Loading**
- Wait 30-60 seconds for initial data loading
- Check console output for error messages
- Verify dataset files are accessible

### Logs and Debugging
```bash
# View container logs
docker logs <container_id>

# Run container in interactive mode
docker run -it -v $(pwd)/data:/app/data bdd-data-analysis /bin/bash
```

## Performance Notes

### Processing Time
- **Dashboard Startup**: 30-60 seconds (loading and analyzing datasets)
- **Full Analysis Script**: 2-5 minutes (includes visualization generation)
- **Memory Usage**: ~2-4GB during processing

### Optimization Tips
- First run may take longer due to data loading
- Subsequent runs are faster due to caching
- For faster development, work with subset of data

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts
- **dash**: Web dashboard framework
- **Pillow**: Image processing and visualization

### Docker Specifications
- **Base Image**: python:3.9-slim
- **Exposed Port**: 8050
- **Working Directory**: /app
- **Entry Point**: dashboard.py

## Assignment Requirements Coverage

✅ **Parser and Data Structure**: Custom JSON parser with box2d filtering  
✅ **Class Distribution Analysis**: Complete with train/val splits  
✅ **Train-Val Split Analysis**: Comparative analysis with percentages  
✅ **Anomaly Detection**: Identifies underrepresented classes  
✅ **Dashboard Visualization**: Interactive web-based dashboard  
✅ **Unique Sample Identification**: Multiple categories identified and visualized  
✅ **Documentation**: Comprehensive analysis documentation  
✅ **PEP8 Compliance**: Code formatted with Black  
✅ **Containerization**: Self-contained Docker container  

## Next Steps (Phase 2 & 3)

Based on the data analysis findings, the next phases will include:
1. **Model Selection**: YOLOv8 for multi-scale object detection
2. **Data Loader**: Custom loader for BDD100k format
3. **Training Pipeline**: Implementation with class-weighted loss
4. **Evaluation**: Comprehensive metrics on validation set
5. **Visualization**: Prediction overlays and performance analysis

## Contact and Support

For questions or issues:
- Review `analysis_documentation.md` for detailed findings
- Check Docker logs for runtime errors
- Verify dataset structure matches expected format

## License and Attribution

- **Dataset**: BDD100k - Berkeley DeepDrive
- **Analysis**: Created for interview assignment
- **Date**: November 2024

---

**Note**: This analysis container is self-contained and requires no additional installations beyond Docker. All dependencies are included in the Docker image.
