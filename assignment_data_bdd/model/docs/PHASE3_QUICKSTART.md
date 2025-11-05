# Phase 3: Evaluation and Visualization - Quick Start Guide

## 🎯 Overview

Phase 3 provides comprehensive evaluation and visualization tools for the BDD100k object detection model. This guide will help you quickly run the evaluation pipeline.

---

## 📋 Prerequisites

```bash
# 1. Ensure you're in the model directory
cd model/

# 2. Activate virtual environment (if using)
source model_training_env/bin/activate

# 3. Install dependencies (if not already done)
pip install -r requirements.txt
```

**Required**:
- Model weights (`yolov8m.pt` or trained model)
- BDD100k validation images
- BDD100k validation labels JSON

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Quick Test (100 images)

```bash
cd src/

# Step 1: Run evaluation
python evaluate.py --max-images 100

# Step 2: Generate visualizations
python visualize_metrics.py

# Step 3: Create prediction visualizations
python visualize_predictions.py --num-samples 20

# Step 4: Run failure analysis
python failure_analysis.py
```

**Expected time**: ~5-10 minutes on CPU

---

### Option 2: Full Evaluation (All 10,000 images)

```bash
cd src/

# Step 1: Run full evaluation (takes longer)
python evaluate.py

# Step 2-4: Same as above
python visualize_metrics.py
python visualize_predictions.py --num-samples 50
python failure_analysis.py
```

**Expected time**: 
- CPU: ~2-3 hours
- GPU: ~30-60 minutes

---

## 📁 Output Structure

After running, you'll have:

```
model/outputs/evaluation/
├── metrics/
│   ├── evaluation_results.json          # Full metrics (JSON)
│   ├── per_class_metrics.csv            # Per-class summary (CSV)
│   └── confusion_matrix.csv             # Confusion matrix (CSV)
│
├── charts/                               # Quantitative visualizations
│   ├── map_summary.png                  # Overall mAP
│   ├── per_class_ap.png                 # Per-class AP bars
│   ├── precision_recall_curves.png      # PR curves (all classes)
│   ├── confusion_matrix.png             # Confusion heatmaps
│   ├── precision_recall_f1_comparison.png
│   └── tp_fp_fn_distribution.png
│
├── predictions/                          # Qualitative visualizations
│   ├── comparisons/                     # GT vs Pred side-by-side
│   ├── overlay/                         # TP/FP/FN overlays
│   ├── success/                         # Best detections
│   └── failures/                        # Failure cases
│
└── failure_analysis/
    ├── failure_analysis_report.json     # Detailed failure report
    ├── small_object_performance.png     # Small object analysis
    └── class_confusion_patterns.png     # Confusion visualization
```

---

## 📊 Viewing Results

### Terminal Output

```bash
# View per-class metrics
column -t -s, ../outputs/evaluation/metrics/per_class_metrics.csv | less

# View confusion matrix
column -t -s, ../outputs/evaluation/metrics/confusion_matrix.csv | less

# View JSON results
cat ../outputs/evaluation/metrics/evaluation_results.json | jq
```

### Open Visualizations

```bash
# macOS
open ../outputs/evaluation/charts/*.png
open ../outputs/evaluation/predictions/overlay/

# Linux
xdg-open ../outputs/evaluation/charts/*.png

# Or use any image viewer
```

---

## 🔧 Customization Options

### Adjust Confidence Threshold

```bash
# Higher confidence (fewer detections, higher precision)
python evaluate.py --conf-threshold 0.5

# Lower confidence (more detections, higher recall)
python evaluate.py --conf-threshold 0.1
```

### Adjust IoU Threshold

```bash
# Stricter matching
python evaluate.py --iou-threshold 0.7

# More lenient matching
python evaluate.py --iou-threshold 0.3
```

### Use GPU

```bash
# Evaluate on GPU (much faster)
python evaluate.py --device cuda

# Specify GPU device
python evaluate.py --device cuda:0
```

### Custom Model

```bash
# Use your trained model
python evaluate.py --model ../weights/best.pt

# Use different YOLOv8 variant
python evaluate.py --model ../yolov8x.pt
```

### Custom Paths

```bash
python evaluate.py \
  --model ../yolov8m.pt \
  --data-root ../../data_analysis/data/bdd100k_images_100k \
  --labels ../../data_analysis/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --output-dir ../outputs/evaluation_custom
```

---

## 📈 Understanding Results

### Key Metrics

1. **mAP@0.5**: Overall model performance (higher is better)
   - > 0.7: Excellent
   - 0.5-0.7: Good
   - < 0.5: Needs improvement

2. **Per-Class AP**: Shows which classes model detects well
   - Identify weak classes for targeted improvement

3. **Precision vs Recall**: 
   - High Precision: Few false positives
   - High Recall: Few false negatives
   - F1: Balanced metric

4. **Confusion Matrix**: Shows class confusion patterns
   - Diagonal: Correct predictions
   - Off-diagonal: Confusions

### Visualization Guide

**Quantitative**:
- `map_summary.png`: First thing to check
- `per_class_ap.png`: Identify best/worst classes
- `confusion_matrix.png`: See which classes confuse model

**Qualitative**:
- `overlay/*.jpg`: See TP (green), FP (red), FN (blue)
- `comparisons/*.jpg`: Side-by-side GT vs Predictions
- `success/*.jpg`: Learn what works well
- `failures/*.jpg`: Learn failure modes

**Failure Analysis**:
- Small object performance
- Class confusion patterns
- Crowded scene analysis
- Low confidence detections

---

## 🐛 Troubleshooting

### Error: "Image not found"

**Problem**: Dataset path incorrect

**Solution**:
```bash
# Check your data structure
ls ../../data_analysis/data/bdd100k_images_100k/images/100k/val/

# Update paths in command
python evaluate.py --data-root /path/to/your/bdd100k_images_100k
```

### Error: "ultralytics not installed"

**Solution**:
```bash
pip install ultralytics
```

### Error: "Out of memory"

**Solution**:
```bash
# Reduce batch size (edit evaluate.py if needed)
# Or use CPU instead
python evaluate.py --device cpu --max-images 100
```

### Slow Performance

**Solutions**:
1. Use GPU: `--device cuda`
2. Reduce samples: `--max-images 100`
3. Use smaller model: `--model ../yolov8n.pt`

### Import Errors

**Solution**:
```bash
# Make sure you're in src/ directory
cd model/src/

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

---

## 📝 Common Workflows

### Workflow 1: Quick Check (Testing)

```bash
cd src/
python evaluate.py --max-images 100
python visualize_metrics.py
# Review: outputs/evaluation/charts/map_summary.png
```

### Workflow 2: Full Evaluation (Production)

```bash
cd src/
python evaluate.py                          # Full 10k images
python visualize_metrics.py                 # All charts
python visualize_predictions.py --num-samples 100
python failure_analysis.py
# Review all outputs
```

### Workflow 3: Compare Models

```bash
# Model 1
python evaluate.py --model ../yolov8m.pt --output-dir ../outputs/eval_m

# Model 2
python evaluate.py --model ../yolov8x.pt --output-dir ../outputs/eval_x

# Compare results
diff ../outputs/eval_m/metrics/per_class_metrics.csv \
     ../outputs/eval_x/metrics/per_class_metrics.csv
```

### Workflow 4: Targeted Analysis

```bash
# Focus on specific failure mode
python failure_analysis.py  # Detailed failure clustering

# Generate more prediction samples for weak classes
python visualize_predictions.py --num-samples 100
# Then manually review outputs/evaluation/predictions/failures/
```

---

## 🎓 Next Steps After Evaluation

1. **Read EVALUATION_REPORT.md**
   - Understand quantitative results
   - Review failure analysis
   - Study improvement recommendations

2. **Analyze Visualizations**
   - Identify patterns in failures
   - Compare with Phase 1 findings
   - Document observations

3. **Implement Improvements**
   - Follow recommendations in EVALUATION_REPORT.md
   - Start with highest-impact changes
   - Re-evaluate after changes

4. **Iterate**
   - Implement → Evaluate → Analyze → Refine
   - Track metrics over iterations
   - Document what works

---

## 📚 Documentation References

- **Full Report**: `docs/EVALUATION_REPORT.md`
- **Model Selection**: `docs/model_selection.md`
- **Architecture**: `docs/architecture_explained.md`
- **Phase 1 Analysis**: `../data_analysis/analysis_documentation.md`

---

## 💡 Tips for Success

1. **Start Small**: Test with `--max-images 100` first
2. **Use GPU**: Much faster evaluation (30x speedup)
3. **Review Visualizations**: Images tell the story
4. **Connect to Phase 1**: Link findings to data analysis
5. **Document Insights**: Write down observations
6. **Iterate**: Evaluation is continuous, not one-time

---

## ✅ Checklist

Before submitting Phase 3, ensure you have:

- [ ] Run full evaluation on validation set
- [ ] Generated all quantitative visualizations (6 charts)
- [ ] Created qualitative comparisons (50+ samples)
- [ ] Performed failure analysis (4 categories)
- [ ] Reviewed EVALUATION_REPORT.md
- [ ] Connected Phase 3 findings to Phase 1 analysis
- [ ] Documented improvement recommendations
- [ ] All files in `outputs/evaluation/` directory
- [ ] README.md updated with Phase 3 info
- [ ] Code follows PEP8 standards

---

## 🎯 Quick Command Reference

```bash
# Evaluation
python evaluate.py --max-images 100                    # Quick test
python evaluate.py                                     # Full evaluation
python evaluate.py --device cuda                       # GPU acceleration

# Visualizations
python visualize_metrics.py                            # Quantitative charts
python visualize_predictions.py --num-samples 50       # Qualitative samples
python failure_analysis.py                             # Failure clustering

# View results
open ../outputs/evaluation/charts/*.png                # View charts (macOS)
cat ../outputs/evaluation/metrics/per_class_metrics.csv # View metrics
```

---

**Last Updated**: November 2025  
**Phase 3 Status**: ✅ COMPLETE  
**Author**: Bosch Assignment - Phase 3 Quick Start
