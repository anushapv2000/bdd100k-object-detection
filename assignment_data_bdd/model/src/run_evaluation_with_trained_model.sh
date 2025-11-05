#!/bin/bash
# Quick evaluation script using the trained model checkpoint

echo "=========================================="
echo "Running Evaluation with Trained Model"
echo "=========================================="
echo ""

# Path to trained model
TRAINED_MODEL="/Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/outputs/training_logs/yolov8m_bdd100k_subset_demo/weights/best.pt"

# Check if model exists
if [ ! -f "$TRAINED_MODEL" ]; then
    echo "ERROR: Trained model not found at:"
    echo "  $TRAINED_MODEL"
    echo ""
    echo "Please run training first:"
    echo "  cd model/src/"
    echo "  python train.py --subset 300 --epochs 1"
    exit 1
fi

echo "Using trained model: $TRAINED_MODEL"
echo ""

# Run evaluation with trained model
cd /Users/ayushsoral/Desktop/code/anusha_assmt/assignment_data_bdd/model/src

echo "Running quick evaluation (10 images)..."
python evaluate.py \
    --model "$TRAINED_MODEL" \
    --max-images 10 \
    --device cpu

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Generate visualizations:"
echo "     python visualize_metrics.py"
echo "     python visualize_predictions.py --model \"$TRAINED_MODEL\" --num-samples 5"
echo ""
echo "  2. View results:"
echo "     open ../outputs/evaluation/charts/"
echo ""
