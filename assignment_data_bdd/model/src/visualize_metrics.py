"""
Quantitative Metrics Visualization for BDD100k Object Detection.

This script creates charts and graphs for quantitative analysis of model performance:
- Per-class AP bar charts
- Precision-Recall curves
- Confusion matrix heatmap
- Confidence score distribution
- Performance by object size

Author: Bosch Assignment - Phase 3
Date: November 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# BDD100k class names
BDD_CLASSES = [
    'bike', 'bus', 'car', 'motor', 'person', 
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]


class MetricsVisualizer:
    """Visualizer for quantitative metrics."""
    
    def __init__(self, results_path: str, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to evaluation results JSON
            output_dir: Directory to save visualizations
        """
        self.results_path = results_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        print(f"Loading results from: {results_path}")
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Visualizations will be saved to: {output_dir}\n")
    
    def plot_per_class_ap(self):
        """Create bar chart of per-class Average Precision."""
        print("Creating per-class AP bar chart...")
        
        # Extract AP values
        class_names = []
        ap_values = []
        
        for class_id in range(len(BDD_CLASSES)):
            class_names.append(BDD_CLASSES[class_id])
            ap = self.results['per_class_curves'][str(class_id)]['ap']
            ap_values.append(ap)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create bars with color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
        bars = ax.bar(class_names, ap_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, ap_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add mean line
        mean_ap = np.mean(ap_values)
        ax.axhline(mean_ap, color='red', linestyle='--', linewidth=2, label=f'Mean AP: {mean_ap:.3f}')
        
        # Formatting
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Precision (AP@0.5)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Average Precision (AP@0.5)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / "per_class_ap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def plot_precision_recall_curves(self):
        """Create precision-recall curves for all classes."""
        print("Creating precision-recall curves...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for class_id in range(len(BDD_CLASSES)):
            ax = axes[class_id]
            class_name = BDD_CLASSES[class_id]
            
            # Get PR curve data
            curve_data = self.results['per_class_curves'][str(class_id)]
            precision = np.array(curve_data['precision'])
            recall = np.array(curve_data['recall'])
            ap = curve_data['ap']
            
            # Plot
            if len(precision) > 0 and len(recall) > 0:
                ax.plot(recall, precision, linewidth=2, label=f'AP={ap:.3f}')
                ax.fill_between(recall, precision, alpha=0.2)
            
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(f'{class_name}', fontsize=11, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Precision-Recall Curves for All Classes', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / "precision_recall_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Create confusion matrix heatmap."""
        print("Creating confusion matrix heatmap...")
        
        # Get confusion matrix
        conf_matrix = np.array(self.results['confusion_matrix'])
        
        # Normalize by row (ground truth)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_norm = np.divide(conf_matrix, row_sums, 
                                      where=row_sums!=0, 
                                      out=np.zeros_like(conf_matrix, dtype=float))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Labels
        labels = BDD_CLASSES + ['background']
        
        # Plot 1: Absolute counts
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # Plot 2: Normalized
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def plot_precision_recall_f1_comparison(self):
        """Create comparison of precision, recall, and F1 across classes."""
        print("Creating precision/recall/F1 comparison...")
        
        # Extract metrics
        class_names = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for class_id in range(len(BDD_CLASSES)):
            class_names.append(BDD_CLASSES[class_id])
            metrics = self.results['precision_recall_f1'][str(class_id)]
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2E86AB')
        bars2 = ax.bar(x, recalls, width, label='Recall', color='#A23B72')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#F18F01')
        
        # Formatting
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Precision, Recall, and F1-Score by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / "precision_recall_f1_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def plot_tp_fp_fn_distribution(self):
        """Create stacked bar chart of TP, FP, FN distribution."""
        print("Creating TP/FP/FN distribution...")
        
        # Extract data
        class_names = []
        tp_counts = []
        fp_counts = []
        fn_counts = []
        
        for class_id in range(len(BDD_CLASSES)):
            class_names.append(BDD_CLASSES[class_id])
            metrics = self.results['precision_recall_f1'][str(class_id)]
            tp_counts.append(metrics['tp'])
            fp_counts.append(metrics['fp'])
            fn_counts.append(metrics['fn'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(class_names))
        
        # Create stacked bars
        ax.bar(x, tp_counts, label='True Positives', color='#06D6A0')
        ax.bar(x, fp_counts, bottom=tp_counts, label='False Positives', color='#EF476F')
        
        bottom = np.array(tp_counts) + np.array(fp_counts)
        ax.bar(x, fn_counts, bottom=bottom, label='False Negatives', color='#FFD166')
        
        # Formatting
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Detection Statistics by Class (TP/FP/FN)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / "tp_fp_fn_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def plot_map_summary(self):
        """Create summary visualization of mAP metrics."""
        print("Creating mAP summary...")
        
        # Extract mAP values
        map_data = self.results['map']
        
        # Get mAP@0.5 and per-class AP
        map_50 = map_data['mAP@0_50']
        map_50_95 = map_data.get('mAP@0_5:0_95', 0)
        
        per_class_ap = []
        for class_id in range(len(BDD_CLASSES)):
            ap = self.results['per_class_curves'][str(class_id)]['ap']
            per_class_ap.append(ap)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 2])
        
        # Left: Overall mAP
        ax1 = fig.add_subplot(gs[0])
        metrics = ['mAP@0.5', 'mAP@0.5:0.95']
        values = [map_50, map_50_95]
        colors = ['#118AB2', '#073B4C']
        
        bars = ax1.barh(metrics, values, color=colors, edgecolor='black', linewidth=2)
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}',
                    ha='left', va='center', fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('mAP Score', fontsize=12, fontweight='bold')
        ax1.set_title('Overall mAP Performance', fontsize=13, fontweight='bold')
        ax1.set_xlim([0, 1.0])
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Per-class AP sorted
        ax2 = fig.add_subplot(gs[1])
        
        # Sort by AP
        sorted_indices = np.argsort(per_class_ap)
        sorted_classes = [BDD_CLASSES[i] for i in sorted_indices]
        sorted_aps = [per_class_ap[i] for i in sorted_indices]
        
        # Color by performance
        colors_sorted = ['#EF476F' if ap < 0.3 else '#FFD166' if ap < 0.6 else '#06D6A0' 
                        for ap in sorted_aps]
        
        bars = ax2.barh(sorted_classes, sorted_aps, color=colors_sorted, 
                       edgecolor='black', linewidth=1)
        
        for bar, value in zip(bars, sorted_aps):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}',
                    ha='left', va='center', fontsize=9)
        
        ax2.set_xlabel('Average Precision', fontsize=12, fontweight='bold')
        ax2.set_title('Per-Class AP (Sorted)', fontsize=13, fontweight='bold')
        ax2.set_xlim([0, 1.0])
        ax2.axvline(map_50, color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {map_50:.3f}', alpha=0.7)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / "map_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def create_all_visualizations(self):
        """Generate all quantitative visualizations."""
        print(f"\n{'='*60}")
        print(f"Creating Quantitative Visualizations")
        print(f"{'='*60}\n")
        
        self.plot_map_summary()
        self.plot_per_class_ap()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrix()
        self.plot_precision_recall_f1_comparison()
        self.plot_tp_fp_fn_distribution()
        
        print(f"\n{'='*60}")
        print(f"All visualizations created successfully!")
        print(f"Saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Create quantitative metrics visualizations")
    
    parser.add_argument(
        '--results',
        type=str,
        default='../outputs/evaluation/metrics/evaluation_results.json',
        help='Path to evaluation results JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../outputs/evaluation/charts',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MetricsVisualizer(
        results_path=args.results,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    visualizer.create_all_visualizations()
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
