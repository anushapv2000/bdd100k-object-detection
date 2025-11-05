#!/usr/bin/env python3
"""
Training Monitor and Analysis Script
Monitors YOLO training progress and diagnoses potential issues
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from datetime import datetime
import subprocess
import signal

class TrainingMonitor:
    """Monitor and analyze YOLO training progress"""
    
    def __init__(self, project_path="../outputs/training_logs", run_name="yolov8m_bdd100k_full"):
        """
        Initialize training monitor
        
        Args:
            project_path: Path to training output directory
            run_name: Name of the training run to monitor
        """
        self.project_path = Path(project_path)
        self.run_name = run_name
        self.run_dir = self.project_path / run_name
        self.results_csv = self.run_dir / "results.csv"
        self.args_yaml = self.run_dir / "args.yaml"
        
    def check_process_running(self):
        """Check if training process is currently running"""
        print("\n" + "="*80)
        print("🔍 CHECKING TRAINING PROCESS STATUS")
        print("="*80)
        
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            
            processes = [line for line in result.stdout.split('\n') 
                        if 'train.py' in line and 'grep' not in line]
            
            if processes:
                print("✅ Training process is RUNNING:")
                for proc in processes:
                    parts = proc.split()
                    print(f"   PID: {parts[1]}")
                    print(f"   CPU: {parts[2]}%")
                    print(f"   Memory: {parts[3]}%")
                    print(f"   Time: {parts[9]}")
                    print(f"   Command: {' '.join(parts[10:])}")
                return True
            else:
                print("❌ No training process found!")
                print("   Training may have completed or not started yet.")
                return False
                
        except Exception as e:
            print(f"⚠️  Error checking process: {e}")
            return False
    
    def check_dataset_integrity(self):
        """Verify dataset paths and files exist"""
        print("\n" + "="*80)
        print("📁 CHECKING DATASET INTEGRITY")
        print("="*80)
        
        if not self.args_yaml.exists():
            print("⚠️  args.yaml not found - training may not have started yet")
            return False
        
        try:
            with open(self.args_yaml, 'r') as f:
                args = yaml.safe_load(f)
            
            data_yaml = args.get('data', '')
            print(f"📄 Dataset config: {data_yaml}")
            
            if not os.path.exists(data_yaml):
                print(f"❌ Dataset YAML not found: {data_yaml}")
                return False
            
            # Load dataset config
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            print(f"✅ Dataset YAML loaded successfully")
            print(f"   Classes: {len(data_config.get('names', []))}")
            print(f"   Names: {', '.join(data_config.get('names', []))}")
            
            # Check paths
            base_path = Path(data_config.get('path', ''))
            train_path = base_path / data_config.get('train', '')
            val_path = base_path / data_config.get('val', '')
            
            print(f"\n📂 Checking paths:")
            print(f"   Base: {base_path} {'✅' if base_path.exists() else '❌'}")
            print(f"   Train: {train_path} {'✅' if train_path.exists() else '❌'}")
            print(f"   Val: {val_path} {'✅' if val_path.exists() else '❌'}")
            
            # Count images and labels
            if train_path.exists():
                train_images = list((train_path / 'images').glob('*.jpg'))
                train_labels = list((train_path / 'labels').glob('*.txt'))
                print(f"\n📊 Training data:")
                print(f"   Images: {len(train_images)}")
                print(f"   Labels: {len(train_labels)}")
                
                if len(train_images) == 0:
                    print(f"   ❌ NO TRAINING IMAGES FOUND!")
                elif len(train_labels) == 0:
                    print(f"   ❌ NO TRAINING LABELS FOUND!")
                elif len(train_images) != len(train_labels):
                    print(f"   ⚠️  Image-label mismatch: {len(train_images)} images vs {len(train_labels)} labels")
                else:
                    print(f"   ✅ Image-label count matches")
            
            if val_path.exists():
                val_images = list((val_path / 'images').glob('*.jpg'))
                val_labels = list((val_path / 'labels').glob('*.txt'))
                print(f"\n📊 Validation data:")
                print(f"   Images: {len(val_images)}")
                print(f"   Labels: {len(val_labels)}")
                
                if len(val_images) == 0:
                    print(f"   ❌ NO VALIDATION IMAGES FOUND!")
                elif len(val_labels) == 0:
                    print(f"   ❌ NO VALIDATION LABELS FOUND!")
                elif len(val_images) != len(val_labels):
                    print(f"   ⚠️  Image-label mismatch: {len(val_images)} images vs {len(val_labels)} labels")
                else:
                    print(f"   ✅ Image-label count matches")
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking dataset: {e}")
            return False
    
    def check_training_progress(self):
        """Check training progress from results.csv"""
        print("\n" + "="*80)
        print("📈 TRAINING PROGRESS ANALYSIS")
        print("="*80)
        
        if not self.results_csv.exists():
            print("⚠️  results.csv not found yet")
            print("   Training may still be in data loading phase")
            print("   This is normal for the first 1-2 minutes")
            return None
        
        try:
            df = pd.read_csv(self.results_csv)
            df.columns = df.columns.str.strip()
            
            print(f"✅ Training has started!")
            print(f"   Total epochs completed: {len(df)}")
            
            if len(df) == 0:
                print("   No epochs completed yet")
                return df
            
            # Get latest metrics
            latest = df.iloc[-1]
            print(f"\n📊 Latest Metrics (Epoch {int(latest['epoch']) + 1}):")
            print(f"   Train Loss: {latest['train/box_loss']:.4f} (box), "
                  f"{latest['train/cls_loss']:.4f} (cls), "
                  f"{latest['train/dfl_loss']:.4f} (dfl)")
            print(f"   Val Loss: {latest['val/box_loss']:.4f} (box), "
                  f"{latest['val/cls_loss']:.4f} (cls), "
                  f"{latest['val/dfl_loss']:.4f} (dfl)")
            print(f"   mAP50: {latest['metrics/mAP50(B)']:.4f}")
            print(f"   mAP50-95: {latest['metrics/mAP50-95(B)']:.4f}")
            
            # Check for issues
            self._diagnose_training_issues(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error reading results: {e}")
            return None
    
    def _diagnose_training_issues(self, df):
        """Diagnose potential training issues"""
        print("\n🔍 DIAGNOSTIC CHECKS:")
        
        if len(df) < 2:
            print("   ⏳ Not enough epochs to diagnose yet")
            return
        
        # Check if loss is decreasing
        train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        
        if train_loss.iloc[-1] > train_loss.iloc[0]:
            print("   ⚠️  Training loss is INCREASING - potential issues:")
            print("      - Learning rate might be too high")
            print("      - Data augmentation might be too aggressive")
            print("      - Check for data quality issues")
        elif train_loss.iloc[-1] < train_loss.iloc[0] * 0.5:
            print("   ✅ Training loss is decreasing well")
        else:
            print("   ℹ️  Training loss is decreasing slowly")
        
        # Check for NaN values
        if df.isnull().any().any():
            print("   ❌ NaN values detected in metrics!")
            print("      - This indicates a critical training issue")
            print("      - Training may have crashed or diverged")
        
        # Check mAP progression
        if 'metrics/mAP50(B)' in df.columns:
            latest_map = df['metrics/mAP50(B)'].iloc[-1]
            if latest_map == 0:
                print("   ⚠️  mAP is 0 - model is not detecting anything yet")
                print("      - This is normal for first few epochs")
                print("      - Should improve after 5-10 epochs")
            elif latest_map < 0.1:
                print(f"   ⚠️  Low mAP ({latest_map:.4f}) - model learning slowly")
            else:
                print(f"   ✅ mAP is improving ({latest_map:.4f})")
        
        # Check validation vs training loss
        val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        if val_loss.iloc[-1] > train_loss.iloc[-1] * 2:
            print("   ⚠️  Large gap between training and validation loss")
            print("      - Possible overfitting")
            print("      - Consider adding more augmentation")
    
    def check_output_files(self):
        """Check what output files have been generated"""
        print("\n" + "="*80)
        print("📄 OUTPUT FILES CHECK")
        print("="*80)
        
        if not self.run_dir.exists():
            print(f"❌ Run directory not found: {self.run_dir}")
            return
        
        print(f"📁 Run directory: {self.run_dir}")
        
        files = list(self.run_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            print("   ⚠️  No files generated yet")
            return
        
        print(f"\n✅ Found {len(files)} files:")
        
        # Categorize files
        weights = [f for f in files if f.suffix == '.pt']
        images = [f for f in files if f.suffix in ['.jpg', '.png']]
        configs = [f for f in files if f.suffix in ['.yaml', '.yml']]
        logs = [f for f in files if f.suffix == '.csv']
        
        if weights:
            print(f"\n   🏋️  Weights ({len(weights)}):")
            for w in weights:
                size = w.stat().st_size / (1024*1024)
                print(f"      - {w.name} ({size:.1f} MB)")
        
        if images:
            print(f"\n   🖼️  Images ({len(images)}):")
            for img in images[:5]:
                print(f"      - {img.name}")
            if len(images) > 5:
                print(f"      ... and {len(images)-5} more")
        
        if configs:
            print(f"\n   ⚙️  Configs ({len(configs)}):")
            for cfg in configs:
                print(f"      - {cfg.name}")
        
        if logs:
            print(f"\n   📊 Logs ({len(logs)}):")
            for log in logs:
                print(f"      - {log.name}")
    
    def plot_training_progress(self, df):
        """Plot training curves"""
        if df is None or len(df) == 0:
            print("\n⚠️  No training data to plot yet")
            return
        
        print("\n" + "="*80)
        print("📊 GENERATING TRAINING PLOTS")
        print("="*80)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Losses
            ax1 = axes[0, 0]
            ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='o')
            ax1.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', marker='s')
            ax1.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', marker='^')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Losses')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Validation Losses
            ax2 = axes[0, 1]
            ax2.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', marker='o')
            ax2.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', marker='s')
            ax2.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', marker='^')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Losses')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: mAP
            ax3 = axes[1, 0]
            ax3.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', marker='o', linewidth=2)
            ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', marker='s', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('mAP')
            ax3.set_title('Mean Average Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Precision & Recall
            ax4 = axes[1, 1]
            ax4.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
            ax4.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.set_title('Precision & Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.run_dir / 'training_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def run_full_analysis(self):
        """Run complete training analysis"""
        print("\n" + "="*80)
        print("🚀 YOLO TRAINING MONITOR & ANALYSIS")
        print("="*80)
        print(f"Run: {self.run_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if process is running
        is_running = self.check_process_running()
        
        # Check dataset integrity
        self.check_dataset_integrity()
        
        # Check output files
        self.check_output_files()
        
        # Check training progress
        df = self.check_training_progress()
        
        # Generate plots if data available
        if df is not None and len(df) > 0:
            self.plot_training_progress(df)
        
        # Summary
        print("\n" + "="*80)
        print("📋 SUMMARY")
        print("="*80)
        
        if is_running:
            print("✅ Training is currently running")
            if df is not None and len(df) > 0:
                print(f"✅ {len(df)} epochs completed")
                print(f"✅ Latest mAP50: {df.iloc[-1]['metrics/mAP50(B)']:.4f}")
            else:
                print("⏳ Training started but no epochs completed yet")
                print("   This is normal - data loading can take 1-2 minutes")
        else:
            print("⚠️  No training process detected")
            if df is not None and len(df) > 0:
                print(f"ℹ️  Training may have completed ({len(df)} epochs found)")
            else:
                print("ℹ️  Training may not have started yet")
        
        print("\n💡 NEXT STEPS:")
        if not is_running and (df is None or len(df) == 0):
            print("   1. Start training with: python src/train.py --full")
        elif is_running and (df is None or len(df) == 0):
            print("   1. Wait 1-2 minutes for data loading to complete")
            print("   2. Run this monitor again to check progress")
        elif is_running and df is not None:
            print("   1. Training is progressing normally")
            print("   2. Run this monitor periodically to check progress")
            print("   3. Check training_analysis.png for visual progress")
        
        print("\n" + "="*80)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor YOLO training progress')
    parser.add_argument('--project', default='../outputs/training_logs', 
                       help='Project directory path')
    parser.add_argument('--run', default='yolov8m_bdd100k_full',
                       help='Run name to monitor')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously monitor (refresh every 30s)')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.project, args.run)
    
    if args.watch:
        print("👀 Watch mode enabled - Will refresh every 30 seconds")
        print("   Press Ctrl+C to stop")
        try:
            while True:
                monitor.run_full_analysis()
                print("\n⏰ Waiting 30 seconds before next check...")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
    else:
        monitor.run_full_analysis()

if __name__ == "__main__":
    main()
