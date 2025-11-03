"""
Model Selection and Configuration Module for BDD100K Object Detection

This module contains the model selection logic, configuration, and initialization
for the BDD100K object detection task.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ultralytics import YOLO


class ModelType(Enum):
    """Supported model types for object detection."""
    YOLOV8_NANO = "yolov8n"
    YOLOV8_SMALL = "yolov8s"
    YOLOV8_MEDIUM = "yolov8m"
    YOLOV8_LARGE = "yolov8l"
    YOLOV8_XLARGE = "yolov8x"


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    model_type: ModelType
    num_classes: int = 10
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 1000
    
    # BDD100K specific classes
    class_names: List[str] = None
    
    def __post_init__(self):
        """Initialize default class names if not provided."""
        if self.class_names is None:
            self.class_names = [
                'traffic light', 'traffic sign', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ]


class BDDModelSelector:
    """
    Model selector class for BDD100K object detection task.
    
    This class handles model selection, initialization, and configuration
    based on the requirements and constraints of the BDD100K dataset.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model selector.
        
        Args:
            config (ModelConfig): Model configuration object
        """
        self.config = config
        self.model = None
        
    def get_model_specs(self) -> Dict:
        """
        Get detailed specifications for different YOLOv8 variants.
        
        Returns:
            Dict: Model specifications including parameters, GFLOPs, etc.
        """
        specs = {
            ModelType.YOLOV8_NANO: {
                "parameters": "3.2M",
                "gflops": "8.7",
                "inference_speed_cpu": "80.4ms",
                "inference_speed_gpu": "0.99ms",
                "map50_coco": 37.3,
                "recommended_use": "Fast prototyping, edge deployment"
            },
            ModelType.YOLOV8_SMALL: {
                "parameters": "11.2M", 
                "gflops": "28.6",
                "inference_speed_cpu": "128.0ms",
                "inference_speed_gpu": "1.20ms", 
                "map50_coco": 44.9,
                "recommended_use": "Balanced performance, recommended for BDD"
            },
            ModelType.YOLOV8_MEDIUM: {
                "parameters": "25.9M",
                "gflops": "78.9", 
                "inference_speed_cpu": "234.7ms",
                "inference_speed_gpu": "1.83ms",
                "map50_coco": 50.2,
                "recommended_use": "High accuracy requirements"
            },
            ModelType.YOLOV8_LARGE: {
                "parameters": "43.7M",
                "gflops": "165.2",
                "inference_speed_cpu": "375.2ms", 
                "inference_speed_gpu": "2.39ms",
                "map50_coco": 52.9,
                "recommended_use": "Maximum accuracy, research purposes"
            },
            ModelType.YOLOV8_XLARGE: {
                "parameters": "68.2M",
                "gflops": "257.8",
                "inference_speed_cpu": "479.1ms",
                "inference_speed_gpu": "3.53ms", 
                "map50_coco": 53.9,
                "recommended_use": "State-of-the-art accuracy"
            }
        }
        return specs
    
    def select_optimal_model(self, constraints: Dict = None) -> ModelType:
        """
        Select optimal model based on constraints and requirements.
        
        Args:
            constraints (Dict, optional): Performance constraints like max_inference_time,
                                        min_accuracy, memory_limit, etc.
        
        Returns:
            ModelType: Recommended model type
        """
        if constraints is None:
            # Default recommendation for BDD100K: balance of speed and accuracy
            return ModelType.YOLOV8_SMALL
        
        specs = self.get_model_specs()
        
        # Apply constraint-based selection logic
        max_inference_time = constraints.get('max_inference_time_ms', 5.0)  # 5ms default
        min_accuracy = constraints.get('min_map50', 40.0)  # 40 mAP default
        memory_limit = constraints.get('max_parameters_m', 50.0)  # 50M parameters
        
        for model_type in ModelType:
            spec = specs[model_type]
            gpu_time = float(spec['inference_speed_gpu'].replace('ms', ''))
            accuracy = spec['map50_coco']
            params = float(spec['parameters'].replace('M', ''))
            
            if (gpu_time <= max_inference_time and 
                accuracy >= min_accuracy and 
                params <= memory_limit):
                return model_type
        
        # Fallback to smallest model if no constraints met
        return ModelType.YOLOV8_NANO
    
    def initialize_model(self, pretrained: bool = True, weights_path: Optional[str] = None) -> YOLO:
        """
        Initialize the selected model.
        
        Args:
            pretrained (bool): Whether to load pretrained weights
            weights_path (str, optional): Path to custom weights
        
        Returns:
            YOLO: Initialized YOLO model
        """
        if weights_path:
            self.model = YOLO(weights_path)
        elif pretrained:
            self.model = YOLO(f"{self.config.model_type.value}.pt")
        else:
            self.model = YOLO(f"{self.config.model_type.value}.yaml")
        
        # Configure for BDD100K classes
        if hasattr(self.model.model, 'nc'):
            self.model.model.nc = self.config.num_classes
        
        return self.model
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the selected model and its configuration.
        
        Returns:
            str: Model summary
        """
        specs = self.get_model_specs()
        spec = specs[self.config.model_type]
        
        summary = f"""
Model Selection Summary for BDD100K Object Detection:

Selected Model: {self.config.model_type.value.upper()}
Parameters: {spec['parameters']}
GFLOPs: {spec['gflops']}
Expected mAP@0.5 on COCO: {spec['map50_coco']}
GPU Inference Speed: {spec['inference_speed_gpu']}

Configuration:
- Number of Classes: {self.config.num_classes}
- Input Size: {self.config.input_size}
- Confidence Threshold: {self.config.confidence_threshold}
- IoU Threshold: {self.config.iou_threshold}
- Max Detections: {self.config.max_detections}

Class Names: {', '.join(self.config.class_names)}

Justification:
{spec['recommended_use']}
        """
        return summary.strip()


def create_bdd_model_config(model_size: str = "small", 
                           custom_config: Dict = None) -> ModelConfig:
    """
    Create a model configuration for BDD100K dataset.
    
    Args:
        model_size (str): Size variant ('nano', 'small', 'medium', 'large', 'xlarge')
        custom_config (Dict, optional): Custom configuration overrides
    
    Returns:
        ModelConfig: Configured model configuration
    """
    size_mapping = {
        'nano': ModelType.YOLOV8_NANO,
        'small': ModelType.YOLOV8_SMALL, 
        'medium': ModelType.YOLOV8_MEDIUM,
        'large': ModelType.YOLOV8_LARGE,
        'xlarge': ModelType.YOLOV8_XLARGE
    }
    
    model_type = size_mapping.get(model_size.lower(), ModelType.YOLOV8_SMALL)
    
    config = ModelConfig(model_type=model_type)
    
    # Apply custom configuration overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # Example usage
    config = create_bdd_model_config("small")
    selector = BDDModelSelector(config)
    
    print(selector.get_model_summary())
    
    # Initialize model
    model = selector.initialize_model(pretrained=True)
    print(f"\nModel initialized successfully: {type(model).__name__}")
