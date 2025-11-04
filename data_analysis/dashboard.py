"""
Interactive Dashboard for BDD Dataset Analysis

This module creates a comprehensive Dash-based dashboard for visualizing
BDD100k dataset statistics, including class distributions, anomalies,
and comparative analysis between train and validation splits.
"""

import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_analysis import (
    load_labels,
    analyze_class_distribution,
    analyze_train_val_split,
    identify_anomalies,
    analyze_objects_per_image,
    identify_unique_samples,
    identify_extremely_dense_samples,
    identify_class_specific_samples,
    identify_diverse_class_samples,
    identify_extreme_bbox_samples,
    identify_occlusion_samples,
    identify_class_cooccurrence_samples,
)
import os

TRAIN_LABELS_PATH = "data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
VAL_LABELS_PATH = "data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

print("Loading datasets for dashboard...")
# Load and analyze the dataset
train_labels = load_labels(TRAIN_LABELS_PATH)
val_labels = load_labels(VAL_LABELS_PATH)

train_class_counts = analyze_class_distribution(train_labels)
val_class_counts = analyze_class_distribution(val_labels)

# Get additional statistics
combined_df = analyze_train_val_split(train_labels, val_labels)
train_anomalies = identify_anomalies(train_class_counts)
train_obj_per_img = analyze_objects_per_image(train_labels)
val_obj_per_img = analyze_objects_per_image(val_labels)

# Generate unique sample visualizations if not already present
output_dir = "output_samples"
if not os.path.exists(output_dir):
    print("Generating unique sample visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        unique_samples = identify_unique_samples(
            train_labels, 
            "data/bdd100k_images_100k/bdd100k/images/100k/train/",
            output_dir
        )
        print(f"Generated visualizations in {output_dir}/")
    except Exception as e:
        print(f"Warning: Could not generate sample visualizations: {e}")

# Generate extremely dense sample visualizations (60-70 objects)
try:
    print("\n" + "=" * 60)
    extremely_dense = identify_extremely_dense_samples(
        train_labels,
        "data/bdd100k_images_100k/bdd100k/images/100k/train/",
        output_dir,
        min_objects=60,
        max_objects=70
    )
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not generate extremely dense samples: {e}")

# Generate class-specific representative samples
try:
    print("\n" + "=" * 60)
    class_specific = identify_class_specific_samples(
        train_labels,
        "data/bdd100k_images_100k/bdd100k/images/100k/train/",
        output_dir
    )
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not generate class-specific samples: {e}")

# Generate diverse class samples
try:
    print("\n" + "=" * 60)
    diverse_classes = identify_diverse_class_samples(
        train_labels,
        "data/bdd100k_images_100k/bdd100k/images/100k/train/",
        output_dir,
        min_classes=6
    )
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not generate diverse class samples: {e}")

# Generate extreme bbox samples
try:
    print("\n" + "=" * 60)
    extreme_bbox = identify_extreme_bbox_samples(
        train_labels,
        "data/bdd100k_images_100k/bdd100k/images/100k/train/",
        output_dir
    )
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not generate extreme bbox samples: {e}")

# Generate occlusion samples
try:
    print("\n" + "=" * 60)
    occlusion = identify_occlusion_samples(
        train_labels,
        "data/bdd100k_images_100k/bdd100k/images/100k/train/",
        output_dir
    )
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not generate occlusion samples: {e}")

# Generate class co-occurrence samples
try:
    print("\n" + "=" * 60)
    cooccurrence = identify_class_cooccurrence_samples(
        train_labels,
        "data/bdd100k_images_100k/bdd100k/images/100k/train/",
        output_dir
    )
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not generate co-occurrence samples: {e}")

# Organize all samples into structured folders
try:
    print("\n" + "=" * 60)
    from data_analysis import organize_output_samples
    organize_output_samples(output_dir)
    print("=" * 60)
except Exception as e:
    print(f"Warning: Could not organize output samples: {e}")

# Convert class counts to DataFrames
data_train_df = pd.DataFrame(
    list(train_class_counts.items()), columns=["Class", "Train Count"]
)
data_val_df = pd.DataFrame(
    list(val_class_counts.items()), columns=["Class", "Validation Count"]
)

# Ensure class categories are in the same order for train and validation data
all_classes = sorted(set(data_train_df["Class"]).union(set(data_val_df["Class"])))
data_train_df = (
    data_train_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()
)
data_val_df = (
    data_val_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()
)

# Merge for combined visualization
combined_viz_df = pd.merge(data_train_df, data_val_df, on="Class")
combined_viz_df = combined_viz_df.sort_values("Train Count", ascending=False)

# Create a Dash app
app = dash.Dash(__name__)

# Create distinct color palette
colors_train = '#3498db'
colors_val = '#2ecc71'
colors_palette = px.colors.qualitative.Set2

# 1. Combined Train vs Val Comparison - Side by Side Bars
fig_combined = go.Figure()

fig_combined.add_trace(go.Bar(
    name='Training',
    x=combined_viz_df['Class'],
    y=combined_viz_df['Train Count'],
    marker_color=colors_train,
    text=combined_viz_df['Train Count'],
    texttemplate='%{text:,}',
    textposition='outside',
    textfont=dict(size=10),
))

fig_combined.add_trace(go.Bar(
    name='Validation',
    x=combined_viz_df['Class'],
    y=combined_viz_df['Validation Count'],
    marker_color=colors_val,
    text=combined_viz_df['Validation Count'],
    texttemplate='%{text:,}',
    textposition='outside',
    textfont=dict(size=10),
))

fig_combined.update_layout(
    title={
        'text': "Class Distribution: Training vs Validation",
        'font': {'size': 20, 'color': '#2c3e50'},
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_tickangle=-45,
    barmode='group',
    height=550,
    xaxis_title="Object Class",
    yaxis_title="Number of Instances",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(240,240,240,0.5)',
    paper_bgcolor='white',
)

# 2. Logarithmic Scale View for Better Visibility of Small Classes
fig_log_scale = go.Figure()

fig_log_scale.add_trace(go.Bar(
    name='Training',
    x=combined_viz_df['Class'],
    y=combined_viz_df['Train Count'],
    marker_color=colors_train,
    text=combined_viz_df['Train Count'],
    texttemplate='%{text:,}',
    textposition='outside',
    textfont=dict(size=10),
))

fig_log_scale.add_trace(go.Bar(
    name='Validation',
    x=combined_viz_df['Class'],
    y=combined_viz_df['Validation Count'],
    marker_color=colors_val,
    text=combined_viz_df['Validation Count'],
    texttemplate='%{text:,}',
    textposition='outside',
    textfont=dict(size=10),
))

fig_log_scale.update_layout(
    title={
        'text': "Class Distribution: Log Scale (Better visibility for rare classes)",
        'font': {'size': 20, 'color': '#2c3e50'},
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_tickangle=-45,
    barmode='group',
    height=550,
    xaxis_title="Object Class",
    yaxis_title="Number of Instances (Log Scale)",
    yaxis_type="log",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(240,240,240,0.5)',
    paper_bgcolor='white',
)

# 3. Percentage Distribution
fig_percentage = go.Figure()

fig_percentage.add_trace(go.Bar(
    name='Training %',
    x=combined_df['Class'],
    y=combined_df['Train %'],
    marker_color=colors_train,
    text=combined_df['Train %'].apply(lambda x: f'{x:.2f}%'),
    textposition='outside',
    textfont=dict(size=10),
))

fig_percentage.add_trace(go.Bar(
    name='Validation %',
    x=combined_df['Class'],
    y=combined_df['Val %'],
    marker_color=colors_val,
    text=combined_df['Val %'].apply(lambda x: f'{x:.2f}%'),
    textposition='outside',
    textfont=dict(size=10),
))

fig_percentage.update_layout(
    title={
        'text': "Percentage Distribution Across Classes",
        'font': {'size': 20, 'color': '#2c3e50'},
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_tickangle=-45,
    barmode='group',
    height=550,
    xaxis_title="Object Class",
    yaxis_title="Percentage (%)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(240,240,240,0.5)',
    paper_bgcolor='white',
)

# 4. Anomalies Visualization
if train_anomalies:
    anomaly_df = pd.DataFrame(
        list(train_anomalies.items()), columns=["Class", "Count"]
    )
    anomaly_df = anomaly_df.sort_values("Count", ascending=True)
    
    fig_anomalies = go.Figure()
    fig_anomalies.add_trace(go.Bar(
        x=anomaly_df['Count'],
        y=anomaly_df['Class'],
        orientation='h',
        marker=dict(
            color=anomaly_df['Count'],
            colorscale='Reds',
            showscale=False
        ),
        text=anomaly_df['Count'],
        texttemplate='%{text:,}',
        textposition='outside',
    ))
    
    fig_anomalies.update_layout(
        title={
            'text': "⚠️ Underrepresented Classes (<1% of total)",
            'font': {'size': 20, 'color': '#e74c3c'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        xaxis_title="Number of Instances",
        yaxis_title="Class",
        plot_bgcolor='rgba(255,240,240,0.5)',
        paper_bgcolor='white',
    )
else:
    fig_anomalies = go.Figure()
    fig_anomalies.add_annotation(
        text="✓ No anomalies detected - All classes well represented (>1%)",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, color='#27ae60'),
    )
    fig_anomalies.update_layout(
        height=400,
        plot_bgcolor='rgba(240,255,240,0.5)',
    )

# 5. Objects Per Image Distribution
fig_obj_per_img = go.Figure()

fig_obj_per_img.add_trace(go.Histogram(
    x=train_obj_per_img["object_count"],
    name="Training",
    opacity=0.75,
    marker_color=colors_train,
    nbinsx=30,
))

fig_obj_per_img.add_trace(go.Histogram(
    x=val_obj_per_img["object_count"],
    name="Validation",
    opacity=0.75,
    marker_color=colors_val,
    nbinsx=30,
))

fig_obj_per_img.update_layout(
    title={
        'text': "Distribution of Objects per Image",
        'font': {'size': 20, 'color': '#2c3e50'},
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title="Number of Objects per Image",
    yaxis_title="Frequency (Number of Images)",
    barmode='overlay',
    height=450,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='rgba(240,240,240,0.5)',
    paper_bgcolor='white',
)

# Calculate summary statistics
total_train_objects = sum(train_class_counts.values())
total_val_objects = sum(val_class_counts.values())
num_train_images = len(train_labels)
num_val_images = len(val_labels)

# Count images with box2d annotations
num_train_images_with_box2d = sum(
    1 for item in train_labels 
    if any('box2d' in label for label in item.get('labels', []))
)
num_val_images_with_box2d = sum(
    1 for item in val_labels 
    if any('box2d' in label for label in item.get('labels', []))
)

# Count images without box2d annotations
num_train_images_no_box2d = num_train_images - num_train_images_with_box2d
num_val_images_no_box2d = num_val_images - num_val_images_with_box2d

avg_train_objects = total_train_objects / num_train_images_with_box2d if num_train_images_with_box2d > 0 else 0
avg_val_objects = total_val_objects / num_val_images_with_box2d if num_val_images_with_box2d > 0 else 0

# Define the layout of the app
app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1(
                    "BDD100k Object Detection Dataset Analysis",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "padding": "20px 0 10px 0",
                        "margin": "0",
                        "fontWeight": "bold"
                    },
                ),
                html.P(
                    "Comprehensive Analysis of Bounding Box Annotations for 10 Object Classes",
                    style={
                        "textAlign": "center",
                        "color": "#7f8c8d",
                        "fontSize": "16px",
                        "margin": "0 0 20px 0",
                    },
                ),
            ],
            style={
                "backgroundColor": "#ecf0f1",
                "marginBottom": "30px",
                "borderBottom": "3px solid #3498db"
            },
        ),
        
        # Summary Statistics
        html.Div(
            [
                html.H2("📊 Dataset Summary", style={"color": "#2c3e50", "marginBottom": "25px", "textAlign": "center"}),
                
                # Key Metrics Row
                html.Div(
                    [
                        # Training Images
                        html.Div(
                            [
                                html.Div([
                                    html.H3(f"{num_train_images_with_box2d:,}", style={"color": "#3498db", "margin": "0", "fontSize": "42px", "fontWeight": "bold"}),
                                    html.P("Training Images", style={"color": "#34495e", "margin": "8px 0 0 0", "fontSize": "16px", "fontWeight": "600"}),
                                    html.P("with bounding boxes", style={"color": "#7f8c8d", "margin": "3px 0 0 0", "fontSize": "12px"}),
                                ], style={"textAlign": "center"}),
                            ],
                            style={
                                "flex": "1",
                                "padding": "30px 20px",
                                "backgroundColor": "#ffffff",
                                "margin": "10px",
                                "borderRadius": "12px",
                                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                "borderTop": "4px solid #3498db",
                            },
                        ),
                        
                        # Validation Images
                        html.Div(
                            [
                                html.Div([
                                    html.H3(f"{num_val_images_with_box2d:,}", style={"color": "#2ecc71", "margin": "0", "fontSize": "42px", "fontWeight": "bold"}),
                                    html.P("Validation Images", style={"color": "#34495e", "margin": "8px 0 0 0", "fontSize": "16px", "fontWeight": "600"}),
                                    html.P("with bounding boxes", style={"color": "#7f8c8d", "margin": "3px 0 0 0", "fontSize": "12px"}),
                                ], style={"textAlign": "center"}),
                            ],
                            style={
                                "flex": "1",
                                "padding": "30px 20px",
                                "backgroundColor": "#ffffff",
                                "margin": "10px",
                                "borderRadius": "12px",
                                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                "borderTop": "4px solid #2ecc71",
                            },
                        ),
                        
                        # Total Training Objects
                        html.Div(
                            [
                                html.Div([
                                    html.H3(f"{total_train_objects:,}", style={"color": "#e74c3c", "margin": "0", "fontSize": "42px", "fontWeight": "bold"}),
                                    html.P("Training Annotations", style={"color": "#34495e", "margin": "8px 0 0 0", "fontSize": "16px", "fontWeight": "600"}),
                                    html.P("total bounding boxes", style={"color": "#7f8c8d", "margin": "3px 0 0 0", "fontSize": "12px"}),
                                ], style={"textAlign": "center"}),
                            ],
                            style={
                                "flex": "1",
                                "padding": "30px 20px",
                                "backgroundColor": "#ffffff",
                                "margin": "10px",
                                "borderRadius": "12px",
                                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                "borderTop": "4px solid #e74c3c",
                            },
                        ),
                        
                        # Average Objects
                        html.Div(
                            [
                                html.Div([
                                    html.H3(f"{avg_train_objects:.1f}", style={"color": "#9b59b6", "margin": "0", "fontSize": "42px", "fontWeight": "bold"}),
                                    html.P("Avg per Image", style={"color": "#34495e", "margin": "8px 0 0 0", "fontSize": "16px", "fontWeight": "600"}),
                                    html.P("objects in training", style={"color": "#7f8c8d", "margin": "3px 0 0 0", "fontSize": "12px"}),
                                ], style={"textAlign": "center"}),
                            ],
                            style={
                                "flex": "1",
                                "padding": "30px 20px",
                                "backgroundColor": "#ffffff",
                                "margin": "10px",
                                "borderRadius": "12px",
                                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                                "borderTop": "4px solid #9b59b6",
                            },
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap", "marginBottom": "30px"},
                ),
                
                # Explanation Note
                html.Div(
                    [
                        html.P([
                            html.Strong("ℹ️ Important: "),
                            f"The training folder contains approximately 70,001 image files. However, only {num_train_images:,} images are included in the labels JSON file. ",
                            f"This means ~138 images exist in the folder but have no annotations. ",
                            html.Br(),
                            html.Br(),
                            f"Of the {num_train_images:,} labeled images, {num_train_images_with_box2d:,} contain bounding box (box2d) annotations for object detection, ",
                            html.Br(),
                            html.Br(),
                            html.Strong("Our analysis focuses exclusively on the {num_train_images_with_box2d:,} images with bounding boxes for the 10 object detection classes."),
                        ], style={"margin": "0", "lineHeight": "1.8", "fontSize": "14px"}),
                    ],
                    style={
                        "padding": "20px 25px",
                        "backgroundColor": "#fff3cd",
                        "borderLeft": "5px solid #ffc107",
                        "borderRadius": "8px",
                        "color": "#856404",
                    },
                ),
            ],
            style={
                "marginBottom": "40px",
                "padding": "30px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "12px",
            },
        ),
        
        # Main Visualizations
        html.Div([
            html.H2("📈 Class Distribution Analysis", style={"color": "#2c3e50", "marginBottom": "25px", "textAlign": "center"}),
            
            # Combined comparison
            html.Div([
                dcc.Graph(figure=fig_combined, config={'displayModeBar': False}),
            ], style={"marginBottom": "30px", "backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
            
            # Log scale view
            html.Div([
                dcc.Graph(figure=fig_log_scale, config={'displayModeBar': False}),
            ], style={"marginBottom": "30px", "backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
            
            # Percentage distribution
            html.Div([
                dcc.Graph(figure=fig_percentage, config={'displayModeBar': False}),
            ], style={"marginBottom": "30px", "backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
        ]),
        
        # Anomalies Section
        html.Div([
            html.H2("⚠️ Data Quality Analysis", style={"color": "#2c3e50", "marginBottom": "25px", "textAlign": "center"}),
            html.Div([
                dcc.Graph(figure=fig_anomalies, config={'displayModeBar': False}),
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
        ], style={"marginBottom": "40px"}),
        
        # Objects Per Image Section
        html.Div([
            html.H2("📊 Object Density Analysis", style={"color": "#2c3e50", "marginBottom": "25px", "textAlign": "center"}),
            html.Div([
                dcc.Graph(figure=fig_obj_per_img, config={'displayModeBar': False}),
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
        ], style={"marginBottom": "40px"}),
        
        # Footer
        html.Div(
            [
                html.P(
                    "BDD100k Dataset Analysis Dashboard | Object Detection Focus | 10 Classes",
                    style={"textAlign": "center", "color": "#7f8c8d", "margin": "0", "padding": "20px"},
                )
            ],
            style={"backgroundColor": "#ecf0f1", "borderTop": "3px solid #3498db"},
        ),
    ],
    style={
        "padding": "0",
        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
    },
)

# Run the app
if __name__ == "__main__":
    print("Starting dashboard server...")
    print("Dashboard will be available at http://0.0.0.0:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
