from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import sys
import pathlib
import glob
import json
import time
import uuid
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from werkzeug.utils import secure_filename
import zipfile
import io
import os.path
import shutil
import tempfile
import datetime
import re
from functools import wraps
from flask import g

# Import the SAM modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Add these imports at the top of the file if not already present
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure app
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MASKS_FOLDER'] = './masks'
app.config['ANNOTATIONS_FOLDER'] = './annotations'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['TEMP_DOWNLOADS'] = './temp_downloads'

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MASKS_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATIONS_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_DOWNLOADS'], exist_ok=True)

# Store sessions
sessions = {}

# Game definitions - each game has specific classes and values
GAMES = {
    "Poker": {
        "classes": ["card", "token", "dice"],
        "card_values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"],
        "token_values": ["white", "red", "blue", "green", "black"],
        "dice_values": ["1", "2", "3", "4", "5", "6"]
    },
    "Splendor": {
        "classes": ["card", "token", "dice"],
        "card_values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "token_values": ["white", "red", "blue", "green", "black", "yellow"],
        "dice_values": ["1", "2", "3", "4", "5", "6"]
    },
    "Azul": {
        "classes": ["card", "token", "dice"],
        "card_values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "token_values": ["white", "red", "blue", "green", "black"],
        "dice_values": ["1", "2", "3", "4", "5", "6"]
    },
    "Concept": {
        "classes": ["card", "token", "dice"],
        "card_values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "token_values": ["white", "red", "blue", "green", "black"],
        "dice_values": ["1", "2", "3", "4", "5", "6"]
    },
    "Uno": {
        "classes": ["card"],
        "card_values": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "skip", "reverse", "draw2", "wild", "wild_draw4"]
    },
    "Chess": {
        "classes": ["piece"],
        "piece_values": ["white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
                         "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"]
    },
    "Other": {
        "classes": ["card", "token", "dice"],
        "card_values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "joker", "none"],
        "token_values": ["generic"],
        "dice_values": ["1", "2", "3", "4", "5", "6"]
    }
}

# Available shapes
SHAPES = ["rectangle", "circle", "triangle", "irregular"]

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_predictor(config_key, compute_device):
    """Create a SAM2 image predictor with the given config"""
    config_files = {
        "sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    checkpoint_files = {
        "sam2.1-hiera-tiny": "checkpoints/sam2.1_hiera_tiny.pt",
        "sam2.1-hiera-small": "checkpoints/sam2.1_hiera_small.pt",
        "sam2.1-hiera-base-plus": "checkpoints/sam2.1_hiera_base_plus.pt",
        "sam2.1-hiera-large": "checkpoints/sam2.1_hiera_large.pt",
    }

    cfg_file = config_files[config_key]
    ckpt_file = checkpoint_files[config_key]

    segmenter = SAM2ImagePredictor(
        build_sam2(cfg_file, ckpt_file, compute_device))
    return segmenter

def get_values_for_class(game, class_name):
    """Get appropriate values based on the class and current game"""
    game_data = GAMES[game]

    # Use the class name to determine which values to use
    value_key = f"{class_name}_values"
    if value_key in game_data:
        return game_data[value_key]
    return ["none"]  # Default if no specific values for this class

@app.route('/')
def index():
    """Render the main annotation interface"""
    return render_template('index.html', games=GAMES, shapes=SHAPES)

@app.route('/api/init_session', methods=['POST'])
def init_session():
    """Initialize a new annotation session"""
    data = request.json
    model = data.get('model', 'sam2.1-hiera-tiny')
    device = data.get('device', 'cpu')
    game = data.get('game', 'Poker')
    
    try:
        # Create a unique session ID
        session_id = str(uuid.uuid4())
        
        # Initialize the segmenter
        segmenter = create_predictor(model, device)
        
        # Store session data
        sessions[session_id] = {
            'segmenter': segmenter,
            'model': model,
            'device': device,
            'game': game,
            'images': [],
            'current_index': 0,
            'mask_storage': [],
            'user_prompts': [],
            'annotation_counter': [],
            'object_metadata': {},
            'manual_counts': {},
            'legends': {},
            'color_palette': []
        }
        
        # Initialize with some random colors for objects
        for _ in range(10):  # Start with 10 colors
            sessions[session_id]['color_palette'].append(
                [np.random.rand(), np.random.rand(), np.random.rand()]
            )
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'game': game,
            'classes': GAMES[game]['classes'],
            'shapes': SHAPES,
            'values': get_values_for_class(game, GAMES[game]['classes'][0])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/upload_images', methods=['POST'])
def upload_images():
    """Upload images to the server"""
    session_id = request.form.get('session_id')
    upload_type = request.form.get('upload_type', 'files')  # 'files' or 'folder'
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
        
    session = sessions[session_id]
    
    if 'files' not in request.files:
        return jsonify({'status': 'error', 'message': 'No files provided'}), 400
        
    files = request.files.getlist('files')
    
    uploaded_images = []
    
    # If folder upload, organize files by subfolder structure
    if upload_type == 'folder':
        # Extract folder structure from file paths
        folder_structure = {}
        for file in files:
            if file and allowed_file(file.filename):
                # Handle possible subdirectory structure
                path_parts = file.filename.split('/')
                filename = path_parts[-1]  # The actual filename is the last part
                
                # Save the file with its structure preserved
                relative_path = os.path.join('folder_uploads', *path_parts)
                os.makedirs(os.path.dirname(os.path.join(app.config['UPLOAD_FOLDER'], relative_path)), exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], relative_path)
                file.save(filepath)
                
                # Add image to session
                session['images'].append(filepath)
                
                # Initialize storage for this image
                img = Image.open(filepath)
                img_array = np.array(img.convert("RGB"))
                
                # Set the image for the segmenter
                session['segmenter'].set_image(img_array)
                
                # Initialize masks and prompts
                session['mask_storage'].append(None)
                session['user_prompts'].append({"positive": [], "negative": [], "box": None})
                session['annotation_counter'].append(1)
                
                uploaded_images.append({
                    'path': filepath,
                    'name': filename
                })
    else:
        # Standard file upload (original behavior)
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Add image to session
                session['images'].append(filepath)
                
                # Initialize storage for this image
                img = Image.open(filepath)
                img_array = np.array(img.convert("RGB"))
                
                # Set the image for the segmenter
                session['segmenter'].set_image(img_array)
                
                # Initialize masks and prompts
                session['mask_storage'].append(None)
                session['user_prompts'].append({"positive": [], "negative": [], "box": None})
                session['annotation_counter'].append(1)
                
                uploaded_images.append({
                    'path': filepath,
                    'name': filename
                })
    
    return jsonify({
        'status': 'success',
        'images': uploaded_images,
        'count': len(uploaded_images)
    })

@app.route('/api/get_images', methods=['GET'])
def get_images():
    """Get all images for the current session"""
    session_id = request.args.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
        
    session = sessions[session_id]
    
    image_list = []
    for img_path in session['images']:
        image_list.append({
            'path': img_path,
            'name': os.path.basename(img_path)
        })
    
    return jsonify({
        'status': 'success',
        'images': image_list,
        'count': len(image_list)
    })

@app.route('/api/image/<path:filename>')
def serve_image(filename):
    """Serve an image file"""
    return send_from_directory(os.path.dirname(filename), 
                              os.path.basename(filename))

@app.route('/api/change_image', methods=['POST'])
def change_image():
    """Change the current image"""
    data = request.json
    session_id = data.get('session_id')
    index = data.get('index', 0)
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
        
    session = sessions[session_id]
    
    if index < 0 or index >= len(session['images']):
        return jsonify({'status': 'error', 'message': 'Invalid image index'}), 400
    
    # Update current image index
    session['current_index'] = index
    
    # Load the image
    img_path = session['images'][index]
    img = Image.open(img_path)
    img_array = np.array(img.convert("RGB"))
    
    # Set the image for the segmenter
    session['segmenter'].set_image(img_array)
    
    # Initialize mask storage if needed
    if len(session['mask_storage']) <= index:
        session['mask_storage'].append(None)
    
    # Initialize user prompts if needed
    if len(session['user_prompts']) <= index:
        session['user_prompts'].append({"positive": [], "negative": [], "box": None})
    
    # Initialize annotation counter if needed
    if len(session['annotation_counter']) <= index:
        session['annotation_counter'].append(1)
    
    # Get metadata for this image
    metadata = session['object_metadata'].get(index, {})
    manual_counts = session['manual_counts'].get(index, {})
    legend = session['legends'].get(index, "")
    
    # Generate display image with masks
    display_img = None
    if session['mask_storage'][index] is not None:
        display_img = generate_display_image(session, index)
    
    return jsonify({
        'status': 'success',
        'image_path': img_path,
        'image_name': os.path.basename(img_path),
        'metadata': metadata,
        'manual_counts': manual_counts,
        'legend': legend,
        'display_image': display_img,
        'width': img.width,
        'height': img.height
    })

def generate_display_image(session, index):
    """Generate a display image with mask overlay"""
    if index >= len(session['images']):
        return None
        
    # Load the original image
    img_path = session['images'][index]
    img = Image.open(img_path)
    img_array = np.array(img.convert("RGB"))
    
    # If no mask exists yet, return the original image encoded
    if session['mask_storage'][index] is None:
        img_pil = Image.fromarray(img_array)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    # Create mask overlay
    normalized_img = img_array / 255.0
    overlay = normalized_img.copy()
    
    # Apply masks for all annotated objects
    max_id = session['annotation_counter'][index]
    for i in range(1, max_id):
        mask_area = session['mask_storage'][index] == i
        if np.any(mask_area):
            color_idx = min(i-1, len(session['color_palette'])-1)
            overlay[mask_area] = (
                0.5 * normalized_img[mask_area] + 
                0.5 * np.array(session['color_palette'][color_idx])
            )
    
    # Convert to PIL and encode
    display_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buffered = BytesIO()
    display_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

@app.route('/api/run_segmentation', methods=['POST'])
def run_segmentation():
    """Run segmentation with the current prompts"""
    data = request.json
    session_id = data.get('session_id')
    obj_class = data.get('class')
    obj_shape = data.get('shape')
    obj_color = data.get('color')
    obj_color_hex = data.get('color_hex', '#000000')  # Get hex color with default
    obj_value = data.get('value')
    is_preview = data.get('is_preview', False)  # Flag to indicate if this is just a preview
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
        
    session = sessions[session_id]
    index = session['current_index']
    
    if index >= len(session['images']):
        return jsonify({'status': 'error', 'message': 'Invalid image index'}), 400
    
    # Load the image
    img_path = session['images'][index]
    img = Image.open(img_path)
    img_array = np.array(img.convert("RGB"))
    
    # Initialize mask storage if needed
    if session['mask_storage'][index] is None:
        session['mask_storage'][index] = np.zeros(img_array.shape[:2], dtype=np.int32)
    
    # Get user prompts
    pos_points = session['user_prompts'][index]["positive"]
    neg_points = session['user_prompts'][index]["negative"]
    bbox = session['user_prompts'][index]["box"]
    
    # Run segmentation
    ann_id = session['annotation_counter'][index]
    seg_input = {}
    
    # For previews, we use a temporary mask or just visualize results without changing the stored mask
    if not is_preview:
        # Clear previous mask with this id only (keep other objects' masks)
        prev_mask = session['mask_storage'][index] == ann_id
        session['mask_storage'][index][prev_mask] = 0
    
    # Set up segmentation inputs
    if pos_points or neg_points:
        all_points = np.array(pos_points + neg_points, dtype=np.float32)
        labels = np.array(
            ([1] * len(pos_points)) + ([0] * len(neg_points)), dtype=np.int32
        )
        seg_input["point_coords"] = all_points
        seg_input["point_labels"] = labels
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        seg_input["box"] = [x1, y1, x2, y2]
    
    # Run the segmentation only if we have some input
    if seg_input:
        try:
            with torch.inference_mode(), torch.autocast(
                session['device'], dtype=torch.bfloat16
            ):
                masks, scores, _ = session['segmenter'].predict(
                    **seg_input,
                    multimask_output=True,
                )
                sorted_idx = np.argsort(scores)[::-1]
                masks = masks[sorted_idx[0]]
            
            # For preview, we create a temporary display image without modifying the stored mask
            if is_preview:
                # Create a copy of the original image
                normalized_img = img_array.copy() / 255.0
                overlay = normalized_img.copy()
                
                # Apply existing masks from stored data (other objects)
                for i in range(1, session['annotation_counter'][index]):
                    if i == ann_id:  # Skip current object's mask for preview
                        continue
                    mask_area = session['mask_storage'][index] == i
                    if np.any(mask_area):
                        color_idx = min(i-1, len(session['color_palette'])-1)
                        overlay[mask_area] = (
                            0.5 * normalized_img[mask_area] + 
                            0.5 * np.array(session['color_palette'][color_idx])
                        )
                
                # Apply the new preview mask with a distinct color (using blue)
                preview_color = np.array([0, 0, 1])  # Blue for preview
                overlay[masks == 1.0] = (
                    0.5 * normalized_img[masks == 1.0] + 
                    0.5 * preview_color
                )
                
                # Convert to PIL and encode
                display_img = Image.fromarray((overlay * 255).astype(np.uint8))
                buffered = BytesIO()
                display_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                display_image_data = f"data:image/png;base64,{img_str}"
                
            else:  # Normal (non-preview) mode
                # Update mask storage - this keeps existing object masks
                # Only update background areas to avoid overwriting other objects
                bg_mask = session['mask_storage'][index] == 0
                session['mask_storage'][index][np.logical_and(bg_mask, masks == 1.0)] = ann_id
                
                # Generate standard display image
                display_image_data = generate_display_image(session, index)
            
            # Calculate centroid and area for the mask
            y_indices, x_indices = np.where(masks == 1.0)
            centroid = None
            area = 0
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                centroid_x = float(np.mean(x_indices))
                centroid_y = float(np.mean(y_indices))
                centroid = [centroid_x, centroid_y]
                area = len(y_indices)
            
            # Store object metadata if this is not a preview
            object_metadata = None
            if not is_preview:
                if index not in session['object_metadata']:
                    session['object_metadata'][index] = {}
                
                object_metadata = {
                    "id": ann_id,
                    "class": obj_class,
                    "shape": obj_shape,
                    "color": obj_color,          # Human-readable color name
                    "color_hex": obj_color_hex,  # Hex value for the color
                    "value": obj_value,
                    "bbox": bbox,
                    "centroid": centroid,
                    "area": area
                }
                
                session['object_metadata'][index][ann_id] = object_metadata
            
            return jsonify({
                'status': 'success',
                'display_image': display_image_data,
                'object_id': ann_id,
                'centroid': centroid,
                'area': area,
                'is_preview': is_preview,
                'metadata': object_metadata
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Segmentation error: {str(e)}'
            }), 500
    else:
        return jsonify({
            'status': 'error',
            'message': 'No segmentation inputs (box or points) provided'
        }), 400

@app.route('/api/add_point', methods=['POST'])
def add_point():
    """Add a point (positive or negative) to the current image"""
    data = request.json
    session_id = data.get('session_id')
    point_type = data.get('type')  # 'positive' or 'negative'
    x = data.get('x')
    y = data.get('y')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    # Add point to the appropriate list
    if point_type == 'positive':
        session['user_prompts'][index]["positive"].append([int(x), int(y)])
    else:
        session['user_prompts'][index]["negative"].append([int(x), int(y)])
    
    return jsonify({
        'status': 'success',
        'point_type': point_type,
        'x': x,
        'y': y,
        'points': {
            'positive': session['user_prompts'][index]["positive"],
            'negative': session['user_prompts'][index]["negative"]
        }
    })

@app.route('/api/set_box', methods=['POST'])
def set_box():
    """Set a bounding box for the current image"""
    data = request.json
    session_id = data.get('session_id')
    x1 = int(data.get('x1'))
    y1 = int(data.get('y1'))
    x2 = int(data.get('x2'))
    y2 = int(data.get('y2'))
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    # Set the box with proper ordering (min/max)
    session['user_prompts'][index]["box"] = [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2)
    ]
    
    return jsonify({
        'status': 'success',
        'box': session['user_prompts'][index]["box"]
    })

@app.route('/api/reset_current', methods=['POST'])
def reset_current():
    """Reset the current object annotation"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    # Reset current mask
    if session['mask_storage'][index] is not None:
        ann_id = session['annotation_counter'][index]
        session['mask_storage'][index][session['mask_storage'][index] == ann_id] = 0
    
    # Reset prompts
    session['user_prompts'][index]["positive"] = []
    session['user_prompts'][index]["negative"] = []
    session['user_prompts'][index]["box"] = None
    
    # Remove metadata for this object
    if index in session['object_metadata'] and ann_id in session['object_metadata'][index]:
        del session['object_metadata'][index][ann_id]
    
    # Generate updated display image
    display_img = generate_display_image(session, index)
    
    return jsonify({
        'status': 'success',
        'display_image': display_img
    })

@app.route('/api/create_new_object', methods=['POST'])
def create_new_object():
    """Increment the annotation counter to create a new object"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    # Increment annotation counter
    session['annotation_counter'][index] += 1
    
    # Reset prompts for new object but keep the existing masks
    session['user_prompts'][index]["positive"] = []
    session['user_prompts'][index]["negative"] = []
    session['user_prompts'][index]["box"] = None
    
    return jsonify({
        'status': 'success',
        'new_object_id': session['annotation_counter'][index]
    })

@app.route('/api/set_manual_count', methods=['POST'])
def set_manual_count():
    """Set manual object counts for the current image"""
    data = request.json
    session_id = data.get('session_id')
    counts = data.get('counts', {})
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    session['manual_counts'][index] = counts
    
    return jsonify({
        'status': 'success',
        'counts': counts
    })

@app.route('/api/set_legend', methods=['POST'])
def set_legend():
    """Set a legend for the current image"""
    data = request.json
    session_id = data.get('session_id')
    legend = data.get('legend', "")
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    session['legends'][index] = legend
    
    return jsonify({
        'status': 'success',
        'legend': legend
    })

@app.route('/api/change_game', methods=['POST'])
def change_game():
    """Change the current game and update available classes"""
    data = request.json
    session_id = data.get('session_id')
    game = data.get('game')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    if game not in GAMES:
        return jsonify({'status': 'error', 'message': 'Invalid game'}), 400
    
    session['game'] = game
    
    return jsonify({
        'status': 'success',
        'game': game,
        'classes': GAMES[game]['classes'],
        'values': get_values_for_class(game, GAMES[game]['classes'][0])
    })

@app.route('/api/clear_all_annotations', methods=['POST'])
def clear_all_annotations():
    """Clear all annotations for the current image"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    index = session['current_index']
    
    # Reset mask storage
    if session['mask_storage'][index] is not None:
        session['mask_storage'][index][:] = 0
    
    # Reset annotation counter
    session['annotation_counter'][index] = 1
    
    # Reset prompts
    session['user_prompts'][index]["positive"] = []
    session['user_prompts'][index]["negative"] = []
    session['user_prompts'][index]["box"] = None
    
    # Clear metadata
    if index in session['object_metadata']:
        session['object_metadata'][index] = {}
    
    # Generate updated display image - will be original image
    display_img = generate_display_image(session, index)
    
    return jsonify({
        'status': 'success',
        'display_image': display_img
    })

@app.route('/api/export_annotations', methods=['POST'])
def export_annotations():
    """Export annotations for all images in the session"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    export_results = []
    
    for i in range(len(session['images'])):
        if ((session['mask_storage'][i] is not None and np.any(session['mask_storage'][i])) or
            (i in session['manual_counts'] and any(session['manual_counts'][i].values())) or
            (i in session['legends'] and session['legends'][i])):
            
            base_name = os.path.splitext(os.path.basename(session['images'][i]))[0]
            json_path = os.path.join(app.config['ANNOTATIONS_FOLDER'], f"{base_name}_annotation.json")
            
            annotation_data = {
                "image": session['images'][i],
                "game": session['game'],
                "objects": [],
                "manual_counts": session['manual_counts'].get(i, {}),
                "legend": session['legends'].get(i, ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add object metadata
            if i in session['object_metadata']:
                for obj_id, obj_data in session['object_metadata'][i].items():
                    annotation_data["objects"].append(obj_data)
            
            # Save as JSON
            with open(json_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            
            export_results.append({
                "image": os.path.basename(session['images'][i]),
                "annotations": json_path
            })
    
    return jsonify({
        'status': 'success',
        'results': export_results,
        'count': len(export_results)
    })

@app.route('/api/export_masks', methods=['POST'])
def export_masks():
    """Export masks for all images in the session"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    export_results = []
    
    for i, mask in enumerate(session['mask_storage']):
        if mask is not None and np.any(mask):
            base_name = os.path.splitext(os.path.basename(session['images'][i]))[0]
            mask_path = os.path.join(app.config['MASKS_FOLDER'], f"{base_name}_mask.png")
            
            # Create a colored visualization of the mask
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            for obj_id in range(1, session['annotation_counter'][i]):
                if obj_id-1 < len(session['color_palette']): 
                    color = (np.array(session['color_palette'][obj_id-1]) * 255).astype(np.uint8)
                else:
                    color = np.random.randint(0, 255, 3, dtype=np.uint8)
                
                colored_mask[mask == obj_id] = color
            
            # Save the mask
            Image.fromarray(colored_mask).save(mask_path)
            
            export_results.append({
                "image": os.path.basename(session['images'][i]),
                "mask": mask_path
            })
    
    return jsonify({
        'status': 'success',
        'results': export_results,
        'count': len(export_results)
    })

@app.route('/api/generate_legend', methods=['POST'])
def generate_legend():
    """Generate a legend using Azure OpenAI API"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
            
        session = sessions[session_id]
        index = session['current_index']
        
        # Collect class counts for this image
        class_counts = {}
        for cls in GAMES[session['game']]['classes']:
            class_counts[cls] = 0
        
        # Count objects from metadata
        if index in session['object_metadata']:
            for obj_id, obj_data in session['object_metadata'][index].items():
                obj_class = obj_data.get("class", "")
                if obj_class in class_counts:
                    class_counts[obj_class] += 1
        
        # Add counts from manual counts
        if index in session['manual_counts']:
            for cls, count in session['manual_counts'][index].items():
                if cls in class_counts:
                    class_counts[cls] += count
        
        # Filter out classes with zero count
        actual_objects = {cls: count for cls, count in class_counts.items() if count > 0}
        
        objects_text = ", ".join([f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in actual_objects.items()])
        
        # Build the prompt
        prompt = (
            f"Create a concise description in French for an image containing: {objects_text}.\n"
            f"The description should mention all objects and describe their relative positions. "
            f"Start with 'Dans cette image, on peut voir...' and be specific about quantities. "
            f"Keep it under 3 sentences."
        )
        
        # Get API Key
        API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        if not API_KEY:
            return jsonify({
                'status': 'error',
                'message': 'AZURE_OPENAI_API_KEY environment variable not set'
            }), 400
        
        # ---- OpenAI API Configuration ----
        API_VERSION = "2023-03-15-preview"
        AZURE_ENDPOINT = "https://ai-rd-sweden.openai.azure.com/"
        
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=API_KEY, 
            api_version=API_VERSION, 
            azure_endpoint=AZURE_ENDPOINT
        )
        
        # Call API to generate text
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the appropriate model deployed on your Azure instance
            messages=[
                {"role": "system", "content": "You are an assistant that creates concise image descriptions in French based on object counts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        # Extract the generated text
        generated_text = response.choices[0].message.content
        
        # Save the generated legend
        session['legends'][index] = generated_text
        
        return jsonify({
            'status': 'success',
            'legend': generated_text
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Legend generation failed: {str(e)}'
        }), 500

@app.route('/api/download_annotations', methods=['POST'])
def download_annotations():
    """Create a downloadable zip file of annotations with folder structure preserved"""
    data = request.json
    session_id = data.get('session_id')
    preserve_structure = data.get('preserve_structure', True)
    
    logger.info(f"Download annotations request received: session_id={session_id}, preserve_structure={preserve_structure}")
    
    if session_id not in sessions:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    try:
        # Create a unique identifier for this download
        download_id = f"annotations_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        # Create a temporary directory to store the files
        temp_dir = os.path.join(app.config['TEMP_DOWNLOADS'], download_id)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Path to the zip file we'll create
        zip_filename = os.path.join(temp_dir, 'annotations.zip')
        
        # Count of included annotations
        count = 0
        
        # Create a mapping of original paths to help preserve structure
        path_mapping = {}
        for i, img_path in enumerate(session['images']):
            # For files uploaded as part of a folder, keep their structure
            if 'folder_uploads' in img_path and preserve_structure:
                # Extract the relative path more carefully
                parts = img_path.split('folder_uploads')
                if len(parts) > 1:
                    relative_path = parts[1].lstrip('/')
                    parent_dir = os.path.dirname(relative_path)
                else:
                    # Fallback if splitting doesn't work as expected
                    parent_dir = ""
                filename = os.path.basename(img_path)
            else:
                # For individual uploads, just use the filename
                parent_dir = ""
                filename = os.path.basename(img_path)
            
            path_mapping[i] = {'dir': parent_dir, 'filename': filename}
            logger.debug(f"Path mapping for image {i}: {path_mapping[i]}")
        
        # Create the zip file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i in range(len(session['images'])):
                # Check if we have any annotations for this image
                has_annotations = False
                
                # Check for masks
                if i < len(session['mask_storage']) and session['mask_storage'][i] is not None:
                    has_annotations = has_annotations or np.any(session['mask_storage'][i])
                
                # Check for manual counts
                if i in session['manual_counts']:
                    has_annotations = has_annotations or any(session['manual_counts'][i].values())
                
                # Check for legends
                if i in session['legends']:
                    has_annotations = has_annotations or bool(session['legends'][i])
                    
                # Check for object metadata
                if i in session['object_metadata']:
                    has_annotations = has_annotations or bool(session['object_metadata'][i])
                
                if has_annotations:
                    # Get mapping info
                    mapping = path_mapping[i]
                    parent_dir = mapping['dir']
                    base_name = os.path.splitext(mapping['filename'])[0]
                    
                    # Create annotation data
                    annotation_data = {
                        "image": os.path.basename(session['images'][i]),  # Store just the filename, not the full path
                        "game": session['game'],
                        "objects": [],
                        "manual_counts": session['manual_counts'].get(i, {}),
                        "legend": session['legends'].get(i, ""),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Add object metadata
                    if i in session['object_metadata']:
                        for obj_id, obj_data in session['object_metadata'][i].items():
                            annotation_data["objects"].append(obj_data)
                    
                    # Convert to JSON string with proper formatting
                    json_data = json.dumps(annotation_data, indent=2)
                    
                    # Add to zip with preserved structure if requested
                    json_path = f"{parent_dir}/{base_name}_annotation.json" if parent_dir else f"{base_name}_annotation.json"
                    
                    # Log the path being added
                    logger.info(f"Adding to zip: {json_path}")
                    
                    # Add to zip
                    zipf.writestr(json_path, json_data)
                    count += 1
        
        # Verify the zip was created correctly
        if not os.path.exists(zip_filename):
            logger.error(f"Zip file was not created: {zip_filename}")
            return jsonify({'status': 'error', 'message': 'Failed to create zip file'}), 500
            
        if os.path.getsize(zip_filename) == 0:
            logger.error(f"Created zip file is empty: {zip_filename}")
            return jsonify({'status': 'error', 'message': 'Created zip file is empty'}), 500
            
        logger.info(f"Successfully created zip file with {count} annotations: {zip_filename}")
        
        # Create the download URL for the client - append a cache-busting query parameter
        download_url = f"/api/download_file/{download_id}/annotations.zip?t={int(time.time())}"
        
        return jsonify({
            'status': 'success',
            'count': count,
            'download_url': download_url
        })
    
    except Exception as e:
        logger.exception(f"Error in download_annotations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error preparing annotations: {str(e)}'
        }), 500

@app.route('/api/download_masks', methods=['POST'])
def download_masks():
    """Create a downloadable zip file of masks with folder structure preserved"""
    data = request.json
    session_id = data.get('session_id')
    preserve_structure = data.get('preserve_structure', True)
    
    logger.info(f"Download masks request received: session_id={session_id}, preserve_structure={preserve_structure}")
    
    if session_id not in sessions:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    try:
        # Create a unique identifier for this download - add uuid for uniqueness
        download_id = f"masks_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        # Create a temporary directory to store the files
        temp_dir = os.path.join(app.config['TEMP_DOWNLOADS'], download_id)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Path to the zip file we'll create
        zip_filename = os.path.join(temp_dir, 'masks.zip')
        
        # Count of included masks
        count = 0
        
        # Create a mapping of original paths to help preserve structure
        path_mapping = {}
        for i, img_path in enumerate(session['images']):
            # For files uploaded as part of a folder, keep their structure
            if 'folder_uploads' in img_path and preserve_structure:
                relative_path = img_path.split('folder_uploads/')[-1]
                parent_dir = os.path.dirname(relative_path)
                filename = os.path.basename(img_path)
            else:
                # For individual uploads, just use the filename
                parent_dir = ""
                filename = os.path.basename(img_path)
            
            path_mapping[i] = {'dir': parent_dir, 'filename': filename}
        
        # Create the zip file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, mask in enumerate(session['mask_storage']):
                if mask is not None and np.any(mask):
                    # Get mapping info
                    mapping = path_mapping[i]
                    parent_dir = mapping['dir']
                    base_name = os.path.splitext(mapping['filename'])[0]
                    
                    # Create a colored visualization of the mask
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    
                    for obj_id in range(1, session['annotation_counter'][i]):
                        if obj_id-1 < len(session['color_palette']): 
                            color = (np.array(session['color_palette'][obj_id-1]) * 255).astype(np.uint8)
                        else:
                            color = np.random.randint(0, 255, 3, dtype=np.uint8)
                        
                        colored_mask[mask == obj_id] = color
                    
                    # Save to a temporary file
                    temp_mask_path = os.path.join(temp_dir, f"temp_mask_{i}.png")
                    Image.fromarray(colored_mask).save(temp_mask_path)
                    
                    # Add to zip with preserved structure if requested
                    mask_path = f"{parent_dir}/{base_name}_mask.png" if parent_dir else f"{base_name}_mask.png"
                    
                    # Read file and add to zip
                    with open(temp_mask_path, 'rb') as f:
                        zipf.writestr(mask_path, f.read())
                    
                    # Remove temporary file
                    os.remove(temp_mask_path)
                    count += 1
        
        # Create the download URL for the client - append a cache-busting query parameter
        download_url = f"/api/download_file/{download_id}/masks.zip?t={int(time.time())}"
        
        return jsonify({
            'status': 'success',
            'count': count,
            'download_url': download_url
        })
    
    except Exception as e:
        logger.exception(f"Error in download_masks: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error preparing masks: {str(e)}'
        }), 500

@app.route('/api/download_combined', methods=['POST'])
def download_combined():
    """Create a downloadable zip file containing both annotations and masks"""
    data = request.json
    session_id = data.get('session_id')
    preserve_structure = data.get('preserve_structure', True)
    
    logger.info(f"Combined download request received: session_id={session_id}, preserve_structure={preserve_structure}")
    
    if session_id not in sessions:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session'}), 400
    
    session = sessions[session_id]
    
    try:
        # Create a unique identifier for this download
        download_id = f"combined_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        # Create a temporary directory to store the files
        temp_dir = os.path.join(app.config['TEMP_DOWNLOADS'], download_id)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Path to the zip file we'll create
        zip_filename = os.path.join(temp_dir, 'gaime_export.zip')
        
        # Count of included items
        annotation_count = 0
        mask_count = 0
        
        # Create a mapping of original paths to help preserve structure
        path_mapping = {}
        for i, img_path in enumerate(session['images']):
            # For files uploaded as part of a folder, keep their structure
            if 'folder_uploads' in img_path and preserve_structure:
                parts = img_path.split('folder_uploads')
                if len(parts) > 1:
                    relative_path = parts[1].lstrip('/')
                    parent_dir = os.path.dirname(relative_path)
                else:
                    parent_dir = ""
                filename = os.path.basename(img_path)
            else:
                # For individual uploads, just use the filename
                parent_dir = ""
                filename = os.path.basename(img_path)
            
            path_mapping[i] = {'dir': parent_dir, 'filename': filename}
        
        # Create the zip file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add annotations to the zip
            for i in range(len(session['images'])):
                # Check if we have any annotations for this image
                has_annotations = False
                
                # Check for masks
                if i < len(session['mask_storage']) and session['mask_storage'][i] is not None:
                    has_annotations = has_annotations or np.any(session['mask_storage'][i])
                
                # Check for manual counts
                if i in session['manual_counts']:
                    has_annotations = has_annotations or any(session['manual_counts'][i].values())
                
                # Check for legends
                if i in session['legends']:
                    has_annotations = has_annotations or bool(session['legends'][i])
                    
                # Check for object metadata
                if i in session['object_metadata']:
                    has_annotations = has_annotations or bool(session['object_metadata'][i])
                
                if has_annotations:
                    # Get mapping info
                    mapping = path_mapping[i]
                    parent_dir = mapping['dir']
                    base_name = os.path.splitext(mapping['filename'])[0]
                    
                    # Create annotation data
                    annotation_data = {
                        "image": os.path.basename(session['images'][i]),
                        "game": session['game'],
                        "objects": [],
                        "manual_counts": session['manual_counts'].get(i, {}),
                        "legend": session['legends'].get(i, ""),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Add object metadata
                    if i in session['object_metadata']:
                        for obj_id, obj_data in session['object_metadata'][i].items():
                            annotation_data["objects"].append(obj_data)
                    
                    # Convert to JSON string with proper formatting
                    json_data = json.dumps(annotation_data, indent=2)
                    
                    # Add to zip with preserved structure if requested
                    json_path = f"annotations/{parent_dir}/{base_name}_annotation.json" if parent_dir else f"annotations/{base_name}_annotation.json"
                    
                    # Ensure directory structure exists in the zip
                    zipf.writestr(json_path, json_data)
                    annotation_count += 1
                    
                    # If we have a mask, add it to the zip as well
                    mask = session['mask_storage'][i]
                    if mask is not None and np.any(mask):
                        # Create a colored visualization of the mask
                        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        
                        for obj_id in range(1, session['annotation_counter'][i]):
                            if obj_id-1 < len(session['color_palette']): 
                                color = (np.array(session['color_palette'][obj_id-1]) * 255).astype(np.uint8)
                            else:
                                color = np.random.randint(0, 255, 3, dtype=np.uint8)
                            
                            colored_mask[mask == obj_id] = color
                        
                        # Save to a temporary file
                        temp_mask_path = os.path.join(temp_dir, f"temp_mask_{i}.png")
                        Image.fromarray(colored_mask).save(temp_mask_path)
                        
                        # Add to zip with preserved structure
                        mask_path = f"masks/{parent_dir}/{base_name}_mask.png" if parent_dir else f"masks/{base_name}_mask.png"
                        
                        # Read file and add to zip
                        with open(temp_mask_path, 'rb') as f:
                            zipf.writestr(mask_path, f.read())
                        
                        # Remove temporary file
                        os.remove(temp_mask_path)
                        mask_count += 1
        
        # Verify the zip was created correctly
        if not os.path.exists(zip_filename) or os.path.getsize(zip_filename) == 0:
            logger.error(f"Zip file not created or empty: {zip_filename}")
            return jsonify({'status': 'error', 'message': 'Failed to create export file'}), 500
            
        # Create the download URL for the client
        download_url = f"/api/download_file/{download_id}/gaime_export.zip?t={int(time.time())}"
        
        logger.info(f"Successfully created combined export with {annotation_count} annotations and {mask_count} masks")
        
        return jsonify({
            'status': 'success',
            'annotation_count': annotation_count,
            'mask_count': mask_count,
            'download_url': download_url
        })
    
    except Exception as e:
        logger.exception(f"Error in combined download: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error creating combined export: {str(e)}'
        }), 500

@app.route('/api/download_file/<dir_name>/<filename>')
def download_file(dir_name, filename):
    """Handle downloading the zip file"""
    # Validate that the requested directory name is safe - allow the UUID part
    if not re.match(r'^[a-zA-Z0-9_]+$', dir_name):
        logger.warning(f"Invalid directory name requested: {dir_name}")
        return jsonify({'status': 'error', 'message': 'Invalid directory name'}), 400
    
    file_path = os.path.join(app.config['TEMP_DOWNLOADS'], dir_name, filename)
    logger.info(f"Attempting to download file: {file_path}")
    
    # Verify the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    
    # Verify the file is within the downloads directory (security check)
    abs_temp_downloads = os.path.abspath(app.config['TEMP_DOWNLOADS'])
    abs_file_path = os.path.abspath(file_path)
    if not abs_file_path.startswith(abs_temp_downloads):
        logger.error(f"Security violation: attempted access to file outside downloads directory: {file_path}")
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403
    
    try:
        # Schedule cleanup independently to allow the file to be downloaded completely
        def delayed_remove():
            try:
                # Give the client plenty of time to download
                time.sleep(300)  # 5 minutes
                dir_path = os.path.dirname(file_path)
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path, ignore_errors=True)
                        logger.info(f"Removed temporary directory: {dir_path}")
                    except Exception as e:
                        logger.error(f"Error removing directory {dir_path}: {e}")
            except Exception as e:
                logger.error(f"Error in delayed cleanup: {e}")
        
        # Start cleanup thread
        from threading import Thread
        Thread(target=delayed_remove, daemon=True).start()
        logger.info(f"Scheduled cleanup for directory: {os.path.dirname(file_path)}")
        
        # Use explicit mimetype to ensure browser handles it as a download
        mimetype = 'application/zip'
        
        logger.info(f"Serving file: {file_path} with mimetype: {mimetype}")
        
        response = send_file(
            file_path,
            as_attachment=True, 
            download_name=filename,
            mimetype=mimetype
        )
        
        # Add Cache-Control header to prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        logger.exception(f"Error serving download file: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error downloading file: {str(e)}'
        }), 500

def after_this_request(f):
    """Decorator for actions to run after the request is processed"""
    if not hasattr(g, 'after_request_callbacks'):
        g.after_request_callbacks = []
    g.after_request_callbacks.append(f)
    return f

@app.after_request
def call_after_request_callbacks(response):
    """Execute after-request callbacks"""
    for callback in getattr(g, 'after_request_callbacks', []):
        response = callback(response)
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5020)
