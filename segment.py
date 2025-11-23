import os
import sys
from flask import Flask, render_template, send_file, request, jsonify
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

IMAGE_FOLDER = None
CURRENT_INDEX = 0
IMAGES = []
OUTPUT_FOLDER = None
SAM_MODEL = None
SAM_PREDICTOR = None

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500

@app.route('/test')
def test():
    return "Flask is working!", 200

@app.route('/static/hero.png')
def serve_hero():
    hero_path = Path(__file__).parent / 'public' / 'hero.png'
    if hero_path.exists():
        return send_file(hero_path)
    return jsonify({'error': 'Hero image not found'}), 404

@app.route('/api/current-image')
def get_current_image():
    global CURRENT_INDEX
    if not IMAGES or CURRENT_INDEX >= len(IMAGES):
        return jsonify({'error': 'No images'}), 404
    return jsonify({
        'image': f'/api/image/{CURRENT_INDEX}',
        'index': CURRENT_INDEX,
        'total': len(IMAGES)
    })

@app.route('/api/image/<int:idx>')
def serve_image(idx):
    if idx < 0 or idx >= len(IMAGES):
        return jsonify({'error': 'Invalid index'}), 404
    return send_file(IMAGES[idx])

@app.route('/api/next', methods=['POST'])
def next_image():
    global CURRENT_INDEX
    if CURRENT_INDEX < len(IMAGES) - 1:
        CURRENT_INDEX += 1
    return jsonify({'index': CURRENT_INDEX})

@app.route('/api/prev', methods=['POST'])
def prev_image():
    global CURRENT_INDEX
    if CURRENT_INDEX > 0:
        CURRENT_INDEX -= 1
    return jsonify({'index': CURRENT_INDEX})

@app.route('/api/segment', methods=['POST'])
def segment():
    global SAM_PREDICTOR
    
    data = request.json
    annotations = data.get('annotations', [])
    image_idx = data.get('imageIndex', CURRENT_INDEX)
    
    if image_idx < 0 or image_idx >= len(IMAGES):
        return jsonify({'error': 'Invalid image index'}), 400
    
    if not annotations:
        return jsonify({'error': 'No annotations provided'}), 400
    
    try:
        image_path = IMAGES[image_idx]
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if SAM_PREDICTOR is None:
            return jsonify({'error': 'SAM model not loaded'}), 500
        
        SAM_PREDICTOR.set_image(image_rgb)
        
        canvas_width = data.get('canvasWidth', image_rgb.shape[1])
        canvas_height = data.get('canvasHeight', image_rgb.shape[0])
        
        scale_x = image_rgb.shape[1] / float(canvas_width)
        scale_y = image_rgb.shape[0] / float(canvas_height)
        
        pts = []
        labels = []
        pos_pts = []
        
        for ann in annotations:
            x = ann['x'] * scale_x
            y = ann['y'] * scale_y
            label = int(ann.get('label', 1))
            pts.append([x, y])
            labels.append(label)
            if label == 1:
                pos_pts.append([x, y])
        
        if not pts:
            return jsonify({'error': 'No valid points'}), 400
        
        pts = np.array(pts, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        if len(pos_pts) == 0:
            pos_pts = pts
        pos_pts = np.array(pos_pts, dtype=np.float32)
        center_x, center_y = pos_pts.mean(axis=0)
        
        max_points = 20
        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = pts[idx]
            labels = labels[idx]
        
        input_points = pts
        input_labels = labels
        
        masks, scores, logits = SAM_PREDICTOR.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        h, w = image_rgb.shape[:2]
        cy = int(np.clip(center_y, 0, h - 1))
        cx = int(np.clip(center_x, 0, w - 1))
        
        best_mask = None
        best_score = -1e9
        
        for m, s in zip(masks, scores):
            if not m[cy, cx]:
                continue
            
            area = float(m.sum())
            score = float(s) - 1e-6 * area
            
            if score > best_score:
                best_score = score
                best_mask = m
        
        if best_mask is None:
            best_mask = masks[np.argmax(scores)]
        
        mask = best_mask.astype(np.uint8)
        
        num_labels, cc = cv2.connectedComponents(mask)
        center_label = cc[cy, cx]
        refined_mask = (cc == center_label).astype(bool)
        
        segmented_image = image_rgb.copy()
        segmented_image[~refined_mask] = [0, 0, 0]
        alpha_channel = (refined_mask * 255).astype(np.uint8)
        segmented_image = np.dstack([segmented_image, alpha_channel])
        
        output_path = OUTPUT_FOLDER / f"segmented_{image_idx:04d}.png"
        segmented_pil = Image.fromarray(segmented_image, 'RGBA')
        segmented_pil.save(output_path)
        
        return jsonify({
            'success': True,
            'segmented': f'/api/segmented/{image_idx}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/segmented/<int:idx>')
def serve_segmented(idx):
    segmented_path = OUTPUT_FOLDER / f"segmented_{idx:04d}.png"
    if segmented_path.exists():
        return send_file(segmented_path)
    return jsonify({'error': 'Segmented image not found'}), 404

@app.route('/api/save', methods=['POST'])
def save():
    data = request.json
    image_idx = data.get('imageIndex', CURRENT_INDEX)
    
    if image_idx < 0 or image_idx >= len(IMAGES):
        return jsonify({'error': 'Invalid image index'}), 400
    
    try:
        segmented_path = OUTPUT_FOLDER / f"segmented_{image_idx:04d}.png"
        if not segmented_path.exists():
            return jsonify({'error': 'No segmentation to save'}), 400
        
        original_path = IMAGES[image_idx]
        segmented_image = Image.open(segmented_path)
        segmented_image.save(original_path)
        
        return jsonify({'success': True, 'path': str(original_path)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_sam_model():
    global SAM_MODEL, SAM_PREDICTOR
    
    print("Loading SAM model...")
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    model_configs = [
        ("vit_h", "sam_vit_h_4b8939.pth"),
        ("vit_l", "sam_vit_l_0b3195.pth"),
        ("vit_b", "sam_vit_b_01ec64.pth"),
    ]
    
    for model_type, sam_checkpoint in model_configs:
        if os.path.exists(sam_checkpoint):
            try:
                print(f"Loading {model_type} model from {sam_checkpoint}...")
                SAM_MODEL = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                SAM_MODEL.to(device=device)
                SAM_PREDICTOR = SamPredictor(SAM_MODEL)
                print("SAM model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading {model_type}: {e}")
                continue
    
    print("Warning: No SAM checkpoint found.")
    print("Please download a checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
    print("Place it in the current directory. Recommended: sam_vit_b_01ec64.pth (smallest, fastest)")
    return False

def main():
    global IMAGE_FOLDER, IMAGES, CURRENT_INDEX, OUTPUT_FOLDER
    
    if len(sys.argv) < 2:
        print("Usage: python segment.py <image_folder>")
        sys.exit(1)
    
    IMAGE_FOLDER = Path(sys.argv[1])
    if not IMAGE_FOLDER.exists():
        print(f"Error: Folder {IMAGE_FOLDER} does not exist")
        sys.exit(1)
    
    IMAGES = sorted([
        f for f in IMAGE_FOLDER.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    ])
    
    if not IMAGES:
        print(f"Error: No images found in {IMAGE_FOLDER}")
        sys.exit(1)
    
    OUTPUT_FOLDER = IMAGE_FOLDER / "output"
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    
    print(f"Found {len(IMAGES)} images")
    
    if not load_sam_model():
        print("Warning: Continuing without SAM model. Segmentation will not work.")
    
    print(f"Starting server at http://127.0.0.1:5000")
    print(f"Also accessible at http://localhost:5000")
    print(f"Test endpoint: http://127.0.0.1:5000/test")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()