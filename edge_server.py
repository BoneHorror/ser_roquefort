import torch
import torch.fft
import kornia
import cv2
import numpy as np
import pyscreenshot
from kornia.filters import canny
from torch.nn import functional as F
from flask import Flask, request, jsonify
import time
from config import url_base, url_port

app = Flask(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print(f"[SERVER] Initializing ROCm on: {DEVICE}")
else:
    print("[SERVER] Warning! CPU fallback!")

# --- ROCm HELPER FUNCTIONS ---
def process_edges(tensor_img, low=0.1, high=0.2):
    # Kornia expects (B, C, H, W)
    # color.bgr_to_grayscale expects BGR (OpenCV standard)
    gray = kornia.color.bgr_to_grayscale(tensor_img)
    _, edges = canny(gray, low, high)
    return edges

def find_best_match_location(heatmap):
    B, C, H, W = heatmap.shape
    flat_tensor = heatmap.reshape(B, C, -1)
    max_val, flat_idx = torch.max(flat_tensor, dim=2)
    max_y = torch.div(flat_idx, W, rounding_mode='floor')
    max_x = flat_idx % W
    return max_val, (max_x, max_y)

def match_template_fft(image, template):
    H, W = image.shape[-2:]
    h, w = template.shape[-2:]
    
    pad_h = H - h
    pad_w = W - w
    if pad_h < 0 or pad_w < 0:
        return torch.zeros((1, 1, 1, 1)).to(image.device)

    tmpl_padded = F.pad(template, (0, pad_w, 0, pad_h))
    
    img_f = torch.fft.rfft2(image)
    tmpl_f = torch.fft.rfft2(tmpl_padded)
    
    res_f = img_f * torch.conj(tmpl_f)
    res = torch.fft.irfft2(res_f, s=(H, W))
    
    return res[..., :H-h+1, :W-w+1]

def grab_screen():
    #Not done via mss because X11 is a massive pain on Ubuntu 25+
    try:
        grab = pyscreenshot.grab()
        assert grab is not None, "Screenshot failed"
        
        img = np.array(grab) # Returns RGB (PIL)
        
        # Remove Alpha channel (RGBA -> RGB)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
            
        # RGB -> BGR CONVERSION
        # Grab gives RGB, OpenCV and our Kornia model use BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    except Exception as e:
        print(f"[SERVER] Screenshot error: {e}")
        return None

# --- WARM-UP ---
def warmup():
    print("[SERVER] Compiling ROCm kernels...")
    dummy_screen = torch.rand((1, 3, 1080, 1920)).to(DEVICE) #Hardcoded resolution TODO
    dummy_tmpl = torch.rand((1, 3, 100, 100)).to(DEVICE)
    e1 = process_edges(dummy_screen)
    e2 = process_edges(dummy_tmpl)
    _ = match_template_fft(e1, e2)
    torch.cuda.synchronize()
    print("[SERVER] Warmed up!")

warmup()

@app.route('/match', methods=['POST'])
def match():
    t0 = time.time()
    try:
        data = request.json
        template_path = data.get('path')
        thresh = data.get('thresh', 0.6)
        save_debug = data.get('debug', False) # Save debug image?
        
        if not template_path: return jsonify({"error": "No path"}), 400
        
        # 1. Template
        img_bgr = cv2.imread(template_path)
        if img_bgr is None:
            return jsonify({"found": False, "error": "Cannot open template"}), 404
        tmpl_t = kornia.image_to_tensor(img_bgr, keepdim=False).float() / 255.0
        tmpl_t = tmpl_t.to(DEVICE)
        
        # 2. Screen
        screen_bgr = grab_screen()
        if screen_bgr is None:
            return jsonify({"found": False, "error": "Screen grab failed"}), 500
        screen_t = kornia.image_to_tensor(screen_bgr, keepdim=False).float() / 255.0
        screen_t = screen_t.to(DEVICE)

        # 3. Calculations
        with torch.inference_mode():
            screen_edges = process_edges(screen_t)
            tmpl_edges = process_edges(tmpl_t)
            tmpl_sum = torch.sum(tmpl_edges)
            
            if tmpl_sum == 0:
                 return jsonify({"found": False, "score": 0.0, "reason": "No edges"})

            res = match_template_fft(screen_edges, tmpl_edges)
            max_val, (max_x, max_y) = find_best_match_location(res)
            normalized_score = (max_val / (tmpl_sum + 1e-6)).item()

        found = normalized_score > thresh
        
        # --- DEBUG VISUALIZATION ---
        if save_debug:
            # Convert edges from GPU back to CPU image for drawing
            debug_img = (screen_edges.squeeze().cpu().numpy() * 255).astype(np.uint8)
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
            
            x, y = int(max_x.item()), int(max_y.item())
            h, w = tmpl_edges.shape[2], tmpl_edges.shape[3]
            
            # Color: Green (found) or Red (fail)
            color = (0, 255, 0) if found else (0, 0, 255)
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"{normalized_score:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imwrite("server_last_debug.png", debug_img)
        # ---------------------------

        duration = time.time() - t0
        return jsonify({
            "found": found,
            "score": normalized_score,
            "x": int(max_x.item()),
            "y": int(max_y.item()),
            "duration": duration
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"found": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("[SERVER] Started!")
    app.run(host=url_base, port=url_port, debug=False, threaded=False)