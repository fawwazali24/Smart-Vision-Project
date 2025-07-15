import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import sys
import io
from ultralytics import YOLO
import easyocr

# Set the default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)

# Load your pre-trained vegetable classification model
vegetables = ['annona', 'apple', 'banana', 'beet', 'bell_pepper', 'cabbage', 'carrot', 'cucumber', 'egg', 'eggplant', 'garlic', 'guava', 'onion', 'orange', 'pear', 'pineapple', 'pitaya', 'potato', 'tomato', 'zucchini']
model = YOLO('fresh_model.pt')
pack_names = ['Complan Classic Creme', 'Complan Kesar Badam', 'Complan Nutrigro Badam Kheer', 'Complan Pista Badam', 'Complan Royal Chocolate', 'Complan Royale Chocolate', 'EY AAAM TULSI TURMERIC FACEWASH50G', 'EY ADVANCED GOLDEN GLOW PEEL OFF M- 50G', 'EY ADVANCED GOLDEN GLOW PEEL OFF M- 90G', 'EY EXF WALNUT SCRUB AYR 200G', 'EY HALDICHANDAN FP HF POWDER 25G', 'EY HYD-EXF WALNT APR SCRUB AYR100G', 'EY HYDR - EXF WALNUT APRICOT SCRUB 50G', 'EY NAT GLOW ORANGE PEEL OFF AY 90G', 'EY NATURALS NEEM FACE WASH AY 50G', 'EY RJ CUCUMBER ALOEVERA FACEPAK50G', 'EY TAN CHOCO CHERRY PACK 50G', 'EY_SCR_PURIFYING_EXFOLTNG_NEEM_PAPAYA_50G', 'Everyuth Naturals Body Lotion Nourishing Cocoa 200ml', 'Everyuth Naturals Body Lotion Rejuvenating Flora 200ml', 'Everyuth Naturals Body Lotion Soothing Citrus 200ml', 'Everyuth Naturals Body Lotion Sun Care Berries SPF 15 200ml', 'Glucon D Nimbu Pani', 'Glucon D Nimbu Pani 1-KG', 'Glucon D Regular', 'Glucon D Regular 1-KG', 'Glucon D Regular 2-KG', 'Glucon D Tangy orange', 'Glucon D Tangy orange 1-KG', 'Nutralite ACHARI MAYO 300g-275g-25g-', 'Nutralite ACHARI MAYO 30g', 'Nutralite CHEESY GARLIC MAYO 300g-275g-25g-', 'Nutralite CHEESY GARLIC MAYO 30g', 'Nutralite CHOCO SPREAD CALCIUM 275g', 'Nutralite DOODHSHAKTHI PURE GHEE 1L', 'Nutralite TANDOORI MAYO 300g-275g-25g-', 'Nutralite TANDOORI MAYO 30g', 'Nutralite VEG MAYO 300g-275g-25g-', 'Nycil Prickly Heat Powder', 'SUGAR FREE GOLD 500 PELLET', 'SUGAR FREE GOLD POWDER 100GM', 'SUGAR FREE GOLD SACHET 50', 'SUGAR FREE GOLD SACHET 50 SUGAR FREE GOLD SACHET 50', 'SUGAR FREE GRN 300 PELLET', 'SUGAR FREE NATURA 500 PELLET', 'SUGAR FREE NATURA DIET SUGAR', 'SUGAR FREE NATURA DIET SUGAR 80GM', 'SUGAR FREE NATURA SACHET 50', 'SUGAR FREE NATURA SWEET DROPS', 'SUGAR FREE NATURA_ POWDER_CONC_100G', 'SUGAR FREE_GRN_ POWDER_CONC_100G', 'SUGARLITE POUCH 500G']

yolo_model = YOLO('another_model.pt')  # Adjust the path as needed
CONFIDENCE_THRESHOLD = 0.35

# Initialize EasyOCR reader (for English language)
reader = easyocr.Reader(['en'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(image)
        detections = results[0].boxes
        detected_vegetables = []
        for box in detections:
            if box.conf > CONFIDENCE_THRESHOLD:
                class_index = int(box.cls[0].item())
                vegetable_name = model.names[class_index]
                detected_vegetables.append(vegetable_name)

        return jsonify({'predictions': detected_vegetables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = yolo_model(image)
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        unique_classes = np.unique(class_indices)
        detected_classes = [pack_names[cls] for cls in unique_classes if cls < len(pack_names)]

        return jsonify({'predictions': detected_classes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb_image)
        extracted_text = ' '.join([res[1] for res in results])
        text_present = bool(extracted_text.strip())

        return jsonify({'text_present': text_present, 'extracted_text': extracted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)