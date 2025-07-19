from flask import Flask, render_template, request, send_from_directory, url_for
import os
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECTION_FOLDER = 'static/detections'

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# โหลดโมเดล YOLO
model = YOLO(r"D:\test_web\best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ทำ Object Detection
    results = model(filepath, conf=0.05)  
    result_image = results[0].plot()  # แสดงผลลัพธ์เป็นภาพ

    output_path = os.path.join(DETECTION_FOLDER, file.filename)
    cv2.imwrite(output_path, result_image)

    # ใช้ url_for เพื่อสร้าง URL ที่ถูกต้อง
    image_url = url_for('get_detected_image', filename=file.filename)

    return render_template('result.html', image_url=image_url)

@app.route('/static/detections/<filename>')
def get_detected_image(filename):
    return send_from_directory(DETECTION_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
