from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from src.predict import *
from PIL import Image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded_images/'
app.config['STATIC_FOLDER'] = 'static/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def main():
    caption = ""
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            # Save original file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Save to static folder for display
            static_image_path = os.path.join(app.config['STATIC_FOLDER'], 'uploaded_image.jpg')
            
            # Convert and resize image if needed
            with Image.open(file_path) as img:
                # Resize if too large, maintain aspect ratio
                img.thumbnail((800, 800))
                img.convert('RGB').save(static_image_path, 'JPEG')
            
            # Generate caption
            model, device, dataset, transform = setup_model("artifacts/final_model.pth")
            caption = generate_caption(file_path, model, device, dataset, transform)

    return render_template("index.html", caption=caption)

PORT = 8000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)