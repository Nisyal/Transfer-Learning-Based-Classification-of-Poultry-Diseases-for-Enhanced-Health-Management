import pip
pip.main(['install', 'flask', 'tensorflow', 'numpy', 'Pillow'])
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model(r"C:\Users\NISYAL\OneDrive\Desktop\Project\best_poultry_model (2).h5")
from flask import Flask, send_from_directory

@app.route('/static/<path:path>')
def send_static(path):
    print(f"Serving static file: {path}")
    return send_from_directory('static', path)

# Define class names
class_names = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return f"Hence, The infection type detected as {predicted_class}"

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "No image selected!"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = predict_disease(filepath)
        return render_template('index1.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True,use_reloader=False)
