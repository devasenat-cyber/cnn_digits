# Simple Flask API — Use Your Trained CNN

This project serves the CNN model you already trained and saved. It does **not** train a new model.

## 1. Put your existing model files in the `model` folder

Copy the two files from your existing saved model folder:

```text
digit_classifier.keras
class_names.json
```

The final folder should look like this:

```text
Unit_V_Flask_Simple_API/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── digit_classifier.keras
│   └── class_names.json
│
└── samples/
    └── sample_digit_7.png
```

The spelling of the filenames matters. Windows may hide `.json` in File Explorer, so check that the real filename is `class_names.json`, not `class_names.json.json`.

If `class_names.json` is not available, the app can use output indexes `0`, `1`, `2`, ... as the labels. Your screenshot shows that file, so copy it too.

## 2. Install Python packages

Open PowerShell or Command Prompt inside this project folder.

```bash
py -3.11 -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Start the Flask API

```bash
python app.py
```

You should see:

```text
Loading the trained CNN...
CNN model loaded successfully!
Flask API is starting...
```

Open these URLs in your browser:

- Home API: http://127.0.0.1:5000/
- Health API: http://127.0.0.1:5000/health

## 4. Test the prediction API

Use Postman:

1. Select method `POST`.
2. Enter URL: `http://127.0.0.1:5000/predict`.
3. Select **Body → form-data**.
4. Enter key `file`.
5. Change its type from Text to **File**.
6. Choose a handwritten digit PNG/JPG.
7. Click **Send**.

The API returns JSON similar to:

```json
{
  "status": "success",
  "filename": "digit.png",
  "predicted_class": "7",
  "class_id": 7,
  "confidence": 0.9824,
  "top_3_predictions": [
    {"class": "7", "probability": 0.9824},
    {"class": "1", "probability": 0.0112},
    {"class": "9", "probability": 0.0031}
  ],
  "inference_time_ms": 8.32
}
```

The numbers above are only an example; actual predictions depend on your saved model and uploaded image.

## 5. Important: image format

The CNN was trained on MNIST images:
- 28 × 28 pixels
- grayscale
- black background with a white handwritten digit
- pixel values scaled between 0 and 1

The API converts uploaded images to 28 × 28 grayscale and scales pixel values. For best results, supply a digit image with the same black-background/white-digit appearance as MNIST. This simple teaching app does not automatically invert image colours.

## 6. What does the single Python file do?

Everything is inside `app.py`:

1. Imports Flask, TensorFlow, Pillow and NumPy.
2. Loads the saved `.keras` CNN once when the app starts.
3. Defines a helper to resize and preprocess the uploaded image.
4. Defines `GET /` and `GET /health`.
5. Defines `POST /predict`.
6. Returns the predicted digit, confidence, top-three classes and inference time.

There is no separate model service file, no database and no frontend framework.

## Troubleshooting

| Problem | What to check |
|---|---|
| Model not found | Put `digit_classifier.keras` inside the `model` folder. |
| Class names mismatch | Copy the matching `class_names.json` from the same training run. |
| TensorFlow installation error | Use Python 3.11 in a new virtual environment. |
| `No image uploaded` | Use `file` as the form-data key and select File, not Text. |
| Incorrect predictions | Match MNIST input appearance: black background, white digit, centered digit. |
| Port 5000 is busy | Stop the other Flask app or change the port in `app.py`. |

This is a beginner classroom example. Before production use, add authentication, HTTPS, rate limiting, stronger file validation, logging and deployment monitoring.
