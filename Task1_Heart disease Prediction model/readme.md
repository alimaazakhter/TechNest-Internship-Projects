# Heart Disease Prediction

This project is a web-based application that predicts the risk of heart disease using machine learning. It provides an interactive dashboard, visualizations, and health recommendations based on user input.

## Features

- **Heart Disease Risk Prediction:** Enter patient data to get an instant prediction using a trained Random Forest model.
- **Interactive Dashboard:** Visualizes key health metrics (vital signs, risk distribution, cardiovascular profile, heart rate trend).
- **Health Summary:** Provides insights, risk level, and recommendations.
- **BMI Calculator:** Calculate and interpret Body Mass Index.
- **PDF Report:** Download a summary report of your prediction and recommendations.

## Project Structure

```
app.py
heart_disease.ipynb
heart_model.pkl
heart.csv
Readme.md
.vscode/
static/
    script.js
    style.css
templates/
    index.html
```

- **app.py:** Flask backend serving the web app and handling predictions.
- **heart_disease.ipynb:** Jupyter notebook for data analysis, preprocessing, and model training.
- **heart_model.pkl:** Saved machine learning model (Random Forest).
- **heart.csv:** Dataset containing patient records.
- **static/**: Frontend assets (JavaScript, CSS).
- **templates/**: HTML templates for the web interface.

## How It Works

1. **Data Collection:** Uses [heart.csv](heart.csv) with features like Age, Sex, Chest Pain Type, Resting BP, Cholesterol, etc.
2. **Preprocessing:** Categorical features are label-encoded. Data is cleaned and split into train/test sets.
3. **Model Training:** A Random Forest Classifier is trained in [heart_disease.ipynb](heart_disease.ipynb) and saved as [heart_model.pkl](heart_model.pkl).
4. **Prediction:** User inputs are collected via a web form ([templates/index.html](templates/index.html)), sent to the Flask backend ([app.py](app.py)), and the model predicts the risk.
5. **Visualization:** Results and health metrics are displayed using interactive charts ([static/script.js](static/script.js)).

## How to Run

1. **Install dependencies:**
    ```sh
    pip install flask numpy pandas scikit-learn matplotlib seaborn
    ```

2. **Train the model (optional):**
    - Open [heart_disease.ipynb](heart_disease.ipynb) and run all cells to retrain and save the model.

3. **Start the web app:**
    ```sh
    python app.py
    ```
    - Open your browser and go to `http://localhost:5000`

## Usage

- Fill in the form with patient data.
- Click "Analyze Risk" to get the prediction.
- View the dashboard for detailed metrics and recommendations.
- Use the BMI calculator for additional health assessment.
- Download a PDF report if needed.

## Dataset

- The dataset ([heart.csv](heart.csv)) contains 918 records with the following columns:
    - Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease

## Model Performance

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~89%   |
| Precision  | 0.88   |
| Recall     | 0.87   |
| F1-Score   | 0.87   |

- **Accuracy:** ~89%
- **Algorithm:** Random Forest Classifier
- **Evaluation:** Precision, recall, F1-score, and confusion matrix are available in [heart_disease.ipynb](heart_disease.ipynb).

## Disclaimer

This tool is for educational and informational purposes only. It does **not** replace professional medical advice. Always consult a healthcare provider for diagnosis and treatment.

## License

MIT License

---

**Contact:** support@heartprediction.com
