🧠 Parkinson’s Disease Detection using Machine Learning

📌 Project Overview

This project aims to detect Parkinson’s Disease (PD) using voice-based biomedical features and machine learning.

The system analyzes vocal measurements and predicts whether a person is:

- "0" → Healthy
- "1" → Parkinson’s Disease

---

🚀 Key Components

- 📊 Exploratory Data Analysis (EDA) with heatmaps
- ⚙️ Data preprocessing (cleaning + scaling)
- 🧠 Ensemble model (SACK)
- 🎲 Monte Carlo simulation for robust evaluation

---

📂 Project Structure

project/
│
├── data/
│   └── parkinsons.csv
│
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── monte_carlo.py
│   └── sack.py
│
├── outputs/
│   └── plots/
│       ├── full_correlation.png
│       ├── target_correlation.png
│       ├── missing_values.png
│       ├── feature_intensity.png
│       └── high_correlation.png
│
├── main.py
├── requirements.txt
└── README.md

---

⚙️ How to Run the Project

1. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

2. Install Dependencies

pip install -r requirements.txt

3. Run the Project

python main.py

👉 After running, all plots will be saved in:

outputs/plots/

---

🧠 Model Description (SACK)

The project uses a simple ensemble model called SACK, which combines:

- 🌲 Random Forest Classifier
- 📈 Logistic Regression

How it works:

1. Both models make predictions independently
2. Final prediction is decided using majority voting

👉 This improves accuracy and stability.

---

🎲 Monte Carlo Simulation

Instead of using a single train-test split, the model is evaluated multiple times:

- Random split → Train → Test → Accuracy
- Repeated 30 times

Why this is important:

- Reduces bias
- Provides more reliable results
- Ensures stability across different data splits

---

📊 Exploratory Data Analysis (EDA)

All generated plots are stored in:

outputs/plots/

---

🔥 1. Full Correlation Heatmap

"Full Correlation" (outputs/plots/full_correlation.png)

Description:

Shows correlation between all features.

Interpretation:

- +1 → Strong positive relationship
- -1 → Strong negative relationship
- 0 → No relationship

👉 Helps identify redundant features and relationships.

---

🔥 2. Target Correlation Heatmap

"Target Correlation" (outputs/plots/target_correlation.png)

Description:

Shows correlation of each feature with Parkinson’s label.

Interpretation:

- Higher value → More important feature
- Lower value → Less impact

👉 Helps identify key features influencing the disease.

---

🔥 3. Missing Values Heatmap

"Missing Values" (outputs/plots/missing_values.png)

Description:

Displays missing values in dataset.

Interpretation:

- Light color → Data present
- Dark color → Missing data

👉 Dataset is mostly clean with minimal missing values.

---

🔥 4. Feature Intensity Heatmap

"Feature Intensity" (outputs/plots/feature_intensity.png)

Description:

Shows normalized feature values.

Interpretation:

- Bright → High value
- Dark → Low value

👉 Helps detect patterns and abnormal behavior in features.

---

🔥 5. High Correlation Heatmap

"High Correlation" (outputs/plots/high_correlation.png)

Description:

Shows only highly correlated features (> 0.8).

Interpretation:

- Identifies redundant features
- Helps in feature selection

---

📈 Model Performance

The model was evaluated using Monte Carlo simulation.

Results:

- Mean Accuracy: 0.8897 (~89%)
- Standard Deviation: 0.0547

Interpretation:

- The model correctly predicts Parkinson’s disease in approximately 89% of cases
- The low standard deviation (~5.5%) indicates stable and consistent performance
- The model generalizes well to unseen data

---

🎯 Key Achievements

- ✔ Built a Parkinson’s detection model using voice features
- ✔ Achieved ~89% accuracy
- ✔ Used ensemble learning for better predictions
- ✔ Applied Monte Carlo simulation for robust evaluation
- ✔ Generated clear visual insights using heatmaps

---

📌 Conclusion

This project demonstrates how machine learning can effectively detect Parkinson’s Disease using voice data with strong accuracy and reliability.

---

👨‍💻 Author

Your Name

---

⭐ Important Note

Before pushing to GitHub, ensure that all plots are generated:

outputs/plots/*.png