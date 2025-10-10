# 🫀 Heart Attack Predictor

A **Machine Learning–powered web application** designed to predict the likelihood of a heart attack based on patient health data.  
The project uses **Flask** for the web interface, **Docker** for containerization, and **Miniforge + Mamba** for environment management.

---

## 📁 Project Structure
``` bash
Heart-Attack-Predictor/
├── app/
│ ├── static/ # CSS, JS, and image assets
│ ├── templates/ # HTML templates (Flask)
│ └── app.py # Flask web application entry point
│
├── data/
│ └── raw/
│ └── heart_attack_prediction.csv # Raw dataset
│
├── models/ # Trained ML models (e.g., .pkl files)
│
├── notebooks/ # Jupyter notebooks for individual members
│ ├── FC212006_Akila/
│ ├── FC212013_Maleesha/
│ ├── FC212031_Janudi/
│ ├── FC212041_Isivara/
│ ├── FC212042_Sudesh/
│ └── FC212045_Kavindu/
│
├── src/ # Core Python source files for data processing and ML pipeline
│
├── tests/ # Unit and integration tests
│
├── .devcontainer.json # VS Code Dev Container configuration
├── .gitignore
├── Makefile # Common commands (build, run, test, etc.)
├── Miniforge3-Linux-x86_64.sh # Miniforge installer script
├── pyproject.toml # Project dependencies and build configuration
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
---

## 🚀 Features

- **Machine Learning Prediction:** Predicts heart attack risk based on health metrics.  
- **Flask Web Interface:** Simple and interactive front-end for user input and prediction display.  
- **Dockerized Environment:** Fully containerised for consistent deployment.  
- **Miniforge + Mamba:** Fast and lightweight environment management.  
- **Collaborative Structure:** Includes contributions from multiple team members via Jupyter notebooks.  

---

## 📊 Dataset

### 📁 Source
The dataset used in this project is sourced from Kaggle:  
👉 [Heart Attack Prediction in Indonesia](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia)

### 🧠 About the Dataset
This dataset provides a **comprehensive health profile** of individuals in **Indonesia**, designed for predicting **heart attack risks**.  
It includes demographic, clinical, lifestyle, and environmental variables that capture real-world cardiovascular risk factors.  

With cardiovascular diseases rising in Indonesia, this dataset supports:
- Early prediction and prevention research  
- Machine learning model training and evaluation  
- Public health and epidemiological studies  

### 🔍 Variable Definitions

#### **Demographics**
| Variable | Type | Description |
|-----------|------|-------------|
| `age` | int | Age of the individual (25–90 years) |
| `gender` | str | Gender (Male, Female) |
| `region` | str | Living area (Urban, Rural) |
| `income_level` | str | Socioeconomic status (Low, Middle, High) |

#### **Clinical Risk Factors**
| Variable | Type | Description |
|-----------|------|-------------|
| `hypertension` | int | High blood pressure (1 = Yes, 0 = No) |
| `diabetes` | int | Diagnosed diabetes (1 = Yes, 0 = No) |
| `cholesterol_level` | int | Total cholesterol (mg/dL) |
| `obesity` | int | BMI > 30 (1 = Yes, 0 = No) |
| `waist_circumference` | int | Waist circumference (cm) |
| `family_history` | int | Family history of heart disease (1 = Yes, 0 = No) |

#### **Lifestyle & Behavioral Factors**
| Variable | Type | Description |
|-----------|------|-------------|
| `smoking_status` | str | Smoking habit (Never, Past, Current) |
| `alcohol_consumption` | str | Alcohol intake (None, Moderate, High) |
| `physical_activity` | str | Physical activity level (Low, Moderate, High) |
| `dietary_habits` | str | Diet quality (Healthy, Unhealthy) |

#### **Environmental & Social Factors**
| Variable | Type | Description |
|-----------|------|-------------|
| `air_pollution_exposure` | str | Pollution exposure (Low, Moderate, High) |
| `stress_level` | str | Stress level (Low, Moderate, High) |
| `sleep_hours` | float | Average sleep per night (3–9 hours) |

#### **Medical Screening & Health System Factors**
| Variable | Type | Description |
|-----------|------|-------------|
| `blood_pressure_systolic` | int | Systolic blood pressure (mmHg) |
| `blood_pressure_diastolic` | int | Diastolic blood pressure (mmHg) |
| `fasting_blood_sugar` | int | Blood sugar level (mg/dL) |
| `cholesterol_hdl` | int | HDL cholesterol (mg/dL) |
| `cholesterol_ldl` | int | LDL cholesterol (mg/dL) |
| `triglycerides` | int | Triglyceride level (mg/dL) |
| `EKG_results` | str | Electrocardiogram result (Normal, Abnormal) |
| `previous_heart_disease` | int | Prior heart disease (1 = Yes, 0 = No) |
| `medication_usage` | int | Currently taking heart-related medication (1 = Yes, 0 = No) |
| `participated_in_free_screening` | int | Participated in Indonesia’s free health screening program (1 = Yes, 0 = No) |

#### **🎯 Target Variable**
| Variable | Type | Description |
|-----------|------|-------------|
| `heart_attack` | int | Heart attack occurrence (1 = Yes, 0 = No) |

---

### 📂 File Location
The dataset is stored in:
```bash
data/raw/heart_attack_prediction.csv
```

## 🧠 Model Information

This project explores multiple **machine learning algorithms** to identify the most accurate and reliable model for predicting heart attack risk.

### 🧪 Trained Models
The following models were trained, tuned, and evaluated:

- **Logistic Regression**  
- **Random Forest Classifier**  
- **Support Vector Machine (SVM)**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree Classifier**  
- **XGBoost Classifier**

Each model was trained using the processed dataset and evaluated based on performance metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.  
The model with the **highest evaluation score** was selected as the **final model** for deployment in the Flask web application.

### 🏆 Model Selection Process
1. Data preprocessing and feature scaling were applied to ensure consistency.  
2. Each algorithm was trained using the same training and testing splits.  
3. Evaluation metrics were compared to identify the best-performing model.  
4. The final selected model was serialized and saved as a `.pkl` file in the `models/` directory.

## 🧩 Prerequisites

Ensure you have the following installed before setup:

| Tool | Description | Installation Link |
|------|--------------|------------------|
| [Docker](https://www.docker.com/) | For containerization | [Get Docker](https://docs.docker.com/get-docker/) |
| [Visual Studio Code](https://code.visualstudio.com/) | IDE for development | [Download VS Code](https://code.visualstudio.com/) |
| VS Code Extensions | - `ms-azuretools.vscode-docker`<br>- `ms-vscode-remote.remote-containers` | Install from VS Code Marketplace |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Heart-Attack-Predictor.git
cd Heart-Attack-Predictor
```
### 2. Open the Project in VS Code
Launch Visual Studio Code and open the project folder you just cloned.
```bash
code .
```
### 3. Open in Dev Container
If you have the following VS Code extensions installed:
- `ms-azuretools.vscode-docker`
- `ms-vscode-remote.remote-containers`
VS Code will automatically detect the `.devcontainer.json` file and prompt you to “Reopen in Container.”  
If not prompted, you can manually reopen in container:
```bash
Ctrl + Shift + P
```
Then type and select:
```bash
Dev Containers: Reopen in Container
```
This will:
- Build the Docker container defined in `.devcontainer.json`
- Install Miniforge and Mamba
- Automatically create and activate the Python environment
- Install all dependencies from `requirements.txt`
### 4. Run the Flask Application
Once the container is running and dependencies are installed, start the Flask app with:
```bash
cd app
python app.py
```
Now, open your browser and visit:  
👉 http://localhost:5000  
The web interface should load, allowing you to input health parameters and get heart attack predictions. 

---

## 🧑‍💻 Contributors

| Name          | Student ID |
| ------------- | ---------- |
| Akila         | FC212006   |
| Sudesh        | FC212042   |
| Kavindu       | FC212045   |
| Janudi        | FC212031   |
| Isivara       | FC212041   |
| Maleesha      | FC212013   |

---

## 🧾 License

This project is open-source. Add your license here (e.g., MIT).

---
