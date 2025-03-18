# 📱 Mobile Phone Price Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Library-orange)
![RandomForest](https://img.shields.io/badge/Random%20Forest-Classifier-green)

## 📌 Project Overview
This project is a **Mobile Phone Price Classification** model built using **RandomForestClassifier**. The model predicts the price category of a mobile phone based on various features such as battery power, RAM, screen size, and more. GridSearchCV is used for hyperparameter tuning to optimize model performance.

## ✨ Features
✅ **Random Forest Classifier** for high accuracy.  
✅ **GridSearchCV** for hyperparameter tuning.  
✅ **Cross-validation** to improve generalization.  
✅ **Confusion matrix & ROC curve** for evaluation.  
✅ **Gradio Interface** for easy model inference.  
✅ **Pickle model storage** for deployment.  

## 🗂 Repository Structure
```
Mobile-Price-Classification/
│-- dataset.csv        # Mobile phone dataset
│-- model_training.py  # Model training and tuning script
│-- mobile.pkl         # Saved trained model
│-- app.py             # Gradio interface for inference
│-- requirements.txt   # Dependencies
│-- README.md          # Project documentation

```

## 📝 Dataset
The dataset contains mobile phone specifications such as:
- Battery Power
- RAM
- Internal Memory
- Screen Size
- Processor Speed
- Camera Quality
- Price Range (Target Variable: 0-3 categories)

## 🔄 Installation
To run this project, clone the repository and install dependencies:
```sh
git clone <https://github.com/VedantRD007/Mobile-Pricing-Prediction.git>
cd Mobile-Price-Classification
pip install -r requirements.txt
```

## 💪 Model Training
To train the model and tune hyperparameters, run:
```sh
python mobile_pricing.py
```

## 🎨 Running the Gradio App
To launch the web interface for easy inference:
```sh
python mobile_gr.py
```

## 💻 Model Evaluation
The trained model is evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **ROC Curve**
- **GridSearchCV Best Parameters**

## 🌐 Deployment
The trained model is saved as `mobile.pkl` and deployed using **Gradio**, allowing users to input mobile specifications and get real-time price predictions through a web-based UI.

## 🛠️ Dependencies
Ensure you have the following installed:
```
numpy
pandas
scikit-learn
pickle
matplotlib
seaborn
gradio
```


## ✨ Contributors
- **Your Name** - Developer & ML Engineer

Feel free to submit PRs and suggestions to improve this project! 🚀
