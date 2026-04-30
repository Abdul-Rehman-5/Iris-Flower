Iris Flower Classification 🌸
A professional Machine Learning project that classifies Iris flowers into different species using flower measurements such as sepal length, sepal width, petal length, and petal width.
The project uses Logistic Regression for classification and includes:


Data preprocessing


Feature scaling


Label encoding


Model training & evaluation


Model saving using Joblib


Professional Streamlit web application



📌 Features


Iris flower species prediction


Real-time prediction using Streamlit


Clean and professional UI


Saved ML model using Joblib


High classification accuracy


Deployment-ready project structure



🧠 Machine Learning Workflow


Load dataset


Preprocess data


Encode labels


Scale features


Train Logistic Regression model


Evaluate performance


Save model artifacts


Deploy with Streamlit



📂 Project Structure
iris-project/│├── iris.csv├── train_model.py├── app.py├── model.pkl├── scaler.pkl├── encoder.pkl└── README.md

📊 Dataset Features
FeatureDescriptionsepal_lengthLength of sepalsepal_widthWidth of sepalpetal_lengthLength of petalpetal_widthWidth of petalclassFlower species

🤖 Model Used
Logistic Regression
Logistic Regression was selected because:


Fast and efficient


Excellent performance on Iris dataset


Easy to interpret


High accuracy for classification tasks



⚙️ Technologies Used


Python


Pandas


NumPy


Scikit-learn


Joblib


Streamlit



🚀 Installation
Clone Repository
git clone https://github.com/your-username/iris-flower-classification.gitcd iris-flower-classification

📦 Install Dependencies
pip install -r requirements.txt
Or manually:
pip install pandas numpy scikit-learn streamlit joblib

🧪 Train Model
python train_model.py
This will generate:


model.pkl


scaler.pkl


encoder.pkl



🌐 Run Streamlit App
python -m streamlit run app.py

📈 Model Evaluation
The model is evaluated using:


Accuracy Score


Classification Report


Typical accuracy achieved:
95% - 100%

💾 Model Saving
The trained model and preprocessing objects are saved using Joblib for:


Faster loading


No retraining


Easy deployment



🖥️ Streamlit Application
The web application allows users to:


Input flower measurements


Predict flower species instantly


View prediction probabilities


Interact with a clean modern UI



🔮 Future Improvements


Add confusion matrix visualization


Compare multiple ML models


Deploy online using Streamlit Cloud


Add interactive charts


Build mobile-friendly interface



📚 References


Scikit-learn Documentation


Streamlit Documentation


Iris Dataset


Python Official Documentation



👨‍💻 Author
Abdul Rehman
