Molecular-solubility
This project aims to predict the solubility of molecules using the Delaney Solubility Dataset. The dataset consists of molecular structures and their corresponding solubility values, which can be analyzed using machine learning models.

Dataset
The project uses the Delaney Solubility Dataset, which provides molecular solubility values along with molecular descriptors. These descriptors help in training models to predict solubility.

Features
Data preprocessing and feature engineering from SMILES representations.
Implementation of random forest regressor for solubility prediction.
Model evaluation and performance comparison.
Data augmentation techniques to enhance training.
Overfitting analysis for model robustness.
Deployment of a simple application for predictions.
Requirements
To run this project, install the required dependencies:

pip install -r requirements.txt
Usage
Clone the repository
git clone https://github.com/yourusername/molecular-solubility.git
cd molecular-solubility
Run the data augmentation script
python aug.py
Train the model while addressing overfitting
python overfitting.py
Run the application for making predictions
python app.py
