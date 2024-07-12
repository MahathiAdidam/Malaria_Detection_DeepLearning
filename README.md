# Malaria-Detection
Dataset
The dataset used for this project is sourced from Kaggle. It consists of two classes:

Parasitized: Cells infected with malaria parasites.
Uninfected: Cells that are not infected with malaria parasites.
The dataset is structured into training and testing sets, each containing subfolders for Parasitized and Uninfected cells.

Files and Directory Structure
data_preparation.py: Python script to preprocess and split the dataset into training and testing sets (Dataset/Train and Dataset/Test).
model_training.py: Python script to define, train, and evaluate the Convolutional Neural Network (CNN) model for malaria detection.
streamlitapp.py: Streamlit application script for interactive malaria cell classification and visualization of model predictions.
requirements.txt: List of Python dependencies required to run the project.

Getting Started

Prerequisites
Python 3.x
Install required packages using:
bash
Copy code
pip install -r requirements.txt
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/malaria-detection.git
cd malaria-detection
Set up the dataset:

Download the dataset from Kaggle.
Place the dataset in the Dataset directory with subfolders Train/Parasitized, Train/Uninfected, Test/Parasitized, and Test/Uninfected.
Run data preparation and model training:

bash
Copy code
python data_preparation.py
python model_training.py
Launch the Streamlit app:

bash
Copy code
streamlit run streamlitapp.py
Model Performance
Accuracy: 94.5%
Precision: 95.2%
Recall: 93.8%
F1-Score: 94.5%

Contributing
Contributions to improve the project are welcome. Please fork the repository and submit pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Dataset source: Kaggle
link: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
Additional info: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
Inspiration and guidance from various deep learning tutorials and resources.
