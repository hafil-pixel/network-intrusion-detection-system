Network Intrusion Detection System using Machine Learning

This project is about detecting suspicious network activity using Machine Learning.
I’ve used a Decision Tree Classifier to train the model on a sample dataset of network traffic.

The main goal is simple:
•Feed the model some data about network activity
•Let it learn what “normal” and “malicious” traffic looks like
•Predict whether new traffic is safe or suspicious



∆ Files in this repo

network_intrusion_detection.py → The main Python script for training and testing the model

networkintrusion1.csv → The dataset used for training

README.md → This file you’re reading now



∆ How it works

1. Load the dataset

2. Preprocess the data (cleaning and splitting into training & testing sets)

3. Train the Decision Tree model

4. Test it on unseen data to check accuracy

5. Make predictions on new inputs



∆ Requirements

Python 3.x

pandas

scikit-learn

Install dependencies with:

pip install pandas scikit-learn



∆ How to run

1. Download/clone this repo

2. Open network_intrusion_detection.py in your IDE or terminal

3. Run:

python network_intrusion_detection.py



∆ Project Demo

Sample Output:

Model Accuracy: 92.7%
Prediction for sample data: ['normal', 'attack', 'normal', 'attack']

(Accuracy will vary depending on dataset and split)

You can add your own data to networkintrusion1.csv and re-run the program to see how it performs.




∆ Notes

This is a basic ML model for learning purposes

Can be improved with better datasets & algorithms





⭐ If you like this project or found it useful, give it a star on GitHub!

feel free to contact
+91 748 394 9475
