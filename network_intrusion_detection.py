import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import HTML, display

# Load dataset (local file)
df = pd.read_csv("networkintrusion1.csv")

# Preprocessing: Map categorical columns and remove duplicates/nulls
df['protocol_type'] = df['protocol_type'].map({'tcp': 0, 'udp': 1, 'icmp': 2})
service_map = {service: i for i, service in enumerate(df['service'].unique())}
df['service'] = df['service'].map(service_map)
flag_map = {flag: i for i, flag in enumerate(df['flag'].unique())}
df['flag'] = df['flag'].map(flag_map)
df['class'] = df['class'].map({'normal': 0, 'anomaly': 1})

# Add a mock 'attack_type' column for demonstration
np.random.seed(1)  # For reproducibility
df['attack_type'] = np.random.choice(['DoS', 'Probe', 'R2L', 'U2R'], size=len(df))

# Map attack types to numerical values
attack_map = {attack: i for i, attack in enumerate(df['attack_type'].unique())}
df['attack_type'] = df['attack_type'].map(attack_map)

# Splitting data
y_class = df[['class']]  # Anomaly detection
y_attack = df[['attack_type']]  # Attack type
X = df.drop(columns=['class', 'attack_type'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.33, random_state=1
)
_, _, y_train_attack, y_test_attack = train_test_split(
    X, y_attack, test_size=0.33, random_state=1
)

# Train models
classifier_class = DecisionTreeClassifier(criterion="entropy", max_depth=4)
classifier_attack = DecisionTreeClassifier(criterion="entropy", max_depth=4)

classifier_class.fit(X_train, y_train_class.values.ravel())
classifier_attack.fit(X_train, y_train_attack.values.ravel())

# Function to check suspicious activity
def check_for_suspicious_data(data):
    prediction_class = classifier_class.predict([data])  # Anomaly detection
    if prediction_class == 1:  # If anomaly
        prediction_attack = classifier_attack.predict([data])  # Predict attack type
        attack_type = list(attack_map.keys())[list(attack_map.values()).index(prediction_attack[0])]
        display(HTML('<div style="color:red; font-weight:bold;">🚨 ALERT: Suspicious activity detected! 🚨</div>'))
        print(f"Alert: Suspicious activity detected! Attack type: {attack_type}")
    else:
        print("No suspicious activity detected.")

# Test the function
for i in range(5):
    check_for_suspicious_data(X_test[i])