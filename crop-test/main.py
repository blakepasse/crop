import pandas as pd

# Load the dataset
file_path = '~/Desktop/projects/crop/crop-test/Crop_Recommendation.csv'
crop_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
crop_data.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode the target variable
label_encoder = LabelEncoder()
crop_data['Crop'] = label_encoder.fit_transform(crop_data['Crop'])

# Split the dataset into features and target variable
X = crop_data.drop(columns=['Crop'])
y = crop_data['Crop']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

accuracy, classification_rep

print(accuracy)