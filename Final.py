import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv('/Users/esrayanar/Desktop/CovidVaccinations2023.csv')

# Cleaning the 'FullyVaccinated' column
df['FullyVaccinated'] = df['FullyVaccinated'].str.replace('‚Äì', '').str.replace('%', '').str.replace('>', '')

# Converting the 'FullyVaccinated' column to numeric values
df['FullyVaccinated'] = pd.to_numeric(df['FullyVaccinated'], errors='coerce')

# List of feature columns 
features = ['Vaccinated', 'AdditionalDosesPer100people', 'AdditionalDosesTotal', 'DosesAdministeredPer100people', 'DosesAdministeredTotal']

# Cleaning the percentage columns in the feature set
for feature in features:
    if df[feature].dtype == 'object':
        df[feature] = df[feature].str.replace('‚Äì', '').str.replace('%', '').str.replace('>', '')
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

# Drop rows with missing values in the 'FullyVaccinated' column
df.dropna(subset=['FullyVaccinated'], inplace=True)

# Printing the cleaned DataFrame
print(df)


# Defining your features (X) and target (y)
X = df[['Vaccinated', 'AdditionalDosesPer100people', 'AdditionalDosesTotal', 'DosesAdministeredPer100people', 'DosesAdministeredTotal']]
y = df['FullyVaccinated']

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a SimpleImputer with strategy 'mean'
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your feature data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Creating a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predicting on the test data
y_pred = model.predict(X_test)

# Evaluating your model (you can add more evaluation metrics as needed)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def classify_vaccination_rate(rate):
    if rate >= 70:
        return 'High'
    elif rate >= 40:
        return 'Medium'
    else:
        return 'Low'

# Applying the classification function to create the target variable
df['VaccinationRateCategory'] = df['FullyVaccinated'].apply(classify_vaccination_rate)

# Defining your features (X_classification) and target (y_classification) for classification
X_classification = df[['Vaccinated', 'AdditionalDosesPer100people', 'AdditionalDosesTotal', 'DosesAdministeredPer100people', 'DosesAdministeredTotal']]
y_classification = df['VaccinationRateCategory']

# Spliting the data into training and testing sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Creating a SimpleImputer with strategy 'mean' for classification
imputer_classification = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your feature data for classification
X_train_classification = imputer_classification.fit_transform(X_train_classification)
X_test_classification = imputer_classification.transform(X_test_classification)

# Creating a Random Forest Classifier model
classifier = RandomForestClassifier(random_state=42)

# Fitting the classifier to the training data
classifier.fit(X_train_classification, y_train_classification)

# Predicting the categories on the test data
y_pred_classification = classifier.predict(X_test_classification)

# Evaluating classifier 
accuracy = accuracy_score(y_test_classification, y_pred_classification)
classification_rep = classification_report(y_test_classification, y_pred_classification)
confusion_mat = confusion_matrix(y_test_classification, y_pred_classification)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(confusion_mat)

import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('/Users/esrayanar/Desktop/CovidVaccinations2023.csv')

# Sort the data by vaccination rate in descending order
df.sort_values(by='FullyVaccinated', ascending=False, inplace=True)

# Select the top N countries for visualization
top_countries = df.head(10)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.barh(top_countries['Country'], top_countries['FullyVaccinated'], color='skyblue')
plt.xlabel('Fully Vaccinated Percentage (%)')
plt.ylabel('Country')
plt.title('Top 10 Countries with Highest Fully Vaccinated Percentage')
plt.gca().invert_yaxis()  # Invert the y-axis to show the highest percentage at the top
plt.tight_layout()

# Show the plot
plt.show()
