import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
patients = pd.read_csv('patients.csv')
encounters = pd.read_csv('encounters.csv')
conditions = pd.read_csv('conditions.csv')

# Display basic information about the datasets
print("Patients Dataset:")
print(patients.info())
print("\nEncounters Dataset:")
print(encounters.info())
print("\nConditions Dataset:")
print(conditions.info())

# Exploratory Data Analysis

# 1. Patient Demographics
# Age distribution
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
patients['AGE'] = (pd.to_datetime('today') - patients['BIRTHDATE']).dt.days // 365
plt.figure(figsize=(10, 6))
sns.histplot(patients['AGE'], bins=30, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png')
plt.show()

# Gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=patients, x='GENDER')
plt.title('Gender Distribution of Patients')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('gender_distribution.png')
plt.show()

# 2. Encounter Analysis
# Encounters per year
encounters['START'] = pd.to_datetime(encounters['START'])
encounters['YEAR'] = encounters['START'].dt.year
encounters_per_year = encounters['YEAR'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=encounters_per_year.index, y=encounters_per_year.values)
plt.title('Number of Encounters per Year')
plt.xlabel('Year')
plt.ylabel('Number of Encounters')
plt.savefig('encounters_per_year.png')
plt.show()

# Encounter types
plt.figure(figsize=(12, 6))
sns.countplot(data=encounters, y='ENCOUNTERCLASS', order=encounters['ENCOUNTERCLASS'].value_counts().index)
plt.title('Distribution of Encounter Types')
plt.xlabel('Count')
plt.ylabel('Encounter Class')
plt.savefig('encounter_types.png')
plt.show()

# 3. Condition Analysis
# Most common conditions
common_conditions = conditions['DESCRIPTION'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=common_conditions.values, y=common_conditions.index)
plt.title('Top 10 Most Common Conditions')
plt.xlabel('Count')
plt.ylabel('Condition')
plt.savefig('common_conditions.png')
plt.show()

# Conditions by gender
conditions_by_gender = conditions.merge(patients[['Id', 'GENDER']], left_on='PATIENT', right_on='Id')
condition_gender_counts = conditions_by_gender.groupby(['DESCRIPTION', 'GENDER']).size().unstack().fillna(0)
top_conditions = condition_gender_counts.sum(axis=1).sort_values(ascending=False).head(10)
condition_gender_counts = condition_gender_counts.loc[top_conditions.index]
condition_gender_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Top 10 Conditions by Gender')
plt.xlabel('Condition')
plt.ylabel('Count')
plt.savefig('conditions_by_gender.png')
plt.show()
