import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("cleaned_heart_disease_data.csv")

# Set the style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Age Distribution – Histogram
plt.figure()
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Cholesterol vs Age – Scatter plot
plt.figure()
plt.scatter(df['age'], df['chol'], alpha=0.7, c='tomato', edgecolors='black')
plt.title('Cholesterol vs Age')
plt.xlabel('Age')
plt.ylabel('Cholesterol (mg/dL)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Heart disease presence by gender – Bar chart
# 0 = No disease, 1 = Disease; 0 = Female, 1 = Male
gender_map = {0.0: 'Female', 1.0: 'Male'}
df['sex_label'] = df['sex'].map(gender_map)

heart_disease_by_gender = df.groupby('sex_label')['target'].value_counts().unstack()
heart_disease_by_gender.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'])
plt.title('Heart Disease Presence by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
plt.figure()
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 5. Chest pain type frequency – Pie chart
# Assuming cp = chest pain type (1-4)
cp_counts = df['cp'].value_counts()
cp_labels = [f'Type {int(i)}' for i in cp_counts.index]

plt.figure()
plt.pie(cp_counts, labels=cp_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Chest Pain Type Frequency')
plt.axis('equal')
plt.tight_layout()
plt.show()
