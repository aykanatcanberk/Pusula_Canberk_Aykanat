# Pusula_Canberk_Aykanat

## Contact Information
**Name:** Canberk Aykanat  
**Email:** [kayra.aykanat@gmail.com](mailto:kayra.aykanat@gmail.com)

## Project Overview

This project involves data processing, visualization, and the application of machine learning techniques on a dataset related to medical side effects. It covers various stages of data manipulation, from handling missing values to creating machine learning models for classification purposes.

The main objectives of the project include:
- Data exploration and visualization using libraries such as `pandas`, `matplotlib`, and `seaborn`.
- Handling missing values through imputation and managing categorical data.
- Detecting outliers using Z-score and preprocessing data via scaling and encoding.
- Training and evaluating machine learning models, including K-Nearest Neighbors (KNN) classification.
- Performing statistical analysis such as the Chi-Square test to determine relationships between categorical variables.
- Visualizing correlations and distribution patterns related to side effects and other variables.

# Project Structure

- `side_effect_data.xlsx`: Dataset used for the project.
- `Data_Science.ipynb`: Jupyter notebook containing the code for data processing, analysis, and machine learning.
- `README.md`: This file, providing an overview of the project.

# Requirements and Libraries

This project requires the following libraries, each used for data analysis, visualization, machine learning, and statistical analysis.

## Data Processing and Basic Python Libraries

- **pandas**: For data structures and data analysis.
- **numpy**: For numerical computations and handling large datasets.

## Visualization Libraries

- **matplotlib**: The fundamental library for data visualization.
- **seaborn**: Built on top of matplotlib, used for more aesthetic and complex visualizations.

## Statistics and Analysis Functions

- **scipy**: For statistical tests and mathematical functions, particularly the `chi2_contingency` and `zscore` functions.

## Machine Learning - Data Preprocessing

- **sklearn**: The core library for machine learning.
  - **SimpleImputer**: For filling in missing data.
  - **LabelEncoder**: For converting categorical data into numerical data.
  - **StandardScaler**: For standardizing data.
  - **MinMaxScaler**: For scaling data to a specific range.

## Machine Learning - Modeling

- **RandomForestClassifier**: For classification problems using the random forest algorithm.
- **LinearRegression**: For regression problems.
- **KNeighborsClassifier**: For k-nearest neighbors classification.

## Machine Learning - Evaluation Metrics

- **classification_report**: To evaluate the performance of classification models.
- **confusion_matrix**: For a detailed analysis of classification results.
- **mean_squared_error**: For measuring error in regression models.
- **r2_score**: For assessing the explanatory power of regression models.

# Exploratory Data Analysis (EDA)

In this section, we perform Exploratory Data Analysis (EDA) on the dataset `side_effect_data.xlsx`. The goal of EDA is to summarize the main characteristics of the data, often using visual methods.

## Loading the Data

First, we load the dataset and take a look at the first few rows:

```python
df = pd.read_excel('side_effect_data.xlsx')
df.head()
```

This command displays the first five rows of the dataset, allowing us to understand its structure and contents.

## Data Overview

Next, we obtain a summary of the DataFrame:

```python
df.info()
```

This method provides information on the number of entries, data types, and memory usage, helping us understand the dataset's characteristics.

## Missing Values

To check for missing values in the dataset, we use:

```python
df.isnull().sum()
```

This command counts the number of missing values in each column, which is crucial for data cleaning.

## Statistical Summary

We can also obtain a statistical summary of the numerical columns with:

```python
df.describe()
```

This command provides key statistical metrics, such as mean, standard deviation, min, and max values, offering insights into the distribution of numerical features.

## Categorical Data Distribution

To visualize the distribution of categorical data (e.g., Gender), we create a count plot:

```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Cinsiyet')
plt.title('Cinsiyet Dağılımı')
plt.show()
```

This plot shows the count of each gender in the dataset, giving a visual representation of the distribution.

## Histogram of Numerical Data

For numerical data (e.g., Weight), we visualize the distribution using a histogram:

```python
plt.figure(figsize=(10, 6))
sns.histplot(df['Kilo'].dropna(), bins=20, kde=True)
plt.title('Kilo Dağılımı')
plt.show()
```

This histogram illustrates the weight distribution, with a kernel density estimate (KDE) overlay for better visualization of the distribution shape.

## Missing Data Visualization

To visualize missing data, we can use a heatmap:

```python
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Eksik Veri Görselleştirmesi')
plt.show()
```

This heatmap helps us see the pattern of missing values in the dataset.

## Correlation Matrix

Next, we explore the correlations between numerical features:

```python
df_numeric = df.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()

sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```

This clustermap shows the correlation coefficients between numerical variables, helping us identify relationships.

## Final Check for Missing Values

Finally, we check for missing values after any potential imputation:

```python
missing_values_after_imputation = df.isnull().sum()
print(missing_values_after_imputation)
```

This command re-evaluates the presence of missing data to confirm whether it has been addressed.

# Data Pre-Processing

In this section, we focus on cleaning and preparing the dataset for analysis. This includes handling missing values, encoding categorical variables, normalizing numerical features, detecting outliers, and generating new features.

## Separating Numerical and Categorical Columns

We start by separating numerical and categorical columns:

```python
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns
```

This classification helps in applying specific preprocessing techniques to each type of variable.

## Handling Missing Values

### Numerical Columns

For numerical columns, we use the mean to fill in missing values:

```python
imputer_num = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer_num.fit_transform(df[numeric_columns])
```

### Categorical Columns

For categorical columns, we fill missing values with the mode (most frequent value):

```python
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])
```

## Encoding Categorical Variables

We convert categorical variables into numerical form using label encoding:

```python
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le
```

This transformation allows categorical data to be used in machine learning models.

## Normalization

To standardize the scale of numerical features, we apply Min-Max Scaling:

```python
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
```

Normalization is important for improving model performance and convergence speed.

## Outlier Detection

We detect outliers using the Z-score method. A Z-score indicates how many standard deviations an element is from the mean:

```python
def detect_outliers_zscore(df):
    outliers = {}
    for column in numeric_columns:
        z_scores = zscore(df[column])
        outliers[column] = df[(z_scores > 1.9) | (z_scores < -1.9)]
        print(f"{column} kolonunda tespit edilen outlier sayısı: {len(outliers[column])}")

    return outliers

outliers = detect_outliers_zscore(df)
```

We print the number of outliers detected in each numerical column.

## Feature Generation

### Age Calculation

We calculate age based on the birthdate:

```python
current_year = datetime.now().year
df['Yas'] = current_year - df['Dogum_Tarihi'].dt.year
```

### Age Groups

We create age groups for better categorization:

```python
bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '60+']
df['Yas_Grubu'] = pd.cut(df['Yas'], bins=bins, labels=labels)
```

### Medication Duration

We compute the duration of medication in days:

```python
df['Ilac_Suresi'] = (df['Ilac_Bitis_Tarihi'] - df['Ilac_Baslangic_Tarihi']).dt.days
```

### Body Mass Index (BMI)

We calculate the BMI using weight and height:

```python
df['VKİ'] = df['Kilo'] / (df['Boy'] / 100) ** 2
df['VKİ'].fillna(df['VKİ'].mean())
```

### Adverse Effect Duration

We compute the duration of adverse effect reporting:

```python
df['Yan_Etki_Suresi'] = (df['Yan_Etki_Bildirim_Tarihi'] - df['Ilac_Baslangic_Tarihi']).dt.days
```

### Allergy and Chronic Disease Indicators

We create binary indicators for allergies and chronic diseases:

```python
df['Alerji_Varligi'] = df['Alerjilerim'].apply(lambda x: 1 if x > 0 else 0)
df['Kronik_Hastalik_Varligi'] = df[['Baba Kronik Hastaliklari', 'Anne Kronik Hastaliklari', 
                                     'Kiz Kardes Kronik Hastaliklari', 'Erkek Kardes Kronik Hastaliklari']].any(axis=1).astype(int)
```

## Replacing Infinite Values

We replace infinite values with NaN and then fill NaN values with the mean of their respective columns:

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
```

## Removing Outliers

Finally, we remove detected outliers from the dataset:

```python
for column, outlier_df in outliers.items():
    df = df[~df.index.isin(outlier_df.index)]
```

## Visualization of Relationships

We visualize the relationship between medications and adverse effects, as well as the distribution of adverse effect duration and correlations among variables.

### Example: Medication and Adverse Effect Relationship

```python
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Ilac_Adi', hue='Yan_Etki')
plt.xticks(rotation=90)
plt.title('İlaç ve Yan Etki İlişkisi')
plt.show()
```

This process prepares the dataset for further analysis and model training.

# K-Nearest Neighbors (KNN) Classifier

In this section, we implement a K-Nearest Neighbors (KNN) classifier to predict the health status of individuals based on various features. The process includes data preparation, model training, and evaluation.

## Health Status Classification

We first define the health status based on the adverse effect duration and chronic disease presence:

```python
df['Yan_Etki_Durumu'] = np.where((df['Yan_Etki_Suresi'] > 0) & (df['Yan_Etki'] == 1), 1, 0)
df['Saglik_Durumu'] = np.where((df['Kronik_Hastalik_Varligi'] == 1) & (df['Yan_Etki_Durumu'] == 1), 'Kötü',
                                np.where((df['Kronik_Hastalik_Varligi'] == 1) & (df['Yan_Etki_Durumu'] == 0), 'Orta',
                                         np.where((df['Kronik_Hastalik_Varligi'] == 0) & (df['Yan_Etki_Durumu'] == 1), 'Orta',
                                                  'İyi')))
```

### Features and Target Variable

We define our feature set `X` and target variable `y`:

```python
X = df[['Cinsiyet', 'Kilo', 'Boy', 'Yas', 'Ilac_Suresi', 'VKİ', 'Yan_Etki_Suresi']]
y = df['Saglik_Durumu']
```

## Handling Missing Values

To ensure that our features are complete, we use `SimpleImputer` to fill missing values with the mean:

```python
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
```

We also convert the target variable to numerical format:

```python
y, unique = pd.factorize(y)  # "İyi": 0, "Orta": 1, "Kötü": 2
```

## Train-Test Split

Next, we split the dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

We check the distribution of classes in both sets:

```python
print("Eğitim seti sınıf dağılımı:\n", pd.Series(y_train).value_counts())
print("Test seti sınıf dağılımı:\n", pd.Series(y_test).value_counts())
```

## Feature Scaling

To improve the performance of the KNN algorithm, we standardize the feature values:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Hyperparameter Tuning with GridSearchCV

We use `GridSearchCV` to find the optimal number of neighbors for our KNN model:

```python
param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print("En iyi parametreler:", grid_search.best_params_)
```

We retrieve the best estimator and fit it to the training data:

```python
best_knn = grid_search.best_estimator_
best_knn.fit(X_train_scaled, y_train)
```

## Model Prediction and Evaluation

We predict the health status on the test set and evaluate the model's performance:

```python
y_pred = best_knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred, zero_division=0))
```

### Confusion Matrix Visualization

To visualize the model's performance, we create a confusion matrix:

```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrisi')
plt.xlabel('Gerçek')
plt.ylabel('Tahmin')
plt.show()
```

## Feature Importance Analysis

To understand the importance of each feature, we utilize a Random Forest classifier:

```python
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)

importances = rf.feature_importances_
features = ['Cinsiyet', 'Kilo', 'Boy', 'Yas', 'Ilac_Suresi', 'VKİ', 'Yan_Etki_Suresi']
```

We visualize the importance of each feature:

```python
plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=features)
plt.title('Özelliklerin Önem Dereceleri')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellikler')
plt.show()
```

This KNN classification process provides a robust framework for predicting health status based on various health-related features.
