# Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Datasheet dari Tugas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
data = pd.read_csv(url)
# baris pertama datasheet dan kolom-kolomnya
print(data.head())
print("Kolom dalam dataset:")
print(data.columns)

# Grafik dari datasheet
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribusi {feature}')
    plt.show()

# grafik fitur-numerik
sns.pairplot(data[numerical_features])
plt.show()

# data
data = data.dropna()

# Memerika tool wear
if 'Tool wear [min]' not in data.columns:
    raise KeyError("Kolom 'Tool wear [min]' tidak ditemukan dalam dataset.")

# memisahkan fitur dan target
X = data.drop(columns=['Tool wear [min]', 'UDI', 'Product ID'])
y = data['Tool wear [min]']

# mentransform fitur kategorikal menjadi one-hot encoding
fitur_kategorikal = ['Type']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), fitur_kategorikal)
    ], remainder='passthrough')

X = preprocessor.fit_transform(X)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Pengembangan Model dan insialisasi
model = RandomForestClassifier(random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

# Menampilkan hasil cross-validation
print(f"Rata-rata Akurasi dari Cross-Validation: {cv_results.mean() * 100:.2f}%")
print(f"Rata-rata Error dari Cross-Validation: {(1 - cv_results.mean()) * 100:.2f}%")

# Melatih model dengan seluruh data pelatihan
model.fit(X_train, y_train)

# Memprediksi pada set pengujian
y_pred = model.predict(X_test)

# Evaluasi Model
print("Evaluasi Model Awal:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Menghitung akurasi dan error
akurasi = accuracy_score(y_test, y_pred)
error = 1 - akurasi
print(f"Akurasi Model Awal: {akurasi * 100:.2f}%")
print(f"Error Model Awal: {error * 100:.2f}%")

# Penyempurnaan Hyperparameter (contoh menggunakan GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Parameter terbaik yang ditemukan: ", grid_search.best_params_)

# Mengevaluasi model terbaik
model_terbaik = grid_search.best_estimator_
y_pred_terbaik = model_terbaik.predict(X_test)

print("Evaluasi Model Terbaik:")
print(confusion_matrix(y_test, y_pred_terbaik))
print(classification_report(y_test, y_pred_terbaik))

# Menghitung akurasi dan error untuk model terbaik
akurasi_terbaik = accuracy_score(y_test, y_pred_terbaik)
error_terbaik = 1 - akurasi_terbaik
print(f"Akurasi Model Terbaik: {akurasi_terbaik * 100:.2f}%")
print(f"Error Model Terbaik: {error_terbaik * 100:.2f}%")

# Menyimpan model  (model_dimuat = joblib.load('prediktor_keausan_alat.pkl')
joblib.dump(model_terbaik, 'prediktor_keausan_alat.pkl')

