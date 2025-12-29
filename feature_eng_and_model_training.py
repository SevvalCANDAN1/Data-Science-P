import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-learn Imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

# Diğer Kütüphaneler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
from scipy.stats import randint

# --- 1. Değerlendirme Fonksiyonu (Geliştirilmiş) ---
def evaluate_model(y_true, y_pred, y_probs, model_name, plot_curves=True):
    save_dir = "plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n{'='*20} {model_name} Results {'='*20}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    
    # ROC AUC (Eğer olasılıklar varsa)
    if y_probs is not None:
        roc_score = roc_auc_score(y_true, y_probs)
        print("ROC AUC Score:", roc_score)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        print("Precision-Recall AUC:", pr_auc)
    
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    if plot_curves:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.savefig(f"{save_dir}/{model_name}_confusion_matrix.png")
        plt.close()
        
        if y_probs is not None:
            # ROC Curve
            RocCurveDisplay.from_predictions(y_true, y_probs, name=model_name)
            plt.title(f"{model_name} - ROC Curve")
            plt.savefig(f"{save_dir}/{model_name}_roc_curve.png")
            plt.close()
            
            # Precision-Recall Curve
            PrecisionRecallDisplay(precision=precision, recall=recall).plot()
            plt.title(f"{model_name} - Precision-Recall Curve")
            plt.savefig(f"{save_dir}/{model_name}_precision_recall_curve.png")
            plt.close()

# --- 2. Veri Hazırlama ve Feature Engineering ---
continent_map = {
    "Europe": ["Serbia", "Turkey", "United Kingdom", "Germany", "Sweden", "Romania", "Greece", "Ukraine", "Croatia", "Ireland", "Georgia", "Bosnia And Herzegovina", "Andorra", "Netherlands", "Bulgaria", "Moldova, Republic Of", "Finland", "Belarus", "Poland", "Czech Republic", "Norway", "Italy", "Latvia", "Armenia", "Portugal", "France", "Austria", "Estonia", "Macedonia", "Kosovo", "Kazakhstan", "Montenegro", "Albania", "Slovenia", "Cyprus"],
    "Asia": ["India", "Pakistan", "Bangladesh", "Malaysia", "Sri Lanka", "Georgia", "Iraq", "Palestinian Territory", "Indonesia", "United Arab Emirates", "Jordan", "Lebanon", "Vietnam", "Armenia", "Qatar", "Uzbekistan", "Saudi Arabia", "Kazakhstan", "Japan", "Nepal", "Tajikistan", "Taiwan", "Singapore", "Cambodia", "Thailand", "Kyrgyzstan"],
    "Africa": ["Nigeria", "Egypt", "Tunisia", "Morocco", "Malawi", "Cameroon", "Ethiopia", "Cote D'ivoire", "Madagascar", "Ghana", "Libya", "Zimbabwe", "Algeria", "Mauritius", "Liberia", "Tanzania, United Republic Of", "South Africa"],
    "North America": ["United States", "Canada", "Mexico", "Jamaica", "Panama", "El Salvador", "Costa Rica", "Dominican Republic", "Trinidad And Tobago", "Nicaragua"],
    "South America": ["Venezuela", "Colombia", "Argentina", "Peru", "Uruguay", "Brazil", "Chile", "Paraguay", "Ecuador"],
    "Oceania": ["Australia", "New Zealand"],
    "Other": []
}

def country_to_continent(country):
    for continent, countries in continent_map.items():
        if country in countries:
            return continent
    return "Other"

# Veri Yükleme
df = pd.read_csv("anonymized_data.csv") # Dosya adınızın doğru olduğundan emin olun
df.dropna(subset=["Country"], inplace=True)

# Feature Engineering
df["Continent"] = df["Country"].apply(country_to_continent)
df = pd.get_dummies(df, columns=['Continent'], drop_first=False)

percentage_columns = ['Completed Jobs ', 'On Time', 'On Budget', 'Repeat Hire Rate']
for col in percentage_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Hataları NaN yap

# Target Belirleme
toplam = df['Accept Rate'].sum()
adet = df['Accept Rate'].count()
ortalama = int(toplam / adet)
print(f"Hesaplanan Ortalama Kabul Oranı: {ortalama}")
df['High Accept Rate'] = (df['Accept Rate'] > ortalama).astype(int)

# Diğer Feature'lar
df['Amount Earned'] = df['Amount Earned'] * 1000
df['USD/Hour'] = pd.to_numeric(df['USD/Hour'], errors='coerce')
df['Worked Hours'] = (df['Amount Earned'] / df['USD/Hour']).replace([np.inf, -np.inf], np.nan).fillna(0)
df['Rating Strength'] = (df['Overall Rating'] * np.log10(1 + df['Reviews Count'])).fillna(0)

weight_map = {0: 0.5, 1: 1, 2: 2, 3: 4, None: 0}
df['Work_Experience_Weighted'] = df['Work Experience'].map(weight_map)
df['Education'] = df['Education'].astype(int)
df['Education_x_WorkExp'] = df['Education'] * df['Work Experience']
df['Education_x_Certificates'] = df['Education'] * df['Number of Certificates']

bins = [-1, 0, 3, 7, 15]
labels = ['No Certificates', 'Few', 'Medium', 'Many']
df['Certificates_Binned'] = pd.cut(df['Number of Certificates'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['Certificates_Binned'], drop_first=True)

# Özellik Seçimi
cert_bins_cols = [col for col in df.columns if col.startswith('Certificates_Binned_')]
continent_cols = [col for col in df.columns if col.startswith("Continent_")]

features = [
    'Completed Jobs ', 'On Time', 'On Budget', 'Repeat Hire Rate',
    'USD/Hour', 'Worked Hours', 'Rating Strength', 'Education',  'Work_Experience_Weighted', 
    'Education_x_Certificates', 'Education_x_WorkExp',
] + cert_bins_cols + continent_cols

# Veriyi Hazırla
X = df[features]
y = df['High Accept Rate']

# 3. Split, Scale, Impute, SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Imputation (Eksik veriler için)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled) # Test verisine de aynısını uygula!

# SMOTE (Sadece train setine uygulanır)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)

print(f"Eğitim Seti Boyutu (SMOTE Sonrası): {X_resampled.shape}")

# --- 4. MODELLERİN EĞİTİMİ VE DEĞERLENDİRİLMESİ ---

# --- MODEL 1: XGBoost ---
print("\n--- Model 1: XGBoost Eğitiliyor... ---")
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_resampled, y_resampled)

y_pred_xgb = xgb_model.predict(X_test_imputed)
y_probs_xgb = xgb_model.predict_proba(X_test_imputed)[:, 1]

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(xgb_model, importance_type='weight', height=0.5, show_values=True, ax=ax)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("plots/XGBoost_Feature_Importance.png") # Kaydet
plt.show()

evaluate_model(y_test, y_pred_xgb, y_probs_xgb, model_name="XGBoost")


# --- MODEL 2: Random Forest (Base) ---
print("\n--- Model 2: Random Forest (Base) Eğitiliyor... ---")
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_resampled, y_resampled)

y_pred_rf = rf_base.predict(X_test_imputed)
y_probs_rf = rf_base.predict_proba(X_test_imputed)[:, 1]

# RF Feature Importance
importances = rf_base.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20]) 
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances (Top 20)')
plt.gca().invert_yaxis()
plt.savefig("plots/RandomForest_Feature_Importance.png")
plt.show()

evaluate_model(y_test, y_pred_rf, y_probs_rf, model_name="RandomForest_Base")


# --- MODEL 3: Random Forest (Tuned with RandomizedSearchCV) ---
print("\n--- Model 3: Random Forest (Tuned) Aranıyor... ---")
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10, 
    cv=3,      
    n_jobs=-1, 
    scoring='f1', 
    random_state=42
)

random_search.fit(X_resampled, y_resampled)
print("Best parameters:", random_search.best_params_)

best_rf_model = random_search.best_estimator_
y_pred_tuned = best_rf_model.predict(X_test_imputed)
y_probs_tuned = best_rf_model.predict_proba(X_test_imputed)[:, 1]

evaluate_model(y_test, y_pred_tuned, y_probs_tuned, model_name="RandomForest_Tuned")


# --- MODEL 4: Logistic Regression (PCA - Görselleştirme ve Değerlendirme) ---
print("\n--- Model 4: Logistic Regression (PCA - Görselleştirme) ---")

# 1. PCA Uygulama
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(X_train_imputed)
X_pca_test = pca.transform(X_test_imputed) # Test setini de dönüştür

# 2. Modeli PCA verisiyle eğit
log_model_pca = LogisticRegression()
log_model_pca.fit(X_pca_train, y_train) # Dikkat: SMOTE uygulanmamış y_train ile görsel amaçlı

# 3. TAHMİN ADIMI (Eksik olan kısım burasıydı)
# Evaluate fonksiyonuna gönderebilmek için test seti üzerinde tahmin yapıyoruz
y_pred_pca = log_model_pca.predict(X_pca_test)
y_probs_pca = log_model_pca.predict_proba(X_pca_test)[:, 1]

# 4. Karar Sınırı Çizimi
plt.figure(figsize=(10, 6))
disp = DecisionBoundaryDisplay.from_estimator(
    log_model_pca,
    X_pca_train,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.6
)
scatter = plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Logistic Regression Karar Sınırı (PCA ile 2D)")
plt.legend(*scatter.legend_elements(), title="Sınıf")
plt.savefig("plots/PCA_Decision_Boundary.png")
plt.show()

# 5. Değerlendirme
evaluate_model(y_test, y_pred_pca, y_probs_pca, model_name="LogisticRegression_PCA")