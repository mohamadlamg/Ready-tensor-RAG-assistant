import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

# ============================================================================
# PIPELINE E-COMMERCE : PRÉDICTION D'ACHAT CLIENT
# ============================================================================

# Simulation de données e-commerce réalistes
np.random.seed(42)
n_samples = 5000

# Génération des données
data = {
    # Données comportementales
    'pages_visited': np.random.poisson(8, n_samples),
    'time_on_site': np.random.exponential(300, n_samples),  # secondes
    'previous_purchases': np.random.poisson(2, n_samples),
    'cart_value': np.random.gamma(2, 50, n_samples),  # euros
    
    # Données démographiques
    'age': np.random.normal(35, 12, n_samples),
    'income': np.random.lognormal(10.5, 0.5, n_samples),  # Distribution log-normale typique des revenus
    
    # Données catégorielles
    'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.6, 0.3, 0.1]),
    'traffic_source': np.random.choice(['organic', 'paid', 'social', 'email', 'direct'], n_samples),
    'user_segment': np.random.choice(['new', 'returning', 'vip'], n_samples, p=[0.4, 0.5, 0.1]),
    'region': np.random.choice(['Europe', 'North_America', 'Asia', 'Other'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
}

df = pd.DataFrame(data)

# Ajout de valeurs manquantes (réaliste)
missing_mask_income = np.random.choice([True, False], n_samples, p=[0.15, 0.85])  # 15% manquant
missing_mask_age = np.random.choice([True, False], n_samples, p=[0.08, 0.92])     # 8% manquant

df.loc[missing_mask_income, 'income'] = np.nan
df.loc[missing_mask_age, 'age'] = np.nan

# Variable cible : probabilité d'achat basée sur des règles business réalistes
purchase_probability = (
    0.1 +  # base probability
    0.3 * (df['previous_purchases'] > 0).astype(float) +  # clients existants
    0.2 * (df['time_on_site'] > 180).astype(float) +     # temps passé
    0.2 * (df['cart_value'] > 100).astype(float) +       # panier élevé
    0.1 * (df['user_segment'] == 'vip').astype(float) +  # clients VIP
    0.1 * (df['pages_visited'] > 10).astype(float)       # exploration intensive
)

# Ajout de bruit et conversion en décision binaire
purchase_probability += np.random.normal(0, 0.1, n_samples)
df['will_purchase'] = (np.random.random(n_samples) < purchase_probability).astype(int)

print("=== DONNÉES E-COMMERCE ===")
print(f"Nombre total de clients: {len(df)}")
print(f"Taux de conversion: {df['will_purchase'].mean():.1%}")
print(f"\nValeurs manquantes par colonne:")
print(df.isnull().sum().sort_values(ascending=False))
print(f"\nAperçu des données:")
print(df.head())

# ============================================================================
# DÉFINITION DES TYPES DE VARIABLES
# ============================================================================

# Variables numériques standard
numeric_standard = ['pages_visited', 'previous_purchases', 'age']

# Variables numériques avec distribution asymétrique (transformation log)
numeric_skewed = ['time_on_site', 'cart_value', 'income']

# Variables catégorielles
categorical_features = ['device', 'traffic_source', 'user_segment', 'region']

print(f"\n=== ARCHITECTURE DU PIPELINE ===")
print(f"Variables numériques standard: {numeric_standard}")
print(f"Variables numériques asymétriques: {numeric_skewed}")
print(f"Variables catégorielles: {categorical_features}")

# ============================================================================
# CONSTRUCTION DU PIPELINE SPÉCIALISÉ E-COMMERCE
# ============================================================================

# Transformateur pour variables numériques standard
standard_numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformateur pour variables asymétriques (log + scale)
skewed_numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', FunctionTransformer(lambda x: np.log1p(x), validate=False)),  # log(1+x)
    ('scaler', StandardScaler())
])

# Transformateur pour variables catégorielles
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Assemblage avec ColumnTransformer
preprocessor = ColumnTransformer([
    ('standard_num', standard_numeric_transformer, numeric_standard),
    ('skewed_num', skewed_numeric_transformer, numeric_skewed),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

# ============================================================================
# PIPELINES COMPLETS AVEC DIFFÉRENTS MODÈLES
# ============================================================================

# Pipeline 1: Logistic Regression (interprétable)
pipeline_interpretable = Pipeline([
    ('preprocessing', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=15)),  # Sélection des 15 meilleures features
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Pipeline 2: Random Forest (performance)
pipeline_performance = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42
    ))
])

# ============================================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================

# Séparation des données
X = df.drop('will_purchase', axis=1)
y = df['will_purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=== ENTRAÎNEMENT DES MODÈLES ===")
print(f"Taille train: {X_train.shape[0]} clients")
print(f"Taille test: {X_test.shape[0]} clients")

# Entraînement des pipelines
pipelines = {
    'Logistic Regression': pipeline_interpretable,
    'Random Forest': pipeline_performance
}

results = {}

for name, pipeline in pipelines.items():
    print(f"\n--- {name} ---")
    
    # Validation croisée
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Entraînement final
    pipeline.fit(X_train, y_train)
    
    # Évaluation sur test
    test_score = pipeline.score(X_test, y_test)
    print(f"Test Accuracy: {test_score:.3f}")
    
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score,
        'pipeline': pipeline
    }

# ============================================================================
# ANALYSE DES FEATURES (POUR LE MODÈLE INTERPRÉTABLE)
# ============================================================================

print(f"\n=== ANALYSE DES VARIABLES IMPORTANTES ===")

# Récupération du meilleur modèle
best_pipeline = pipeline_interpretable

# Features sélectionnées
feature_selector = best_pipeline.named_steps['feature_selection']
selected_features_mask = feature_selector.get_support()

# Noms des features après preprocessing
preprocessor_fitted = best_pipeline.named_steps['preprocessing']

# Récupération des noms de colonnes
feature_names = []

# Features numériques standard
feature_names.extend(numeric_standard)

# Features numériques asymétriques  
feature_names.extend([f"{col}_log" for col in numeric_skewed])

# Features catégorielles (après OneHot)
cat_transformer = preprocessor_fitted.named_transformers_['cat']
if hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
    cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names.extend(cat_feature_names)

# Features sélectionnées
selected_feature_names = [name for i, name in enumerate(feature_names) if selected_features_mask[i]]
print(f"Features sélectionnées: {selected_feature_names}")

# Coefficients du modèle logistique
coefficients = best_pipeline.named_steps['classifier'].coef_[0]
feature_importance = list(zip(selected_feature_names, coefficients))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\nTop 10 variables les plus importantes:")
for i, (feature, coef) in enumerate(feature_importance[:10]):
    print(f"{i+1:2d}. {feature:<25} : {coef:>7.3f}")

# ============================================================================
# SIMULATION DE NOUVEAUX CLIENTS
# ============================================================================

print(f"\n=== PRÉDICTIONS SUR NOUVEAUX CLIENTS ===")

# Simulation de 3 profils de clients différents
new_clients = pd.DataFrame({
    'pages_visited': [15, 3, 8],
    'time_on_site': [450, 120, 300],
    'previous_purchases': [5, 0, 1],
    'cart_value': [200, 50, 75],
    'age': [28, 45, 35],
    'income': [45000, 65000, np.nan],  # Valeur manquante pour tester
    'device': ['mobile', 'desktop', 'mobile'],
    'traffic_source': ['organic', 'paid', 'social'],
    'user_segment': ['vip', 'new', 'returning'],
    'region': ['Europe', 'North_America', 'Europe']
})

print("Profils des nouveaux clients:")
print(new_clients)

# Prédictions avec probabilités
probabilities = best_pipeline.predict_proba(new_clients)
predictions = best_pipeline.predict(new_clients)

print(f"\nRésultats des prédictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Client {i+1}: {'Achètera' if pred == 1 else 'N''achètera pas'} "
          f"(probabilité: {prob[1]:.2%})")

 