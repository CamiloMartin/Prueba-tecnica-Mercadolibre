import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler


class ModelPredictor:   
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        self.X_original = X
        self.y_original = y
        self.test_size = test_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.xgb_model = None
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def prepare_data(self, balance: bool = True, strategy: float = 0.80):
        # Hago un balanceo de los datos porque la clase positiva es minoritaria
        print("\n--- Preparación de Datos ---")
        
        X, y = self.X_original, self.y_original
        
        if balance:
            rus = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
            X, y = rus.fit_resample(X, y)
            print(f"Datos balanceados (estrategia {strategy})")
        
        # Divido en train y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        print(f"Train: {len(self.X_train):,} | Test: {len(self.X_test):,}")
        
    def train_models(self):
        ################### Entrenamiento de los dos modelos ###################
        print("\n--- Entrenamiento de Modelos ---")
        
        # Random Forest
        print("Entrenando Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20,
            random_state=42, n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train)
        
        y_pred_rf = self.rf_model.predict(self.X_test)
        y_proba_rf = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        self.results['random_forest'] = {
            'model': self.rf_model,
            'accuracy': accuracy_score(self.y_test, y_pred_rf),
            'roc_auc': roc_auc_score(self.y_test, y_proba_rf)
        }
        
        print(f"  Accuracy: {self.results['random_forest']['accuracy']:.4f}")
        print(f"  ROC-AUC:  {self.results['random_forest']['roc_auc']:.4f}")
        
        # XGBoost
        print("\nEntrenando XGBoost...")
        # En mi notebook use gridseacrh para optimizar hiperparámetros y estos fueron los mejores
        self.xgb_model = XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.1,
            random_state=42, scale_pos_weight=2.0, eval_metric='logloss'
        )
        self.xgb_model.fit(self.X_train, self.y_train)
        
        y_pred_xgb = self.xgb_model.predict(self.X_test)
        y_proba_xgb = self.xgb_model.predict_proba(self.X_test)[:, 1]
        
        self.results['xgboost'] = {
            'model': self.xgb_model,
            'accuracy': accuracy_score(self.y_test, y_pred_xgb),
            'roc_auc': roc_auc_score(self.y_test, y_proba_xgb)
        }
        
        print(f"  Accuracy: {self.results['xgboost']['accuracy']:.4f}")
        print(f"  ROC-AUC:  {self.results['xgboost']['roc_auc']:.4f}")
    
    def cross_validate(self, cv: int = 5):
        # Valifación cruzada con 5 folds
        print(f"\n--- Validación Cruzada ({cv}-Fold) ---")
        
        cv_rf = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        cv_xgb = cross_val_score(self.xgb_model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        
        self.results['random_forest']['cv_mean'] = cv_rf.mean()
        self.results['random_forest']['cv_std'] = cv_rf.std()
        self.results['xgboost']['cv_mean'] = cv_xgb.mean()
        self.results['xgboost']['cv_std'] = cv_xgb.std()
        
        print(f"Random Forest: {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")
        print(f"XGBoost:       {cv_xgb.mean():.4f} (+/- {cv_xgb.std():.4f})")
    
    def select_best_model(self, metric: str = 'accuracy'):
        # Mejor modelo según métrica
        best_score = 0
        best_name = None
        
        for name, results in self.results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\n Mejor Modelo: {best_name.upper()} ({metric}: {best_score:.4f})")
        
        return self.best_model, best_name
    
    def show_classification_report(self):
        # Reporte de clasificación del mejor modelo
        y_pred = self.best_model.predict(self.X_test)
        print(f"\n--- Reporte de Clasificación mejor modelo ({self.best_model_name.upper()}) ---")
        print(classification_report(self.y_test, y_pred, target_names=['Sin Ventas', 'Con Ventas']))
    
    def plot_feature_importance(self, save_path: str = 'feature_importance.png'):
        # Feature Importance del mejor modelo
        importances = self.best_model.feature_importances_
        feature_names = self.X_train.columns
        
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n--- Top 5 Features Más Importantes ---")
        for idx, row in feature_imp.head(5).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_imp['feature'], feature_imp['importance'])
        plt.xlabel('Importancia')
        plt.title(f'Feature Importance - {self.best_model_name.upper()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGráfico guardado: {save_path}")
        
        return feature_imp
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        # Predicciones con el mejor modelo
        return self.best_model.predict(X_new)
    
    def get_results_summary(self) -> dict:
        # Mostra un resumen de los resultados
        summary = {}
        for name, results in self.results.items():
            summary[name] = {
                'accuracy': results.get('accuracy'),
                'roc_auc': results.get('roc_auc'),
                'cv_mean': results.get('cv_mean'),
                'cv_std': results.get('cv_std')
            }
        return summary