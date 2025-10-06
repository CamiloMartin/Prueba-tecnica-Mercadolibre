import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler


class FeatureEngineer:  
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_list = []
        
    def create_all_features(self) -> pd.DataFrame:
        # Crear todas las features necesarias
        print("\n--- Creando Features ---")
        
        # Features de precio, suavizo con el logaritmo el precio
        self.df['log_price'] = np.log1p(self.df['price'])
        self.df['log_base_price'] = np.log1p(self.df['base_price'])
        
        # Creación de ratio precio-categoría, precio-medio categoría
        category_stats = self.df.groupby('category_id')['price'].agg(['mean']).to_dict('index')
        self.df['price_category_ratio'] = self.df.apply(
            lambda row: row['price'] / category_stats.get(row['category_id'], {}).get('mean', 1), axis=1
        )
        
        # Features de tiempo
        self.df['date_created_dt'] = pd.to_datetime(self.df['date_created'], errors='coerce')
        self.df['year_month'] = self.df['date_created_dt'].dt.to_period('M')
        
        # Filtrar fechas antiguas, encontré que datos antes de 2015 son muy escasos, esto podria servir para analisis de tendencia
        self.df = self.df[self.df['year_month'] >= '2015-08']
        
        # Features binarias
        self.df['shipping_is_free'] = self.df['shipping_is_free'].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0})
        self.df['shipping_admits_pickup'] = self.df['shipping_admits_pickup'].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0})
        
        if 'is_new' in self.df.columns:
            self.df['is_new'] = self.df['is_new'].fillna(0).astype(int)
        
        # Encodings
        self.df['province_enc'] = LabelEncoder().fit_transform(self.df['seller_province'])
        self.df['city_enc'] = LabelEncoder().fit_transform(self.df['seller_city'])
        self.df['category_encoded'] = LabelEncoder().fit_transform(self.df['category_id'])
        self.df['category_scaled'] = MinMaxScaler().fit_transform(self.df[['category_encoded']])
        
        # Ordinal encoding para seller_loyalty
        loyalty_order = [['bronze', 'gold_special', 'free', 'gold_pro', 'silver', 'gold', 'gold_premium']]
        self.df['loyalty_enc'] = OrdinalEncoder(categories=loyalty_order).fit_transform(self.df[['seller_loyalty']])
        
        # Features de texto
        self.df['title_length'] = self.df['title'].str.len()
        self.df['title_word_count'] = self.df['title'].fillna('').str.split().str.len()
        
        # Features de media
        self.df['qty_photos'] = self.df['pictures'].str.count("'id':").fillna(0).astype(int)
        
        # Limpiar columnas innecesarias
        self.df.drop(['warranty', 'sub_status'], axis=1, inplace=True, errors='ignore')
        
        # Reset index
        self.df.reset_index(drop=True, inplace=True)
        
        print(f"Features creadas exitosamente")
        
        return self.df
    
    def create_target(self) -> pd.DataFrame:
        # Crear variable objetivo
        self.df['has_sales'] = (self.df['sold_quantity'] > 0).astype(int)
        pct_with_sales = self.df['has_sales'].mean() * 100
        print(f"Target creado - Productos con ventas: {pct_with_sales:.1f}%")
        return self.df
    
    def select_features(self, feature_names: List[str] = None) -> List[str]:
        # Selecciono fueatures basadas en análisis de mi notebook, matriz de correlación y feature importance
        if feature_names is None:
            feature_names = [
                'log_price',
                'shipping_is_free',
                'is_new',
                'initial_quantity',
                'title_length',
                'qty_photos'
            ]
        
        self.feature_list = [f for f in feature_names if f in self.df.columns]
        print(f"Features seleccionadas: {len(self.feature_list)}")
        return self.feature_list
    
    def get_X_y(self, target_column: str = 'has_sales') -> tuple:
        # Separo features en X y target en y
        if not self.feature_list:
            raise ValueError("Ejecuta select_features() primero")
        
        X = self.df[self.feature_list].fillna(0)
        y = self.df[target_column]
        return X, y
    
    def get_processed_data(self) -> pd.DataFrame:
        # Retorno el dataset procesado
        return self.df