import pandas as pd

class DataAnalyzer:
    def __init__(self, filepath: str = None):
        self.df = None
        self.filepath = filepath
        self.initial_rows = 0
        
    def read_dataset(self, filepath: str = None) -> pd.DataFrame:
        # Lectura de datos
        if filepath:
            self.filepath = filepath
        
        self.df = pd.read_csv(self.filepath)
        self.initial_rows = len(self.df)
        print(f"Dataset cargado: {len(self.df):,} filas, {len(self.df.columns)} columnas")
        return self.df
    
    def get_summary(self) -> None:
        # Resumen estadístico básico
        print("\n--- Resumen Estadístico ---")
        print(f"Precio - Media: ${self.df['price'].mean():,.2f} | Mediana: ${self.df['price'].median():,.2f}")
        print(f"Precio - Q99: ${self.df['price'].quantile(0.99):,.2f}")
        print(f"Ventas - Media: {self.df['sold_quantity'].mean():.1f} | Mediana: {self.df['sold_quantity'].median():.1f}")
    
    def clean_data(self) -> pd.DataFrame:
        # Limpieza de datos
        print("\n--- Limpieza de Datos ---")
        # Eliminar nulos en columnas que considero críticas
        self.df.dropna(subset=['id', 'title', 'category_id', 'seller_id'], inplace=True)
        # Imputar valores faltantes
        self.df.fillna({
            'seller_country': 'Argentina',
            'seller_province': 'Capital Federal',
            'seller_city': 'CABA'
        }, inplace=True)
        # Filtros de calidad de datos
        rows_before = len(self.df)
        self.df = self.df[self.df['base_price'] >= self.df['price']]
        self.df = self.df[self.df['price'] < 10000000.0]
        self.df = self.df[self.df['price'] > 5.0]
        print(self.df.head(5))

        return self.df
    
    def get_clean_data(self) -> pd.DataFrame:
        # Retornar el df
        return self.df