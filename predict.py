import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.features import FeatureEngineer


def load_model(models_dir: str = './outputs'):
    # Cargar modelo entrenado
    with open(f'{models_dir}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde: {models_dir}/")
    return model


def predict(data_path: str, model_path: str = './outputs/best_model.pkl', output_path: str = 'predictions.csv'):
    print("="*60)
    print("PREDICCIONES")
    print("="*60)
    
    # Cargar datos
    df = pd.read_csv(data_path)
    print(f"\nDatos cargados: {len(df):,} registros")
    # Tomar una muestra fija de 500 registros se podria dejar aleatoria la cantidad de registros
    df = df.sample(n=500)
    print(f"\nSe usó una muestra de 500 registros")
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Modelo cargado")
    
    # Preparar features
    print("\nPreparando features...")
    engineer = FeatureEngineer(df)
    engineer.create_all_features()
    
    # Features usados en el modelo
    features = ['log_price', 'shipping_is_free', 'is_new', 'initial_quantity', 'title_length', 'qty_photos']
    X = engineer.df[features].fillna(0)
    
    # Predecir
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Agregar al DataFrame
    df['predicted_has_sales'] = predictions
    df['probability_sales'] = probabilities
    df['prediction_label'] = pd.Series(predictions).map({0: 'Sin Ventas', 1: 'Con Ventas'}).values
    
    # Guardar
    df.to_csv(output_path, index=False)
    
    # Resultados
    print("\n--- Resultados ---")
    print(f"Predicciones completadas: {len(predictions):,}")
    print(f"\nDistribución:")
    print(df['prediction_label'].value_counts())
    print(f"\nProbabilidad promedio: {probabilities.mean():.2%}")
    print(f"\nArchivo guardado: {output_path}")
    
    # Top 5 productos con mayor probabilidad
    print("\n--- Top 5 Productos (Mayor Probabilidad de venta) ---")
    top_5 = df.nlargest(5, 'probability_sales')[['title', 'price', 'probability_sales']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['title'][:50]}... (${row['price']:,.0f} - {row['probability_sales']:.1%})")
    print("\n" + "="*60)
    print("✓ Predicción completada")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Configuración de mi entrypoint para las predicciones
    DATA_PATH = 'outputs/processed_data.csv'
    OUTPUT_PATH = 'predictions_output.csv'
    
    # Ejecutar predicción y manejo de errores
    try:
        predictions = predict(DATA_PATH, output_path=OUTPUT_PATH)
    except FileNotFoundError:
        print("\n Error: Archivo no encontrado")
        print("Asegúrate de:")
        print("  1. Haber ejecutado 'python main.py' primero")
        print(f"  2. Que exista el archivo: {DATA_PATH}")