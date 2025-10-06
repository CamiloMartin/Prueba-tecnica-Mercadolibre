import warnings
warnings.filterwarnings('ignore')

from src.analyzer import DataAnalyzer
from src.features import FeatureEngineer
from src.models import ModelPredictor


def main():
    ############# Pipeline completo de análisis, ingeniería, entrenamiento y evaluación #############
    
    print("="*60)
    print("PIPELINE ML - PRUEBA TÉCNICA MERCADOLIBRE")
    print("="*60)
    
    # ANÁLISIS Y LIMPIEZA
    print("\n[1/4] Análisis y Limpieza de Datos")
    analyzer = DataAnalyzer()
    df = analyzer.read_dataset('data/new_items_dataset.csv')
    analyzer.get_summary()
    df_clean = analyzer.clean_data()
    
    # INGENIERÍA DE FEATURES
    print("\n[2/4] Ingeniería de Features")
    engineer = FeatureEngineer(df_clean)
    engineer.create_all_features()
    engineer.create_target()
    engineer.select_features()
    X, y = engineer.get_X_y()
    
    # ENTRENAMIENTO
    print("\n[3/4] Entrenamiento de Modelos")
    predictor = ModelPredictor(X, y, test_size=0.2)
    predictor.prepare_data(balance=True, strategy=0.80)
    predictor.train_models()
    predictor.cross_validate(cv=5)
    
    # EVALUACIÓN
    print("\n[4/4] Evaluación y Selección")
    best_model, best_name = predictor.select_best_model(metric='accuracy')
    predictor.show_classification_report()
    predictor.plot_feature_importance('feature_importance_modelo1.png')
    
    # RESUMEN FINAL
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    results = predictor.get_results_summary()
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  CV: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    # Exportar
    export_artifacts({
        'predictor': predictor,
        'engineer': engineer,
        'best_model': best_model
    })
    
    print("\n" + "="*60)
    print("✓ Pipeline completado exitosamente")
    print("="*60)
    
    return predictor, engineer


def export_artifacts(objects: dict, output_dir: str = './outputs'):
    # Exportar el módelo y artefactos generados para predicciones futuras
    import pickle
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Exportando Artefactos ---")
    
    with open(f'{output_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(objects['best_model'], f)
    
    with open(f'{output_dir}/predictor.pkl', 'wb') as f:
        pickle.dump(objects['predictor'], f)
    
    with open(f'{output_dir}/feature_engineer.pkl', 'wb') as f:
        pickle.dump(objects['engineer'], f)
    
    objects['engineer'].get_processed_data().to_csv(f'{output_dir}/processed_data.csv', index=False)
    
    print(f"Artefactos guardados en: {output_dir}/")


if __name__ == "__main__":
    predictor, engineer = main()