# Prueba-tecnica-MELI
Este repo contiene el desarrollo del desafío técnico de MELI

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para predecir si un producto en MercadoLibre tendrá ventas basándose en sus características. El dataset contiene **100,000 registros** con 26 variables que describen productos publicados en la plataforma.

### Problema de Negocio

Se planteo como problema: Predecir qué productos tienen mayor probabilidad de vender lo cual permitiría:

- Optimizar estrategias de pricing y promoción
- Segmentación inteligente
- Identificar características clave de productos exitosos

## Estructura del Proyecto

```

MELI-prueba-tecnica/

data
    - new_items_dataset.csv    # Set de datos proporcionado
outputs
    - best_model.pkl           # Archivo pkl donde se aloja el modelo entrenado
    - feature_engineer.pkl     # Ingeniería de caracteristicas con los encoder
    - predictor.pkl            # Objeto predictor completo
    - processed_data.csv       # Datos procesados
src
    - analizer.py    # Clase para lectura y limpieza de datos
    - features.py    # Clase para ingenieria de caracteristicas y filtrado de datos
    - models.py       # Clase para entrenamiento y evaluación

main.py            # Script principal del pipeline
predict.py         # Script principal para hacer predicciones 
README.md          # Archivo readme del proyecto
requeriments.txt   # Archivo que contiene las dependencias

```

## Instalación y Configuración

### Requisitos

- Python 3.8+
- pip
- Homebrew (solo para Mac, para instalar libomp) esto lo requiere XGBoost

### Instalación

1. **Clonar el repositorio**
```bash
git clone 
```

2. **Crear entorno virtual**
```bash
python -m venv env
source env/bin/activate  # Comando en mac
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```
## Ejecución del Proyecto

### Entrenamiento Completo del Pipeline

Ejecuta el pipeline completo (limpieza, features, entrenamiento, evaluación):

Para realizar la ejecución completa del pipeline, se debe correr el archivo main.py desde la consola (python main.py) que se encuentra en la raiz del proyecto, se debe tener en cuenta que se debe tener activo el ambiente virtual y la instalación de las librerias requeridas. Al ejecutar el pipeline completo se generan los archivos correspondientes .pkl

**IMPORTANTE:** Se debe agregar el dataset (raw data) enviado para la prueba en la carpeta 'data' del proyecto con el nombre de 'new_items_dataset.csv', esta ruta y nombre de archivo es el configurado en el main del proyecto y lo paso como argumento a la clase de analyzer. 

### 2. Realizar Predicciones

Para realizar predicciones basta con correr desd econsola el archivo predict.py (python predict.py) el cual consume los archivos con el modelo entrenado alojados en la carpeta de outputs que se generan luego de la ejecución del pipeline

## Análisis de resultados

Para evaluar el desempeño del modelo de clasificación, se probaron dos algoritmos principales:

Random Forest Classifier
XGBoost Classifier

Ambos fueron entrenados bajo el mismo conjunto de features y evaluados mediante un esquema de validación cruzada de 5 pliegues, utilizando las métricas de Accuracy, ROC-AUC, y el reporte de clasificación (precision, recall, F1-score).

El pipeline se configuró para seleccionar el mejor modelo (Random Forest)  con mejor desempeño general entre precisión y estabilidad, sin embargo, se identificó en las revisiones de escritorio realizadas que XGBoost mostró un rendimiento muy cercano, destacando especialmente en la recuperación de la clase “Con Ventas” que seria la clase de mayor importancia para negocio, lo cual indica una buena capacidad para detectar productos con un mayor potencial comercial, aunque con un mayor sacrificio en falsos positivos.

**Resultados CV:**

RANDOM_FOREST:
  Accuracy: 0.7377 | ROC-AUC: 0.8121
  CV: 0.7316 (+/- 0.0035)

XGBOOST:
  Accuracy: 0.7243 | ROC-AUC: 0.8104
  CV: 0.7212 (+/- 0.0013)

Con lo anterior se logra evidenciar que el modelo seleccionado tiene una mejor precisión, una ligera mejora en la discriminación entre clases pero XGBoost es más estable y consistente de acuerdo al resultado de la validación cruzada

### Pasos previos

Se identificó en el set de datos que el percentil 99.99% daba como resultado al rededor de 6.500.000 unidades monetarias por lo que se tomó la decisión de filtrar los datos y trabajar unicamente con aquellos por debajo de 10.000.000 asumiendo que los datos para estos articulos costosos son reales

De la misma manera filtramos los datos en la cola inferior para dejar productos mayores a 5.0 unidades monetarias, el percentil (0.05) daba como resultado 2.0 encontrando en una revisión manual de los datos productos con precio de 1.0, productos de prueba y en un porcentaje menor productos de 4.0

Se realizó un filtrado de datos para tomar únicamente lo que está por encima de Agosto de 2015 lo anterior a esta fecha se consideró poco relevante por la cantidad y dispersión de los registros en el tiempo.

Se aplicó logaritmo al precio y precio base para suavizarlos y mejorar la visualización de la distribución de los datos.

Una variable que consideré importante para crear fue la de qué tan caro o barato es un producto en relación con el promedio de su propia categoría (price_category_ratio) aunque para este modelo en los resultados de la correlación e important features no fue relevante para la clasificación se podría usar para otro tipo de casos de uso, así como tratar de tener un feature de descuento ya que en los datos el base_price y price fuera de los anómalos que se quitaron (base_price < price), todos los registros tienen la misma cifra en ambas columnas, pero en un escenario diferente o con otras variables esto podría mostrar algún tipo de relación con variables objetivo para diferentes casos de uso.

## Insights para marketing y negocio

El modelo desarrollado permite predecir la probabilidad de que un producto nuevo publicado se venda o no, usando variables como el precio, condición del producto (nuevo o no), cantidad inicial, número de fotos y longitud del título.

El estado del producto y la cantidad inicial con la que se publica el articulo son los factores más importantes y relevantes en el éxito de ventas, seguidos por el precio y la calidad de la descripción (longitud del título).
Esto quiere decir que mejorar la presentación y disponibilidad de los productos puede impactar directamente en las conversiones.

Los productos con precios dentro del rango medio y títulos descriptivos tienen una mayor probabilidad de venderse, lo que indica que la optimización de contenido y pricing debe ser una prioridad.

### ¿Qué se puede hacer desde marketing?

Optimización de publicaciones:
El equipo de marketing podría usar el modelo para evaluar automáticamente la probabilidad de éxito de nuevas publicaciones antes de lanzarlas, sugiriendo ajustes de precio, fotos o descripciones.

Segmentación:
Identificar los productos con baja probabilidad de venta permite diseñar campañas específicas o estrategias de descuento dirigidas a esos segmentos.

Gestión de inventario:
Conocer qué productos tienen mayor potencial de rotación ayuda al equipo de negocio a planificar mejor el stock y la logística.

## Estrategia de monitoreo

Se podrían montar dos cosas: 
1. Una interfaz web para negocio con el fin de explorar el modelo visualmente, probar nuevos títulos o productos, ver predicciones al instante y que desde la visión de negocio se pueda interpretar a una escala pequeña si hay variaciones en los resultados obtenidos (Monitoreo con criterio de negocio).
2. Un pipeline de monitoreo que permita evaluar el modelo a lo largo del tiempo, para asegurar que las predicciones sigan siendo validas para negocio. Aqui es importante identificar cambios en la precisión del modelo, distribución de los datos o degradación para reentrenamiento, sin embargo, es necesario alinear con negocio umbrales para tomar decisiones por lo que es necesario desarrollar informes que muestren dicho rendimiento.

## Implementación técnica

Se suben nuevos productos a BigQuery.
Airflow lanza una tarea que entrena un nuevo modelo y lo registra en MLflow.
Se evalúa contra el modelo actual en producción.
Si mejora el accuracy o ROC-AUC se despliega automáticamente.
Un dashboard muestra la evolución del rendimiento semana a semana.

## Conclusiones

Este proyecto demuestra la viabilidad de aplicar Machine Learning para predecir el éxito comercial de productos en MercadoLibre, logrando un **accuracy del 73.77%** y un **ROC-AUC de 81.21%** con Random Forest como modelo óptimo.

### Hallazgos Clave:

1. **Factor determinante:** El estado del producto ('is_new') representa el 40% de la importancia predictiva, confirmando que productos nuevos tienen mayor probabilidad de venta.

2. **Stock inicial crítico:** La cantidad inicial ('initial_quantity') con 35% de importancia sugiere que vendedores con confianza en su inventario tienen mejor desempeño.

3. **Precio como modulador:** Aunque con impacto moderado (11%), el precio normalizado ('log_price') indica que existe un rango óptimo por categoría donde la conversión mejora significativamente.

4. **Calidad descriptiva:** La longitud del título (7%) y cantidad de fotos (5%) confirman que una presentación completa del producto mejora sustancialmente las probabilidades de venta.

### Valor de Negocio:

- **Capacidad predictiva:** El modelo detecta 74% de productos que sí venderán (recall), permitiendo acciones preventivas en inventario.

- **Optimización de recursos:** Identificar productos de bajo potencial de manera anticipada reduce esfuerzos innecesarios de marketing y otras áreas.

- **Escalabilidad:** Pipeline modularizado (analyzer, features, models) permite reentrenamiento automátizado y despliegue ágil.

### Limitaciones y Trabajo Futuro:

- **Precision clase positiva (69%):** 31% de falsos positivos requiere validación manual para decisiones críticas de alto costo.

- **Features no incluidas:** Loyalty del vendedor, competencia en categoría, limpieza del texto (stemming, eliminar stop wors, etc.) se realizó pero no se incluyeron, podria mejorar el resultado.

- **Desbalance temporal:** Dataset concentrado a partir de Agosto 2015 (ago, sep, oct); reentrenamiento periódico es crítico para capturar tendencias actuales.

- **Desbalance en precios:** Productos muy baratos y productos muy caros, entrenamiento separado para que el módelo infiera bien entre ambos mundos, si los productos caros son reales.

- **Generalización por categoría:** Un modelo por familia de productos podría mejorar precisión específica.

### Recomendaciones de Implementación:

1. **Fase piloto (Mes 1-2):** Desplegar en categoría controlada (ejemplo: celulares) con monitoreo constante.

2. **Testeo (Mes 3-4):** Comparar conversión entre productos con y sin optimización sugerida por modelo.

3. **Liberación incremental (Mes 5-6):** Expandir a más categorías con ajustes por aprendizajes.

4. **Monitoreo continuo:** Dashboard con métricas semanales y re-entrenamiento mensual automático.
