# Tire Texture Analysis - Deep Learning Project

Proyecto desarrollado para la evaluación de maestría en IA. El objetivo es automatizar la detección de grietas estructurales en llantas mediante arquitecturas CNN y ResNet-50.

## Estructura del Proyecto
- `/notebooks`: Contiene el flujo de entrenamiento y evaluación (`Examen_Parcial.ipynb`).
- `/src`: Scripts de modelado y funciones de interpretabilidad (Grad-CAM).
- `/assets`: Visualizaciones de la matriz de confusión y mapas térmicos de fallos detectados.

## Resultados Clave
- **Arquitecturas:** Comparativa entre CNN propia (Baseline) vs. ResNet-50 (Transfer Learning).
- **Métricas:** AUC-ROC de 0.852.
- **Interpretabilidad:** Análisis Grad-CAM aplicado a errores críticos (Falsos Negativos).

## Análisis de fallos
Se identificó que el modelo presenta un sesgo hacia características estructurales dominantes (relieve y surcos), lo cual genera falsos negativos en presencia de grietas de bajo contraste.
