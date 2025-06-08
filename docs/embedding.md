# Sistema de Embedding y Similitud

## Estructura y Funcionamiento

- Los embeddings se generan principalmente mediante n-gramas y se representan como vectores de dimensión configurable (por defecto, 64).
- Cada MicroNeurona, Neurona, MacroNeurona e Interconectora posee su propio embedding, calculado a partir del concepto o relación que representa.
- Los embeddings se almacenan como atributos en cada nodo y pueden evolucionar con el aprendizaje.

## Métodos de Similitud

- **Similitud coseno:** Principal métrica para comparar embeddings y determinar activación o relevancia.
- **Métodos clave:**
  - `calcular_embedding(texto, dim)`: Genera el embedding de un texto/concepto.
  - `similitud_coseno(vec1, vec2)`: Calcula la similitud entre dos vectores.
  - `get_index_data()`: Exporta datos para índices vectoriales y búsquedas eficientes.

## Integración con el Sistema

- Los embeddings permiten la activación flexible de nodos por similitud, no solo por coincidencia exacta.
- Facilitan la transferencia de información y la generalización semántica entre conceptos relacionados.
- Se utilizan en la recuperación de memorias, la activación de patrones y la inferencia de escenarios.

## Ejemplo de Uso

```python
vec1 = mn.calcular_embedding("perro")
vec2 = mn.calcular_embedding("canino")
sim = mn.similitud_coseno(vec1, vec2)
if sim > 0.8:
    mn.activar([vec2], frase_original="canino")