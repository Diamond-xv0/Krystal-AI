# Neurona de Interconexión

## Estructura Interna

- **Atributos principales:**
  - `id`
  - `neuronas_conectadas`: IDs de neuronas/conceptos enlazados.
  - `embedding`: Vector de relación.
  - `reglas`: Lógica de conexión adicional.

- **Métodos clave:**
  - `generar_embedding(dim)`: Embedding aleatorio de relación.
  - `es_relevante(concepto)`: Determina si conecta con un concepto dado.
  - `similitud_embedding(otro_embedding)`: Similitud coseno entre embeddings.

## Función y Rol

No participa directamente en el razonamiento, pero conecta conceptos y facilita la transferencia de información entre módulos o áreas semánticas.

## Interacción

- Permite relaciones transversales entre Neuronas y MacroNeuronas.
- Puede ser consultada para determinar relevancia o similitud de relaciones.

## Ejemplo de Flujo

```python
ni = NeuronaInterconectora(id="ni1", neuronas_conectadas=["n1", "macro1"])
ni.es_relevante("n1")  # True
sim = ni.similitud_embedding(otro_embedding)