# MicroNeurona

## Estructura Interna

- **Atributos principales:**
  - `id`: Identificador único.
  - `concepto`: Concepto semántico representado.
  - `tipo`: Clasificación funcional.
  - `embedding`: Vector semántico (dim=64 por defecto).
  - `metadata`: Información adicional.
  - `activa`: Estado booleano de activación.
  - `activation_level`: Nivel continuo de activación.
  - `decay_rate`: Tasa de decaimiento.
  - `umbral_activacion`: Umbral para activación.
  - `historial_activacion`: Registro de activaciones.
  - `historial_embeddings`: Evolución de embeddings.
  - `mini_chain_of_thought`: Rastro de razonamiento local.
  - `memoria_episodica`: Memoria de eventos asociados.

- **Métodos clave:**
  - `normalizar(texto)`: Limpieza y normalización textual.
  - `calcular_embedding(texto, dim)`: Generación de embedding semántico.
  - `activar(vectores_entrada, frase_original, umbral, activation_fn)`: Activación flexible (por concepto o similitud).
  - `activar_async(...)`: Versión asíncrona.
  - `similitud_coseno(vec1, vec2)`: Métrica de similitud.
  - `aplicar_decaimiento(refuerzo, window)`: Decaimiento contextual.
  - `reset()`: Reinicio de estado.
  - `get_index_data()`: Exportación para índices vectoriales.

## Función y Rol

Unidad semántica mínima. Procesa conceptos y responde a entradas textuales o embeddings, propagando su activación a Neuronas superiores.

## Interacción

- Entrada básica para Neuronas.
- Puede ser activada por coincidencia textual directa o por similitud de embedding.
- Su activación puede influir en la activación de Neuronas y MacroNeuronas.

## Ejemplo de Flujo

```python
mn = MicroNeurona(id="mn1", concepto="perro", tipo="sustantivo")
entrada = [mn.calcular_embedding("canino")]
mn.activar(entrada, frase_original="El perro ladra")