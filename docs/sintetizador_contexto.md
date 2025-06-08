# Sintetizador de Escenarios

## Propósito y Rol General

El Sintetizador de Escenarios analiza el estado neuronal, memorias recuperadas y memoria de pensamiento para inferir y construir hipótesis de contexto ("escenarios") que representan la situación cognitiva actual. Transforma la "tabla bruta" de activaciones en una representación semántica estructurada.

## Estructura Interna

- **Clase principal:** `SintetizadorContexto`
- **Atributos:**
  - `razonador`: Referencia al objeto Razonador.
  - `sistema_neuronal`: (Opcional) Estructura bruta del sistema neuronal.
  - `interconectoras`: (Opcional) Neuronas de interconexión.

- **Métodos clave:**
  - `sintetizar(neural_state, retrieved_memories, num_iterations=5)`: Orquesta la síntesis de escenarios.
  - `_generar_hipotesis_iniciales(...)`: Construye hipótesis iniciales.
  - `_refinar_y_evaluar_hipotesis(...)`: Refina hipótesis.
  - `_inferir_escenario_desde_patrones(...)`: Inferencia semántica.

## Función y Flujo de Procesamiento

1. Recupera memoria de pensamiento.
2. Genera hipótesis iniciales a partir de activaciones y memorias.
3. Refina hipótesis iterativamente.
4. Retorna hipótesis de contexto estructuradas.

## Interacción con Otros Módulos

- Accede a neuronas y memorias a través del razonador.
- Considera activaciones y relaciones de nodos.
- Extrae neuronas clave de la memoria de pensamiento.

## Ejemplo de Uso

```python
sintetizador = SintetizadorContexto(razonador)
neural_state = {'neuronas': {'n1': 0.9}, 'micro_neuronas': {'mn1': 0.95}}
retrieved_memories = {'concepto_saludo': {'activation': 0.8}}
escenarios = sintetizador.sintetizar(neural_state, retrieved_memories, num_iterations=3)
for escenario in escenarios:
    print(escenario)