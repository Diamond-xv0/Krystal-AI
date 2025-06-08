# Neurona

## Estructura Interna

- **Atributos principales:**
  - `id`, `nombre`
  - `condiciones_mn`: IDs de MicroNeuronas de entrada.
  - `exclusiones_mn`: MNs que inhiben la activación.
  - `umbral_activacion`
  - `metadata`
  - `activa`, `activation_level`
  - `decay_rate`
  - `historial_activacion`, `historial_pesos`
  - `weights`: Pesos sinápticos para cada MN de entrada.
  - `embedding`: Vector semántico del nombre.

- **Métodos clave:**
  - `evaluar(input_activations, umbral, activation_fn, micro_neuronas_dict, attention_window)`: Evalúa activación según entradas y pesos.
  - `update_weights(input_activations, learning_rate)`: Aprendizaje hebbiano.
  - `aplicar_decaimiento(refuerzo, window)`: Decaimiento contextual.
  - `reset()`: Reinicio de estado.

## Función y Rol

Integra patrones de activación de MicroNeuronas, aplicando pesos y funciones de activación no lineales. Puede ser inhibida por exclusiones.

## Interacción

- Recibe activaciones de MNs.
- Puede ser excluida por ciertas MNs.
- Propaga su activación a MacroNeuronas.
- Ajusta pesos según co-activación (aprendizaje).

## Ejemplo de Flujo

```python
n = Neurona(id="n1", nombre="animal", condiciones_mn=["mn1", "mn2"])
input_activations = {"mn1": 1.0, "mn2": 0.8}
n.evaluar(input_activations)
n.update_weights(input_activations)