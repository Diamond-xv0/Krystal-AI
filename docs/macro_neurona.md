# MacroNeurona

## Estructura Interna

- **Atributos principales:**
  - `id`, `nombre`
  - `condiciones_n`: IDs de Neuronas de entrada.
  - `exclusiones_mn`: MNs que pueden inhibirla.
  - `umbral_activacion`
  - `metadata`
  - `activa`
  - `historial_activacion`
  - `embedding`

- **Métodos clave:**
  - `evaluar(ns_activas_ids, mns_activas_ids)`: Evalúa activación según Neuronas activas y exclusiones.
  - `reset()`: Reinicio de estado.

## Función y Rol

Agrega patrones complejos, integrando múltiples Neuronas. Permite la representación de conceptos de alto nivel.

## Interacción

- Se activa si suficientes Neuronas de entrada están activas y no es excluida por MNs.
- Propaga patrones a niveles superiores o módulos externos.

## Ejemplo de Flujo

```python
macro = MacroNeurona(id="macro1", nombre="mamífero", condiciones_n=["n1", "n2"])
macro.evaluar(ns_activas_ids=["n1"], mns_activas_ids=[])