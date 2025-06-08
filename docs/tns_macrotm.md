# Sistema de Pensamiento: TNs y MacroTN

## Estructura Interna

- **Clases principales:**
  - `BaseThinkingNeuron`: Clase base para TNs.
  - Clases derivadas: TNs especializadas (emocional, filosófica, etc.).
  - `MacroTN`: Neurona de pensamiento macro.

- **Atributos principales:**
  - `id`, `tipo`, `punto_de_vista`
  - `historial_propuestas`
  - `relevancia`
  - `estado_interno`
  - `propuestas_generadas`

- **Métodos clave:**
  - `generar_propuesta(escenario, contexto)`: Genera propuesta desde un punto de vista.
  - `evaluar_propuesta(propuesta, contexto)`: Evalúa relevancia y adecuación.
  - `actualizar_estado(...)`: Ajusta relevancia y aprendizaje.
  - `MacroTN.seleccionar_respuesta(propuestas)`: Selecciona o combina propuestas.

## Función y Rol

Las TNs generan propuestas de respuesta desde distintos puntos de vista. MacroTN selecciona, combina o rechaza propuestas según contexto, intención y relevancia.

## Interacción

- Reciben escenarios del sintetizador.
- Interactúan con el razonador y la memoria de pensamiento.
- MacroTN coordina la selección final antes de pasar a consciencia.

## Ejemplo de Flujo

```python
tn_emocional = TNEmocional()
propuesta = tn_emocional.generar_propuesta(escenario, contexto)
macrotn = MacroTN()
respuesta_final = macrotn.seleccionar_respuesta([propuesta, ...])