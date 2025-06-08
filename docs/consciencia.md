# Módulo de Consciencia

## Estructura Interna

- **Clase principal:** `Conciencia`
- **Atributos principales:**
  - `historial_pensamiento`
  - `propuestas_recibidas`
  - `razonador`
  - `sistema_gramatical`
  - `respuesta_final`
  - `estado_interno`

- **Métodos clave:**
  - `procesar_propuestas(propuestas, historial)`: Recibe propuestas de MacroTN y revisa el historial.
  - `aplicar_razonamiento_bruto(...)`: Usa razonamiento bruto para validar o ajustar la respuesta.
  - `aplicar_gramatica(...)`: Ajusta la respuesta según reglas gramaticales.
  - `generar_respuesta_final()`: Produce la respuesta coherente y adecuada.

## Función y Rol

Recibe propuestas de respuesta, revisa el historial de pensamiento, utiliza razonamiento bruto y aplica reglas gramaticales para generar la respuesta final.

## Interacción

- Recibe propuestas de MacroTN.
- Consulta el historial de pensamiento.
- Interactúa con el razonador y el sistema gramatical.
- Devuelve la respuesta final al usuario o sistema externo.

## Ejemplo de Flujo

```python
consciencia = Conciencia()
consciencia.procesar_propuestas(propuestas, historial)
consciencia.aplicar_razonamiento_bruto()
consciencia.aplicar_gramatica()
respuesta = consciencia.generar_respuesta_final()