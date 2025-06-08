# Sistema de IA Basado en Micro Neuronas (MNs)

## ---- ADVERTENCIA -----
El proyecto está en fases de desarrollo extremadamente tempranas (alfa), si eres un usuario que busca ejecutar esta IA, **NO LO HAGAS**, aún no está lista y podría incluso no funcionar, no se incluyó main.py ni otros archivos cruciales para su ejecución para así evitar que se intente.

## Notas del autor:
Con un diseño y arquitectura hechos desde 0, esta IA intenta prometer de verdad entender el lenguaje, su sistema basado en MNs le permite entender las reglas gramaticales, las palabras, los conceptos abstractos, entre varias otras cosas. 
¿Conoces la metafora del hombre que sabe como hablar chino pero no lo entiende? Pues nuestro sistema MNs, a diferencia del LLM, vendría siendo el chino que sabe de verdad chino, ya no más respuestas extrañas, ya no más inventar cosas, ¿Entrenamientos complejos? ¡Claro que no! Alimenta a Krystal-ai con un buen artículo cientifíco, lo entenderá todo y creará nuevas neuronas para conectar todo su nuevo conocimiento.

## Visión General

Este proyecto implementa un sistema de inteligencia artificial inspirado en una arquitectura neuronal jerárquica, no basada en LLMs, sino en micro neuronas (MNs), neuronas, macro neuronas y neuronas de interconexión. Cada nodo representa conceptos de distinta complejidad y se conecta para formar un sistema de razonamiento bruto, capaz de comprender, aprender y generar lenguaje de manera granular y explicable.

## Arquitectura

- **MicroNeuronas (MNs):** Unidades semánticas mínimas, cada una representa un concepto básico y contiene información gramatical, sinónimos, etc.
- **Neuronas:** Agrupan MNs para formar conceptos más complejos.
- **MacroNeuronas:** Integran varias neuronas para representar conceptos de alto nivel.
- **Neuronas de Interconexión:** Conectan nodos con características comunes (ej: todos los verbos).
- **Sistema de Embedding:** Permite similitud semántica mediante n-gramas y vectores.
- **Sintetizador de Escenarios:** Transforma la descomposición del input en escenarios semánticos.
- **Sistema de Pensamiento (TNs/MacroTN):** Genera y selecciona propuestas de respuesta desde distintos puntos de vista.
- **Módulo de Consciencia:** Supervisa el historial de pensamiento y genera la respuesta final, aplicando reglas gramaticales.
- **Aprendizaje y Dinámica de Conexiones:** El sistema aprende, ajusta relevancias y crea nuevos nodos/conexiones de forma dinámica.

## Flujo de Procesamiento

1. **Entrada del usuario:** Se descompone y activa MNs relevantes.
2. **Activación jerárquica:** MNs activan Neuronas, que pueden activar MacroNeuronas.
3. **Interconexión:** Neuronas de interconexión facilitan relaciones transversales.
4. **Sintetizador de escenarios:** Construye hipótesis de contexto a partir de activaciones y memorias.
5. **Pensamiento:** TNs generan propuestas de respuesta; MacroTN selecciona o combina.
6. **Consciencia:** Revisa historial, aplica razonamiento bruto y gramática, y genera la respuesta final.
7. **Aprendizaje:** El sistema ajusta pesos, refuerza o debilita nodos y crea nuevas conexiones según la experiencia.

## Documentación Detallada

La documentación técnica de cada módulo se encuentra en la carpeta `docs/`:

- [MicroNeurona](docs/micro_neurona.md)
- [Neurona](docs/neurona.md)
- [MacroNeurona](docs/macro_neurona.md)
- [Neurona de Interconexión](docs/neurona_interconectora.md)
- [Sistema de Embedding](docs/embedding.md)
- [Sintetizador de Escenarios](docs/sintetizador_contexto.md)
- [Sistema de Pensamiento (TNs/MacroTN)](docs/tns_macrotm.md)
- [Módulo de Consciencia](docs/consciencia.md)
- [Aprendizaje y Dinámica de Conexiones](docs/aprendizaje.md)


## Aprendizaje y Adaptabilidad

El sistema puede aprender nuevos conceptos, ajustar relevancias y crear nuevas conexiones dinámicamente, permitiendo una evolución continua y explicable.

---

Para detalles de implementación, consulte la documentación en `docs/`.
