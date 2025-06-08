# Aprendizaje y Dinámica de Conexiones

## Estructura Jerárquica

El sistema está compuesto por MicroNeuronas, Neuronas, MacroNeuronas e Interconectoras, conectadas jerárquicamente y de forma transversal.

## Métodos y Procesos de Aprendizaje

- **Ajuste hebbiano de pesos:** Las neuronas ajustan sus pesos sinápticos según la co-activación de entradas.
- **Refuerzo y decaimiento contextual:** Los nodos refuerzan o debilitan su relevancia según el contexto y la frecuencia de activación.
- **Creación dinámica de nodos y enlaces:** El sistema puede crear nuevas MicroNeuronas, Neuronas, MacroNeuronas y conexiones cuando se le enseña un concepto nuevo o se detecta un patrón relevante.

## Integración con Otros Módulos

- El razonador coordina el aprendizaje y la actualización de pesos.
- La memoria almacena configuraciones aprendidas y permite la recuperación de patrones previos.
- Los módulos auxiliares (embedding, sintetizador, TNs) pueden disparar la creación de nuevas conexiones.

## Ejemplo de Flujo de Aprendizaje

```python
# Ajuste de pesos por co-activación
n.update_weights(input_activations, learning_rate=0.1)

# Refuerzo de relevancia
mn.aplicar_decaimiento(refuerzo=1.0, window=10)

# Creación dinámica de una nueva MicroNeurona
nueva_mn = MicroNeurona(id="mn_nueva", concepto="nuevo_concepto", tipo="sustantivo")
```

## Diagrama de Flujo

```mermaid
flowchart TD
    A[Entrada o Enseñanza Nueva] --> B[Creación de Nodo/Conexión]
    B --> C[Ajuste de Pesos y Relevancia]
    C --> D[Actualización de Memoria]
    D --> E[Integración en el Sistema]