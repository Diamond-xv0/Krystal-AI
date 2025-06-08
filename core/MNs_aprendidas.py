from core.micro_neurona import MicroNeurona

neuronas_aprendidas = []

def poblar_neuronas_aprendidas():
    # Ejemplo 1: Neurona aprendida para patrón "qué tal"
    neuronas_aprendidas.append(MicroNeurona(
        id="n_patron_que_tal",
        concepto="qué tal",
        tipo="patron",
        embedding=None,  # Se puede recalcular al cargar
        metadata={
            "secuencia": ["que", "tal"],
            "regex": r"que tal",
            "activacion": {
                "inicio_frase": True,
                "umbral": 0.95,
                "contextos": ["conversacion", "saludo"]
            },
            "significados": {
                "default": ["saludo_casual"],
                "con_calor": ["comentario_clima"]
            },
            "macro_tags": ["macro_saludo", "macro_patron"],
            "campos_semanticos": ["saludo", "social", "patron"],
            "es_saludo": 1,
            "es_pregunta": 0,
            "es_macro_de": ["macro_saludo"],
            "sinonimos": ["mn_que_hay"],
            "variantes": ["mn_que_tal_amigo"],
            "personalidad": {
                "tono": "amigable",
                "emocion": "positiva",
                "registro": "informal"
            },
            "explicacion": "Secuencia usada para saludar de manera casual.",
            "ejemplos": ["¡Qué tal!", "¡Hola, qué tal?"],
            "origen": "enseñada_usuario"
        }
    ))

    # Ejemplo 2: Neurona aprendida para patrón "hace calor"
    neuronas_aprendidas.append(MicroNeurona(
        id="n_patron_hace_calor",
        concepto="hace calor",
        tipo="patron",
        embedding=None,
        metadata={
            "secuencia": ["hace", "calor"],
            "regex": r"hace calor",
            "activacion": {
                "umbral": 0.9,
                "contextos": ["clima", "conversacion"]
            },
            "significados": {
                "default": ["comentario_clima"],
                "con_pregunta": ["pregunta_estado"]
            },
            "macro_tags": ["macro_clima", "macro_patron"],
            "campos_semanticos": ["clima", "comentario", "patron"],
            "es_saludo": 0,
            "es_pregunta": 0,
            "es_macro_de": ["macro_clima"],
            "sinonimos": ["mn_que_calor"],
            "variantes": [],
            "personalidad": {
                "tono": "neutral",
                "emocion": "neutra",
                "registro": "informal"
            },
            "explicacion": "Secuencia usada para comentar sobre el clima.",
            "ejemplos": ["¡Hace calor!", "Hoy hace calor."],
            "origen": "autoaprendida"
        }
    ))

    return neuronas_aprendidas