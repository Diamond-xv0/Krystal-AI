from core.micro_neurona import MicroNeurona
from core.neurona import Neurona

def poblar_modelo_base(razonador, umbral_neurona=0.7):
    # =================================================================================
    # CAPA 1: MicroNeuronas (MNs) - MODELO UNIFICADO
    # =================================================================================
    
    # --- 1.1 Vocabulario Unificado de Palabras Clave ---
    # Se fusionan las MNs de comprensión y generativas en una sola fuente de verdad.
    # Cada palabra clave ahora tiene metadatos ricos (gramaticales, semánticos, etc.)
    # para potenciar tanto la comprensión como la generación.
    
    vocabulario_unificado = [
        # --- Conceptos de Saludo y Despedida ---
        ("mn_hola", "hola", {'semantic_field': 'saludo', 'personalidad': {'tono': 'casual', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'saludo', 'PERMITE_DESPUES': ['interrogativo_estado', 'fin_frase']}}),
        ("mn_buenos", "buenos", {'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_dias", "días", {'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_buenas", "buenas", {'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_tardes", "tardes", {'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_noches", "noches", {'GRAMATICA': {'TIPO': 'parte_saludo_formal'}}),
        ("mn_adios", "adiós", {'semantic_field': 'despedida', 'personalidad': {'tono': 'neutral', 'emocion': 'neutral'}, 'GRAMATICA': {'TIPO': 'despedida', 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_chao", "chao", {'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_informal'}}),
        ("mn_hasta_luego", "hasta luego", {'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_compuesta'}}),
        ("mn_nos_vemos", "nos vemos", {'semantic_field': 'despedida', 'GRAMATICA': {'TIPO': 'despedida_compuesta'}}),

        # --- Conceptos Interrogativos ---
        ("mn_que", "qué", {'semantic_field': 'pregunta_general', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'interrogativo_general', 'PERMITE_DESPUES': ['sustantivo', 'verbo']}}),
        ("mn_q", "q", {'semantic_field': 'pregunta_general', 'GRAMATICA': {'TIPO': 'interrogativo_informal'}}),
        ("mn_quien", "quién", {'semantic_field': 'pregunta_identidad', 'GRAMATICA': {'TIPO': 'interrogativo_persona'}}),
        ("mn_como", "cómo", {'semantic_field': 'pregunta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'interrogativo_estado', 'PERMITE_DESPUES': ['verbo_estado']}}),
        ("mn_cuando", "cuándo", {'semantic_field': 'pregunta_tiempo', 'GRAMATICA': {'TIPO': 'interrogativo_temporal'}}),
        ("mn_donde", "dónde", {'semantic_field': 'pregunta_lugar', 'GRAMATICA': {'TIPO': 'interrogativo_lugar'}}),
        ("mn_por_que", "por qué", {'semantic_field': 'pregunta_razon', 'GRAMATICA': {'TIPO': 'interrogativo_causal'}}),
        ("mn_cual", "cuál", {'semantic_field': 'pregunta_especifica', 'GRAMATICA': {'TIPO': 'interrogativo_seleccion'}}),
        ("mn_tal", "tal", {'semantic_field': 'pregunta_bienestar', 'GRAMATICA': {'TIPO': 'adverbio_interrogativo'}}),

        # --- Conceptos de Verbos Ser/Estar/Tener/Hacer/Poder/Querer ---
        ("mn_ser", "ser", {'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'ser'}}),
        ("mn_eres", "eres", {'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_es", "es", {'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_soy", "soy", {'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_estado', 'REQUIERE': ['pronombre'], 'PERMITE_DESPUES': ['sustantivo', 'adjetivo'], 'PROHIBE_DESPUES': ['verbo_estado']}}),
        ("mn_somos", "somos", {'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 1, 'PLURAL': True, 'TIEMPO': 'presente'}}),
        ("mn_estar", "estar", {'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'estar'}}),
        ("mn_estas", "estás", {'semantic_field': 'pregunta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_estado', 'REQUIERE': ['interrogativo_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("mn_esta", "está", {'GRAMATICA': {'TIPO': 'verbo_estado', 'PERSONA': 3, 'TIEMPO': 'presente'}}),
        ("mn_estoy", "estoy", {'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_estado', 'REQUIERE': ['pronombre'], 'PERMITE_DESPUES': ['adverbio_estado', 'adjetivo'], 'PROHIBE_DESPUES': ['verbo_estado']}}),
        ("mn_tener", "tener", {'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'tener'}}),
        ("mn_tienes", "tienes", {'GRAMATICA': {'TIPO': 'verbo_posesion', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_tengo", "tengo", {'GRAMATICA': {'TIPO': 'verbo_posesion', 'PERSONA': 1, 'TIEMPO': 'presente'}}),
        ("mn_hacer", "hacer", {'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'hacer'}}),
        ("mn_haces", "haces", {'GRAMATICA': {'TIPO': 'verbo_accion', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_hago", "hago", {'GRAMATICA': {'TIPO': 'verbo_accion', 'PERSONA': 1, 'TIEMPO': 'presente'}}),
        ("mn_poder", "poder", {'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'poder'}}),
        ("mn_puedes", "puedes", {'GRAMATICA': {'TIPO': 'verbo_modal', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_puedo", "puedo", {'semantic_field': 'auto_revelacion_capacidad', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'verbo_modal', 'PERMITE_DESPUES': ['verbo_infinitivo']}}),
        ("mn_querer", "querer", {'GRAMATICA': {'TIPO': 'verbo_infinitivo', 'RAIZ': 'querer'}}),
        ("mn_quieres", "quieres", {'GRAMATICA': {'TIPO': 'verbo_deseo', 'PERSONA': 2, 'TIEMPO': 'presente'}}),
        ("mn_quiero", "quiero", {'GRAMATICA': {'TIPO': 'verbo_deseo', 'PERSONA': 1, 'TIEMPO': 'presente'}}),

        # --- Conceptos de Afirmación/Negación y Cortesía ---
        ("mn_si", "sí", {'semantic_field': 'afirmacion', 'GRAMATICA': {'TIPO': 'afirmacion'}}),
        ("mn_no", "no", {'semantic_field': 'negacion', 'GRAMATICA': {'TIPO': 'negacion'}}),
        ("mn_claro", "claro", {'semantic_field': 'afirmacion', 'personalidad': {'tono': 'positivo', 'emocion': 'seguridad'}, 'GRAMATICA': {'TIPO': 'afirmacion', 'PERMITE_DESPUES': ['pronombre', 'fin_frase']}}),
        ("mn_ok", "ok", {'semantic_field': 'afirmacion', 'GRAMATICA': {'TIPO': 'afirmacion_informal'}}),
        ("mn_gracias", "gracias", {'semantic_field': 'agradecimiento', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia', 'PERMITE_DESPUES': ['fin_frase']}}),

        # --- Conceptos Generales y Entidades ---
        ("mn_nombre", "nombre", {'semantic_field': 'identidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_tiempo", "tiempo", {'semantic_field': 'temporalidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_clima", "clima", {'semantic_field': 'ambiente', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("mn_ayuda", "ayuda", {'semantic_field': 'asistencia', 'GRAMATICA': {'TIPO': 'sustantivo_o_verbo'}}),
        ("mn_tu", "tú", {'semantic_field': 'identidad_externa', 'GRAMATICA': {'TIPO': 'pronombre_personal'}}),
        ("mn_yo", "yo", {'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'pronombre', 'PERMITE_DESPUES': ['verbo_estado', 'verbo_accion']}}),
        
        # --- Conceptos que solo existían en el vocabulario generativo ---
        ("gen_y_tu", "y tú?", {'semantic_field': 'pregunta_reciproca', 'personalidad': {'tono': 'casual', 'emocion': 'curiosidad'}, 'GRAMATICA': {'TIPO': 'pregunta_reciproca', 'REQUIERE': ['adverbio_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_ayudarte", "ayudarte", {'semantic_field': 'asistencia', 'personalidad': {'tono': 'proactivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'verbo_infinitivo_complejo', 'REQUIERE': ['verbo_modal'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_analizar", "analizar", {'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_aprender", "aprender", {'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_procesar", "procesar", {'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'verbo_infinitivo'}}),
        ("gen_ia", "una IA", {'semantic_field': 'identidad_propia', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'sustantivo', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['adjetivo_calificativo', 'fin_frase']}}),
        ("gen_texto", "texto", {'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("gen_lenguaje", "lenguaje", {'semantic_field': 'auto_revelacion_capacidad', 'GRAMATICA': {'TIPO': 'sustantivo'}}),
        ("gen_bien", "bien", {'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_genial", "genial", {'semantic_field': 'respuesta_bienestar', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adverbio_estado', 'PERMITE_DESPUES': ['pregunta_reciproca', 'fin_frase']}}),
        ("gen_muy", "muy", {'semantic_field': 'intensificador', 'personalidad': {'tono': 'neutral'}, 'GRAMATICA': {'TIPO': 'adverbio_intensidad', 'REQUIERE': ['verbo_estado'], 'PERMITE_DESPUES': ['adverbio_estado', 'adjetivo']}}),
        ("gen_servicial", "servicial", {'semantic_field': 'rasgo_positivo', 'personalidad': {'tono': 'positivo', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'adjetivo_calificativo', 'REQUIERE': ['sustantivo', 'verbo_estado'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_de_nada", "de nada", {'semantic_field': 'respuesta_agradecimiento', 'personalidad': {'tono': 'amable', 'emocion': 'positiva'}, 'GRAMATICA': {'TIPO': 'cortesia_respuesta', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['fin_frase']}}),
        ("gen_pues", "pues", {'semantic_field': 'conector_discursivo', 'GRAMATICA': {'TIPO': 'conector', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['pronombre']}}),
        ("gen_en_que", "en qué", {'semantic_field': 'pregunta_especifica', 'GRAMATICA': {'TIPO': 'interrogativo_complejo', 'REQUIERE': ['inicio_frase'], 'PERMITE_DESPUES': ['verbo_modal']}}),
        ("gen_fin", "<FIN>", {'GRAMATICA': {'TIPO': 'fin_frase'}}),
    ]

    for mn_id, concepto, metadata in vocabulario_unificado:
        razonador.registrar_micro_neurona(MicroNeurona(mn_id, concepto, "palabra_clave", metadata=metadata))

    # --- 1.2 MNs Abstractas (Pensamiento) ---
    # Estas neuronas representan ideas y no palabras. No necesitan cambios.
    mn_conceptos = [
        ("concepto_saludo", "la idea de un saludo"), ("concepto_despedida", "la idea de una despedida"),
        ("concepto_pregunta_bienestar", "la idea de preguntar como esta alguien"),
        ("concepto_respuesta_bienestar", "la idea de afirmar el propio bienestar"),
        ("concepto_pregunta_reciproca", "la idea de devolver una pregunta social"),
        ("concepto_agradecimiento", "la idea de dar las gracias"), ("concepto_respuesta_agradecimiento", "la idea de responder a un gracias"),
        ("concepto_peticion_info_general", "la idea de pedir información general"), ("concepto_peticion_info_especifica", "la idea de pedir información sobre un tema"),
        ("concepto_ofrecer_ayuda", "la idea de ofrecer asistencia"), ("concepto_pedir_ayuda", "la idea de solicitar ayuda"),
        ("concepto_auto_revelacion_identidad", "la idea de revelar quién soy"), ("concepto_auto_revelacion_capacidad", "la idea de revelar qué puedo hacer"),
        ("concepto_acuerdo", "la idea de estar de acuerdo"), ("concepto_desacuerdo", "la idea de no estar de acuerdo"),
        ("concepto_empatia_positiva", "la idea de compartir la alegría de alguien"), ("concepto_empatia_negativa", "la idea de mostrar comprensión ante el malestar"),
        ("concepto_clarificacion", "la idea de pedir que se aclare algo"), ("concepto_iniciar_conversacion", "la idea de empezar a hablar proactivamente"),
    ]
    for mn_id, concepto in mn_conceptos:
        razonador.registrar_micro_neurona(MicroNeurona(mn_id, concepto, "concepto_abstracto"))

    # =================================================================================
    # CAPA 2: Neuronas (Ns)
    # =================================================================================
    # No se necesitan cambios aquí, ya que los IDs de las MNs de condición
    # (ej. "mn_hola") se han conservado como los canónicos.
    razonador.registrar_neurona(Neurona("n_patron_saludo_hola", "Patrón: Hola", ["mn_hola"]))
    razonador.registrar_neurona(Neurona("n_patron_saludo_formal", "Patrón: Buenos días/tardes/noches", ["mn_buenos", "mn_dias"]))
    razonador.registrar_neurona(Neurona("n_patron_despedida", "Patrón: Adiós/Chao/etc", ["mn_adios"]))
    razonador.registrar_neurona(Neurona("n_pregunta_como_estas", "Patrón: ¿Cómo estás?", ["mn_como", "mn_estas"], exclusiones_mn=["mn_no"]))
    razonador.registrar_neurona(Neurona("n_saludo_que_tal", "Patrón: ¿Qué tal?", ["mn_que", "mn_tal"]))
    razonador.registrar_neurona(Neurona("n_pregunta_quien_eres", "Patrón: ¿Quién eres?", ["mn_quien", "mn_eres"]))
    razonador.registrar_neurona(Neurona("n_pregunta_que_haces", "Patrón: ¿Qué haces?", ["mn_que", "mn_haces"], exclusiones_mn=["mn_tal"]))
    razonador.registrar_neurona(Neurona("n_peticion_ayuda", "Patrón: ¿Puedes ayudar?", ["mn_puedes", "mn_ayuda"]))
    razonador.registrar_neurona(Neurona("n_peticion_nombre", "Patrón: Pregunta por nombre", ["mn_cual", "mn_es", "mn_tu", "mn_nombre"]))
    razonador.registrar_neurona(Neurona("n_agradecimiento", "Patrón: Usuario da las gracias", ["mn_gracias"]))
    razonador.registrar_neurona(Neurona("n_afirmacion_positiva", "Patrón: Usuario afirma 'si/claro/ok'", ["mn_si"]))
    razonador.registrar_neurona(Neurona("n_negacion", "Patrón: Usuario dice 'no'", ["mn_no"]))
    razonador.registrar_neurona(Neurona("n_charla_sobre_clima", "Patrón: Usuario menciona el clima", ["mn_clima"]))

    # =================================================================================
    # CAPA 3: MacroNeuronas (MacroNs) - OBSOLETO
    # =================================================================================
    # Esta sección se mantiene obsoleta.
