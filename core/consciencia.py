import math
from core.micro_neurona import MicroNeurona

class Conciencia:
    def __init__(self, personalidad, contexto, plan_ganador, adjudicator, razonador, sistema_neuronal=None, interconectoras=None):
        self.personalidad = personalidad
        self.contexto = contexto
        self.plan_ganador = plan_ganador
        self.adjudicator = adjudicator
        self.razonador = razonador
        self.sistema_neuronal = sistema_neuronal
        self.interconectoras = interconectoras
        
        # Obtenemos la referencia al vocabulario unificado del razonador.
        # Esta es ahora la única fuente de verdad para todas las palabras clave.
        self.vocabulario = self.razonador.vocabulario_palabras_clave

    def _calcular_vector_promedio(self, vectores):
        """Calcula el promedio de una lista de vectores sin dependencias externas."""
        if not vectores: return []
        dim = len(vectores[0])
        if not all(len(v) == dim for v in vectores): return []
        meta_vector = [0.0] * dim
        for v in vectores:
            if not v: continue
            for j in range(dim):
                meta_vector[j] += v[j]
        num_vectores = len(vectores)
        return [x / num_vectores for x in meta_vector] if num_vectores > 0 else []

    def _sintetizar_respuesta_emergente(self, plan):
        """
        Sintetiza una respuesta usando la puntuación híbrida (vectorial + semántica + macro-neuronal + tela de araña)
        para una mayor precisión y naturalidad.
        """
        print("\n[DEBUG-CONCIENCIA] Iniciando SÍNTESIS HÍBRIDA (Vectorial + Semántica + MacroNeurona + Interconectoras).")
        plan_conceptual = plan.get("plan_conceptual")
        
        # Ajuste de relevancia usando interconectoras (tela de araña)
        interconectoras = getattr(self.razonador, "interconectoras", {})
        if plan_conceptual and interconectoras:
            fuerza_total = 0.0
            count = 0
            for i in range(len(plan_conceptual)):
                for j in range(i+1, len(plan_conceptual)):
                    for inter in interconectoras.values():
                        if inter.es_relevante(plan_conceptual[i]) and inter.es_relevante(plan_conceptual[j]):
                            emb_i = getattr(self.razonador.micro_neuronas.get(plan_conceptual[i], None), "embedding", None)
                            fuerza = inter.similitud_embedding(emb_i) if emb_i is not None else 0.0
                            fuerza_total += fuerza
                            count += 1
            if count > 0:
                print(f"[DEBUG-CONCIENCIA] Boost de relevancia por interconectoras: {fuerza_total/count:.3f}")
                # Aquí podrías usar este boost para ajustar la puntuación de selección o la confianza global

        if not plan_conceptual:
            print("[DEBUG-CONCIENCIA] ERROR: El plan no contenía un 'plan_conceptual'.")
            return "Mi pensamiento no está claro."

        # Get embeddings for concepts in the plan and key neurons from thinking memory (from context)
        all_concept_embeddings = []
        for id_c in plan_conceptual:
            if id_c in self.razonador.micro_neuronas:
                all_concept_embeddings.append(self.razonador.micro_neuronas[id_c].embedding)

        # Retrieve thinking memory content directly from the Razonador's memory instance
        thinking_memory_content = self.razonador.memoria.thinking_memory.retrieve()
        print(f"[DEBUG-CONCIENCIA] Retrieved from thinking memory: {thinking_memory_content}")

        # Add embeddings for thinking memory elements
        if thinking_memory_content:
             print(f"[DEBUG-CONCIENCIA] Using thinking memory elements for synthesis: {[item.get('id') for item in thinking_memory_content if isinstance(item, dict)]}")
             for element_info in thinking_memory_content:
                 if isinstance(element_info, dict) and element_info.get('id') in self.razonador.micro_neuronas:
                     neuron_id = element_info['id']
                     # We could potentially weight the embedding by initial_activation here if needed
                     all_concept_embeddings.append(self.razonador.micro_neuronas[neuron_id].embedding)
        else:
             print("[DEBUG-CONCIENCIA] No thinking memory elements found for synthesis.")

        # Calculate the conceptual guidance vector including thinking memory embeddings
        vector_guia_conceptual = self._calcular_vector_promedio(all_concept_embeddings)

        campos_semanticos_objetivo = set()
        for id_concepto in plan_conceptual:
            partes = id_concepto.split('_', 1)
            if len(partes) > 1:
                campos_semanticos_objetivo.add(partes[1])
        print(f"[DEBUG-CONCIENCIA] Plan a ejecutar: {' -> '.join(plan_conceptual)}")
        print(f"[DEBUG-CONCIENCIA] Campos semánticos objetivo: {campos_semanticos_objetivo}")
        
        secuencia_mns_elegidas = []
        # Ajustamos los pesos para dar cabida a la nueva dimensión de personalidad.
        PESO_VECTORIAL = 0.2
        PESO_SEMANTICO = 0.5
        PESO_PERSONALIDAD = 0.3 # La personalidad ahora influye en la elección de palabras.
        
        humor_actual = self.personalidad.estado_emocional['humor']
        print(f"[DEBUG-CONCIENCIA] Humor actual: {humor_actual}")

        # --- MacroNeuronas activas ---
        macro_activas = [m for m in self.razonador.macro_neuronas.values() if getattr(m, "activa", False)]
        macro_ids = [m.id for m in macro_activas]

        for i in range(15):
            mn_anterior = secuencia_mns_elegidas[-1] if secuencia_mns_elegidas else None
            palabras_validas = self.adjudicator.get_valid_next_words(self.vocabulario, mn_anterior, macro_activas=macro_activas)
            
            if not palabras_validas: break

            candidatas_puntuadas = []
            for mn_candidata in palabras_validas:
                if mn_candidata.metadata.get('GRAMATICA', {}).get('TIPO') == 'fin_frase':
                    # Give a low score to 'fin_frase' unless the sentence has some length
                    fin_frase_score = 0.0
                    if len(secuencia_mns_elegidas) >= 2: # Allow ending after at least 2 words
                         fin_frase_score = 0.1 # Slightly higher score to allow ending
                    candidatas_puntuadas.append({"mn": mn_candidata, "score": fin_frase_score, "debug": "FIN"})
                    continue

                # --- Puntuación Lógica y Semántica (como antes) ---
                score_vectorial = MicroNeurona.similitud_coseno(vector_guia_conceptual, mn_candidata.embedding)
                score_semantico = 0.0
                campo_candidata = mn_candidata.metadata.get('semantic_field')
                if campo_candidata:
                    score_semantico = 0.2 # Base score for having a semantic field
                    if campo_candidata in campos_semanticos_objetivo:
                        score_semantico = 1.0 # Full score for matching a target field
                
                # --- Nueva Puntuación de Personalidad ---
                score_personalidad = 0.0
                meta_pers = mn_candidata.metadata.get('personalidad', {})
                tono_palabra = meta_pers.get('tono')
                emocion_palabra = meta_pers.get('emocion')
                
                # Bonificación si la emoción de la palabra coincide con el humor
                if emocion_palabra and emocion_palabra in humor_actual:
                    score_personalidad += 0.5
                # Bonificación si el tono de la palabra coincide con algún rasgo
                if tono_palabra and any(tono_palabra in rasgo for rasgo in self.personalidad.rasgos.values()):
                     score_personalidad += 0.5

                # --- Boost for Clarification Plan ---
                clarification_boost = 0.0
                if 'concepto_clarificacion' in plan_conceptual:
                    # Check if the candidate micro-neuron is related to clarification
                    # This could be based on semantic field or specific IDs
                    campo_candidata = mn_candidata.metadata.get('semantic_field')
                    # Add specific micro-neuron IDs known to be clarification words
                    clarification_mn_ids = ['mn_que', 'mn_cual', 'mn_como', 'mn_por_que', 'mn_aclarar', 'mn_especificar', 'mn_entender'] # Add more as needed
                    if campo_candidata == 'clarificacion' or mn_candidata.id in clarification_mn_ids:
                        clarification_boost = 0.5 # Apply a significant boost

                # --- Puntuación Final Ponderada ---
                # --- MacroNeurona boost ---
                macro_boost = 0.0
                macro_tags = mn_candidata.metadata.get('macro_tags', [])
                if macro_ids and any(mid in macro_tags for mid in macro_ids):
                    macro_boost = 0.5 # Prioriza palabras alineadas con macro activas

                puntaje_total = (PESO_VECTORIAL * score_vectorial) + \
                                (PESO_SEMANTICO * score_semantico) + \
                                (PESO_PERSONALIDAD * score_personalidad) + \
                                clarification_boost + macro_boost

                debug_info = f"V:{score_vectorial:.4f} S:{score_semantico:.4f} P:{score_personalidad:.4f} B:{clarification_boost:.4f} M:{macro_boost:.4f}"
                print(f"[DEBUG-CONCIENCIA] Candidate '{mn_candidata.concepto}': Total:{puntaje_total:.4f} ({debug_info})")
                if puntaje_total > 0:
                    candidatas_puntuadas.append({"mn": mn_candidata, "score": puntaje_total, "debug": debug_info})

            if not candidatas_puntuadas: break

            candidatas_puntuadas.sort(key=lambda x: x['score'], reverse=True)
            print(f"[DEBUG-CONCIENCIA] PASO {i+1} - Top 3 candidatas (sorted): {[{c['mn'].concepto: f'Total:{c['score']:.4f} ({c['debug']})'} for c in candidatas_puntuadas[:3]]}")

            mejor_candidata = candidatas_puntuadas[0]
            mn_elegida = mejor_candidata['mn']
            secuencia_mns_elegidas.append(mn_elegida)

            if mn_elegida.metadata.get('GRAMATICA', {}).get('TIPO') == 'fin_frase': break

        palabras = [mn.concepto for mn in secuencia_mns_elegidas if mn.concepto != '<FIN>']
        respuesta = " ".join(palabras).capitalize()
        if respuesta and respuesta[-1] not in ".!?":
             respuesta += "."

        return respuesta if respuesta else "Me he quedado sin palabras."

    def construir_respuesta(self):
        print("\n[PENSANDO...]")
        if not self.plan_ganador:
            return "No estoy segura de qué decir."
        return self._sintetizar_respuesta_emergente(self.plan_ganador)
