import collections

class SintetizadorContexto:
    """
    Analiza la tabla bruta de activaciones neuronales para inferir y construir
    dinámicamente un objeto de 'escenario' que representa la situación actual.
    """
    def __init__(self, razonador, sistema_neuronal=None, interconectoras=None):
        """
        Inicializa el sintetizador con referencias al razonador, sistema neuronal bruto e interconectoras.
        """
        self.razonador = razonador
        self.sistema_neuronal = sistema_neuronal
        self.interconectoras = interconectoras

    def _inferir_escenario_desde_patrones(self, patrones_activos, conceptos_activos):
        """
        Infiere un escenario de manera más abstracta y compleja basada en todos los elementos activos.
        """
        active_elements = []
        for pattern_id, confidence in patrones_activos.items():
            active_elements.append({'id': pattern_id, 'tipo': 'patron', 'confianza': confidence})

        for concept_id, confidence in conceptos_activos.items():
            active_elements.append({'id': concept_id, 'tipo': 'concepto', 'confianza': confidence})

        if not active_elements:
            return None # No active elements, no scenario

        # Sort active elements by confidence
        sorted_active_elements = sorted(active_elements, key=lambda x: x['confianza'], reverse=True)

        # Basic synthesis: include all active elements and infer a general type
        # This is a placeholder for more complex synthesis logic
        inferred_tipo = 'general'
        inferred_subtipo = 'elementos_activos'
        inferred_confianza = sorted_active_elements[0]['confianza'] if sorted_active_elements else 0.0
        implicaciones = ['requiere_analisis_profundo'] # Indicates a complex scenario needing more processing

        # Example of slightly more complex inference: if a high-confidence social pattern is active
        if sorted_active_elements and sorted_active_elements[0]['tipo'] == 'patron' and sorted_active_elements[0]['confianza'] > 0.8:
             # This is a simplification; real logic would check pattern type metadata
             if 'saludo' in sorted_active_elements[0]['id'] or 'pregunta' in sorted_active_elements[0]['id']: # Basic keyword check
                 inferred_tipo = 'interaccion_social'
                 inferred_subtipo = sorted_active_elements[0]['id'] # Use the pattern ID as subtype for now
                 implicaciones.extend(['requiere_respuesta_contextual'])


        return {
            'tipo': inferred_tipo,
            'subtipo': inferred_subtipo,
            'confianza': inferred_confianza,
            'elementos_activos': sorted_active_elements, # List all active elements with details
            'implicaciones': implicaciones
        }

    def sintetizar(self, neural_state, retrieved_memories, num_iterations=5):
        """
        Sintetiza múltiples hipótesis de contexto basadas en el estado neuronal completo,
        las memorias recuperadas y la thinking memory, refinándolas iterativamente.
        """
        # Retrieve information from thinking memory
        # Retrieve information from thinking memory directly from the Razonador's memory instance
        thinking_memory_content = self.razonador.memoria.thinking_memory.retrieve()
        print(f"DEBUG: SintetizadorContexto - Retrieved from thinking memory: {thinking_memory_content}")

        # 1. Generar hipótesis iniciales
        context_hypotheses = self._generar_hipotesis_iniciales(neural_state, retrieved_memories, thinking_memory_content)

        # 2. Refinar y evaluar hipótesis iterativamente
        for i in range(num_iterations):
            print(f"DEBUG: SintetizadorContexto - Refinando hipótesis, iteración {i+1}/{num_iterations}")
            context_hypotheses = self._refinar_y_evaluar_hipotesis(context_hypotheses, neural_state, retrieved_memories, thinking_memory_content)
            # Optional: Add a stopping condition if hypotheses converge

        print(f"DEBUG: SintetizadorContexto - Síntesis de contexto finalizada. {len(context_hypotheses)} hipótesis generadas.")
        return context_hypotheses # Return the final list of hypotheses

    def _generar_hipotesis_iniciales(self, neural_state, retrieved_memories, thinking_memory_content, activation_threshold=0.6):
        """
        Genera un conjunto inicial de hipótesis de contexto basado en el estado neuronal,
        las memorias recuperadas, la thinking memory, MacroNeuronas activas y patrones/hubs aprendidos.
        """
        print("DEBUG: SintetizadorContexto - Generando hipótesis iniciales.")
        hypotheses = []
        active_neurons = {n_id: level for n_id, level in neural_state.get('neuronas', {}).items() if level > activation_threshold}
        active_micro_neurons = {mn_id: level for mn_id, level in neural_state.get('micro_neuronas', {}).items() if level > activation_threshold}

        # --- MacroNeuronas activas ---
        macro_activas = [m for m in self.razonador.macro_neuronas.values() if getattr(m, "activa", False)]
        if macro_activas:
            for macro in macro_activas:
                hypotheses.append({
                    'tipo': 'macro_neurona_focus',
                    'descripcion': f'Contexto dominado por MacroNeurona: {macro.id}',
                    'confianza': 1.0,
                    'elementos_clave': [macro.id],
                    'soporte_evidencia': {'macro': [macro.id]}
                })

        # --- Patrones y hubs aprendidos activos ---
        patrones_activos = []
        hubs_activos = []
        for mn_id in active_micro_neurons:
            mn = self.razonador.micro_neuronas.get(mn_id)
            if mn and mn.tipo == "patron":
                patrones_activos.append(mn)
            if mn and "hub" in mn.tipo:
                hubs_activos.append(mn)
        # Hipótesis por patrones aprendidos
        for patron in patrones_activos:
            hypotheses.append({
                'tipo': 'patron_aprendido',
                'descripcion': f'Patrón aprendido detectado: {patron.concepto}',
                'confianza': active_micro_neurons[patron.id],
                'elementos_clave': [patron.id],
                'soporte_evidencia': {'patron': [patron.id]},
                'explicacion': patron.metadata.get("explicacion", "")
            })
        # Hipótesis por hubs activos
        for hub in hubs_activos:
            hypotheses.append({
                'tipo': 'hub_activo',
                'descripcion': f'Hub semántico activo: {hub.concepto}',
                'confianza': active_micro_neurons[hub.id],
                'elementos_clave': [hub.id],
                'soporte_evidencia': {'hub': [hub.id]},
                'explicacion': hub.metadata.get("explicacion", "")
            })

        # Hypothesis 1: Based on highly active Neuronas
        if active_neurons:
            main_neuron_id = max(active_neurons, key=active_neurons.get) # Get the most active neuron
            hypotheses.append({
                'tipo': 'neuronal_focus',
                'descripcion': f'Contexto centrado en la actividad de la neurona {main_neuron_id}.',
                'confianza': active_neurons[main_neuron_id],
                'elementos_clave': [main_neuron_id],
                'soporte_evidencia': {'neurons': [main_neuron_id]}
            })

        # Hypothesis 2: Based on highly relevant retrieved memories (associative)
        if retrieved_memories:
            main_memory_concept = max(retrieved_memories, key=lambda k: retrieved_memories[k].get('activation', 0.0))
            hypotheses.append({
                'tipo': 'memory_focus_associative',
                'descripcion': f'Contexto influenciado por la memoria asociativa clave: {main_memory_concept}.',
                'confianza': retrieved_memories[main_memory_concept].get('activation', 0.0),
                'elementos_clave': [main_memory_concept],
                'soporte_evidencia': {'memories': [main_memory_concept]}
            })

        # Hypothesis 3: Based on key neurons from Thinking Memory
        if thinking_memory_content:
            for key_neuron_info in thinking_memory_content:
                hypotheses.append({
                    'tipo': 'thinking_memory_focus',
                    'descripcion': f'Contexto centrado en la neurona clave de pensamiento: {key_neuron_info.get("id", "unknown")}.',
                    'confianza': key_neuron_info.get('initial_activation', 0.5),
                    'elementos_clave': [key_neuron_info.get('id')],
                    'soporte_evidencia': {'thinking_neurons': [key_neuron_info.get('id')]}
                })

        # Hypothesis 4: Combination of active neurons and retrieved memories (placeholder logic)
        if active_neurons and retrieved_memories:
            combined_elements = []
            for n_id in active_neurons:
                neuron = self.razonador.neuronas.get(n_id)
                if neuron:
                    linked_concept = neuron.metadata.get('memory_concept_id')
                    if linked_concept and linked_concept in retrieved_memories:
                        combined_elements.append(n_id)
                        combined_elements.append(linked_concept)

            if combined_elements:
                hypotheses.append({
                    'tipo': 'combined_focus_associative',
                    'descripcion': 'Contexto emergente de la interacción neuronal y memoria asociativa.',
                    'confianza': 0.7,
                    'elementos_clave': list(set(combined_elements)),
                    'soporte_evidencia': {'neurons': [e for e in combined_elements if e in active_neurons], 'memories': [e for e in combined_elements if e in retrieved_memories]}
                })

        # Hypothesis 5: Combination of active neurons and thinking memory neurons (placeholder logic)
        if active_neurons and thinking_memory_content:
            combined_elements = []
            thinking_neuron_ids = [item.get('id') for item in thinking_memory_content if item.get('id')]
            for n_id in active_neurons:
                if n_id in thinking_neuron_ids:
                    combined_elements.append(n_id)

            if combined_elements:
                hypotheses.append({
                    'tipo': 'combined_focus_thinking',
                    'descripcion': 'Contexto emergente de la interacción neuronal y thinking memory.',
                    'confianza': 0.8,
                    'elementos_clave': list(set(combined_elements)),
                    'soporte_evidencia': {'neurons': [e for e in combined_elements if e in active_neurons], 'thinking_neurons': combined_elements}
                })

        # Ensure unique hypotheses (basic check based on type and main element)
        unique_hypotheses = []
        seen_identifiers = set()
        for hypo in hypotheses:
            identifier = (hypo['tipo'], tuple(sorted(str(e) for e in hypo['elementos_clave'])))
            if identifier not in seen_identifiers:
                unique_hypotheses.append(hypo)
                seen_identifiers.add(identifier)

        # Ajuste de confianza usando interconectoras (tela de araña)
        interconectoras = getattr(self.razonador, "interconectoras", {})
        for hypo in unique_hypotheses:
            elementos = hypo.get('elementos_clave', [])
            fuerza_total = 0.0
            count = 0
            for i in range(len(elementos)):
                for j in range(i+1, len(elementos)):
                    for inter in interconectoras.values():
                        if inter.es_relevante(elementos[i]) and inter.es_relevante(elementos[j]):
                            fuerza_total += inter.similitud_embedding(getattr(self.razonador.neuronas.get(elementos[i], None), "embedding", getattr(self.razonador.micro_neuronas.get(elementos[i], None), "embedding", None)))
                            count += 1
            if count > 0:
                fuerza_prom = fuerza_total / count
                hypo['confianza'] = min(1.0, hypo['confianza'] + 0.2 * fuerza_prom)
        return unique_hypotheses

    def _refinar_y_evaluar_hipotesis(self, hypotheses, neural_state, retrieved_memories, thinking_memory_content, refinement_strength=0.1, generation_threshold=0.8):
        """
        Refina y evalúa un conjunto de hipótesis de contexto, y genera nuevas hipótesis,
        considering the thinking memory.
        """
        print("DEBUG: SintetizadorContexto - Refinando y evaluando hipótesis.")
        next_hypotheses = []
        newly_generated_hypotheses = []

        active_neurons = neural_state.get('neuronas', {}) # Get continuous activation levels
        active_micro_neurons = neural_state.get('micro_neuronas', {})

        # Evaluate and refine existing hypotheses
        for hypothesis in hypotheses:
            # Evaluate consistency with neural state and memories, including thinking memory
            support_from_neurons = self._evaluate_hypothesis_support_neurons(hypothesis, active_neurons, thinking_memory_content)
            support_from_memories = self._evaluate_hypothesis_support_memories(hypothesis, retrieved_memories) # Thinking memory handled in _evaluate_hypothesis_support_neurons

            # Adjust confidence based on support
            # This is a simplified update rule
            hypothesis['confianza'] = max(0.0, min(1.0, hypothesis['confianza'] + (support_from_neurons + support_from_memories) * refinement_strength - (1 - support_from_neurons - support_from_memories) * refinement_strength))

            # Refine hypothesis elements (placeholder)
            # Add highly supporting neurons/memories to key elements
            # Remove elements that contradict evidence

            # Keep hypotheses above a certain confidence threshold for the next iteration
            if hypothesis['confianza'] > 0.1: # Lower threshold to allow refinement
                next_hypotheses.append(hypothesis)

        # Generate new hypotheses based on strong signals not fully explained (placeholder)
        # Example: If there are highly active neurons or relevant memories not strongly linked to existing hypotheses
        unexplained_neurons = {n_id: level for n_id, level in active_neurons.items() if level > generation_threshold and not self._is_neuron_explained_by_hypotheses(n_id, next_hypotheses)}
        unexplained_memories = {m_id: info for m_id, info in retrieved_memories.items() if info.get('activation', 0.0) > generation_threshold and not self._is_memory_explained_by_hypotheses(m_id, next_hypotheses)}

        if unexplained_neurons or unexplained_memories:
            # Generate a new hypothesis combining these unexplained elements (placeholder)
            new_hypo_elements = list(unexplained_neurons.keys()) + list(unexplained_memories.keys())
            if new_hypo_elements:
                 newly_generated_hypotheses.append({
                     'tipo': 'emergent_context',
                     'descripcion': 'Contexto emergente de señales inexplicadas.',
                     'confianza': 0.6, # Initial confidence for new hypotheses
                     'elementos_clave': list(set(new_hypo_elements)),
                     'soporte_evidencia': {'neurons': list(unexplained_neurons.keys()), 'memories': list(unexplained_memories.keys())}
                 })

        # Combine existing and newly generated hypotheses, ensuring uniqueness
        all_hypotheses = next_hypotheses + newly_generated_hypotheses
        unique_hypotheses = []
        seen_identifiers = set()
        for hypo in all_hypotheses:
            identifier = (hypo['tipo'], tuple(sorted(str(e) for e in hypo['elementos_clave'])))
            if identifier not in seen_identifiers:
                unique_hypotheses.append(hypo)
                seen_identifiers.add(identifier)

        return unique_hypotheses

    # Helper methods for evaluation
    def _evaluate_hypothesis_support_neurons(self, hypothesis, active_neurons, thinking_memory_content):
        """Evaluates how well active neurons and thinking memory neurons support a hypothesis."""
        support = 0.0

        # Support from currently active neurons
        if 'neurons' in hypothesis.get('soporte_evidencia', {}):
            for n_id in hypothesis['soporte_evidencia']['neurons']:
                if n_id in active_neurons and active_neurons[n_id] > 0.5: # Check if neuron is active above a threshold
                    support += active_neurons[n_id] # Add activation level as support

        # Support from key neurons in thinking memory
        if 'thinking_neurons' in hypothesis.get('soporte_evidencia', {}):
            # Ensure thinking_memory_content is a list of dictionaries
            if isinstance(thinking_memory_content, list):
                thinking_neuron_ids = [item.get('id') for item in thinking_memory_content if isinstance(item, dict) and item.get('id')]
                for n_id in hypothesis['soporte_evidencia']['thinking_neurons']:
                    # Find the corresponding info in thinking memory to get initial activation
                    key_neuron_info = next((item for item in thinking_memory_content if isinstance(item, dict) and item.get('id') == n_id), None)
                    if key_neuron_info:
                        # Add support based on the initial activation from thinking memory
                        support += key_neuron_info.get('initial_activation', 0.0) * 1.5 # Give higher weight to thinking memory (example)
            else:
                print(f"Warning: thinking_memory_content is not a list: {type(thinking_memory_content)}")


        # Normalize support (example)
        return min(1.0, support)

    def _evaluate_hypothesis_support_memories(self, hypothesis, retrieved_memories):
        """Placeholder: Evaluates how well retrieved memories support a hypothesis."""
        # Example: Check if key elements of the hypothesis are in retrieved memories
        support = 0.0
        if 'memories' in hypothesis.get('soporte_evidencia', {}):
            for m_id in hypothesis['soporte_evidencia']['memories']:
                if m_id in retrieved_memories:
                    support += retrieved_memories[m_id].get('activation', 0.0) # Add memory relevance as support
        # Normalize support (example)
        return min(1.0, support)

    # Placeholder helper methods for checking if elements are explained
    def _is_neuron_explained_by_hypotheses(self, neuron_id, hypotheses):
        """Placeholder: Checks if a neuron's activity is explained by existing hypotheses."""
        # Example: Check if the neuron is a key element or supporting evidence in any hypothesis
        for hypo in hypotheses:
            if neuron_id in hypo.get('elementos_clave', []):
                return True
            if 'neurons' in hypo.get('soporte_evidencia', {}) and neuron_id in hypo['soporte_evidencia']['neurons']:
                return True
        return False

    def _is_memory_explained_by_hypotheses(self, memory_id, hypotheses):
        """Placeholder: Checks if a retrieved memory is explained by existing hypotheses."""
        # Example: Check if the memory is a key element or supporting evidence in any hypothesis
        for hypo in hypotheses:
            if memory_id in hypo.get('elementos_clave', []):
                return True
            if 'memories' in hypo.get('soporte_evidencia', {}) and memory_id in hypo['soporte_evidencia']['memories']:
                return True
        return False