from core.micro_neurona import MicroNeurona
from core.neurona import Neurona
from core.macro_neurona import MacroNeurona
from core.MemoryNs import registrar_memoria
from core.indices_vectoriales import VectorIndex
import asyncio
import concurrent.futures
import threading
from typing import List, Tuple, Dict, Any
from core.neural_events import NeuralEvent, NeuralEventPublisher
from core.priority_manager import PriorityManager
from core.neurona_interconectora import NeuronaInterconectora

class Razonador:
    def __init__(self, memoria, personalidad):
        self.memoria = memoria
        self.personalidad = personalidad
        self.micro_neuronas = {}
        self.neuronas = {}
        self.macro_neuronas = {}
        self.interconectoras = {}  # id: NeuronaInterconectora

    def registrar_interconectora(self, interconectora):
        self.interconectoras[interconectora.id] = interconectora

        # Initialize event publisher and priority manager
        self.event_publisher = NeuralEventPublisher()
        self.priority_manager = PriorityManager()

        # Initialize ThreadPoolExecutor for parallel evaluation
        self.executor = concurrent.futures.ThreadPoolExecutor()
        
        # --- Vocabulario Unificado ---
        # Ya no hay separación entre comprensión y generación. Todas las palabras
        # con sus metadatos ricos viven en una sola lista para potenciar ambas capacidades.
        self.vocabulario_palabras_clave = []
        self.vector_index = VectorIndex() # Add VectorIndex instance
        
        self.historial_ciclos = []

    def meta_ajuste_parametros(self, window=10):
        """
        Meta-razonamiento: ajusta umbrales y tasas de decaimiento según desempeño reciente.
        window: número de ciclos a analizar.
        """
        if not self.historial_ciclos:
            return
        recientes = self.historial_ciclos[-window:]
        # Ajuste de micro_neuronas
        for mn in self.micro_neuronas.values():
            activaciones = [c['micro_activaciones'][mn.id]['activation_level'] for c in recientes if mn.id in c['micro_activaciones']]
            if activaciones:
                avg = sum(activaciones) / len(activaciones)
                # Si la activación promedio es muy baja, bajar el umbral y el decaimiento
                if avg < 0.2:
                    mn.umbral_activacion = max(0.1, mn.umbral_activacion - 0.05)
                    mn.decay_rate = max(0.01, mn.decay_rate - 0.01)
                # Si es muy alta, subir umbral y decaimiento
                elif avg > 0.8:
                    mn.umbral_activacion = min(1.0, mn.umbral_activacion + 0.05)
                    mn.decay_rate = min(1.0, mn.decay_rate + 0.01)
        # Ajuste de neuronas
        for n in self.neuronas.values():
            activaciones = [c['neuronas_detalle'][n.id]['activation_level'] for c in recientes if n.id in c['neuronas_detalle']]
            if activaciones:
                avg = sum(activaciones) / len(activaciones)
                if avg < 0.2:
                    n.umbral_activacion = max(0.1, n.umbral_activacion - 0.05)
                    n.decay_rate = max(0.01, n.decay_rate - 0.01)
                elif avg > 0.8:
                    n.umbral_activacion = min(1.0, n.umbral_activacion + 0.05)
                    n.decay_rate = min(1.0, n.decay_rate + 0.01)

    def registrar_micro_neurona(self, mn):
        self.micro_neuronas[mn.id] = mn
        # Clasificamos la neurona en la lista apropiada según su tipo.
        # El tipo 'palabra_clave' ahora es el estándar para todas las palabras del vocabulario.
        if mn.tipo == 'palabra_clave':
            self.vocabulario_palabras_clave.append(mn)
            # Add the neuron's vector and metadata to the index
            vector_id, vector, metadata = mn.get_index_data()
            self.vector_index.add_vector(vector_id, vector, metadata)

    def registrar_neurona(self, n):
        self.neuronas[n.id] = n

    def registrar_macro_neurona(self, macro_n):
        self.macro_neuronas[macro_n.id] = macro_n

    def reset(self):
        """Resetea el estado de activación de todas las neuronas en todas las capas."""
        # Importante: reseteamos TODAS las neuronas, no solo las de un vocabulario.
        for mn in self.micro_neuronas.values():
            mn.reset()

    def evaluar_capa_neuronas_paralelo(self, activated_mn_ids: Dict[str, bool]):
        """
        Evalúa la capa de Neuronas en paralelo utilizando ThreadPoolExecutor.
        """
        futures = []
        results = {}

        # Submit evaluation tasks for each Neurona
        for n_id, neurona in self.neuronas.items():
            # Ajuste dinámico de pesos usando interconectoras antes de evaluar
            for mn_id in neurona.condiciones_mn:
                for inter_id, inter in self.interconectoras.items():
                    if inter.es_relevante(mn_id) and inter.es_relevante(n_id):
                        # Ajustar el peso de la conexión según la similitud de embeddings
                        similitud = inter.similitud_embedding(neurona.embedding)
                        # Peso base + refuerzo por similitud (puede ajustarse la fórmula)
                        neurona.weights[mn_id] = neurona.weights.get(mn_id, 0.0) + 0.2 * similitud
            # Pass the activated micro-neuron IDs to the evaluar method
            future = self.executor.submit(neurona.evaluar, activated_mn_ids)
            futures.append((future, n_id))

        # Collect results as they complete
        for future, n_id in futures:
            try:
                # The evaluar method updates the neuron's state directly,
                # but we can still check for exceptions or get return values if needed.
                # For now, we just wait for completion.
                future.result()
                results[n_id] = self.neuronas[n_id].activa # Store the final activation state
            except Exception as exc:
                print(f'Neurona {n_id} generated an exception: {exc}')
                results[n_id] = False # Mark as not active in case of error

        return results # Return the dictionary of final activation states

    def procesar_entrada_iterativo(self, vectores_entrada, frase_original=None, umbral_mn=0.8, num_iteraciones=10):
        """
        Procesa la entrada iterativamente a través de las capas neuronales con feedback y memoria.
        """
        # Reset all neuron activations at the start of a new reasoning cycle
        self.reset()
        print("DEBUG: Razonador - All neurons reset.")

        # Clear thinking memory
        self.memoria.clear_memory("thinking")
        print("DEBUG: Razonador - Thinking memory cleared.")

        # Initial activation based on input vectors (similar to original ciclo_activacion)
        activated_mn_ids = set()
        initial_activations = {} # Store initial activation levels
        for i, vec_entrada in enumerate(vectores_entrada):
            # Use the vector index to find similar neurons based on the input vectors
            similar_results = self.vector_index.search_similar(vec_entrada, top_k=10) # Adjust top_k as needed

            # Ajuste de activación usando interconectoras entre micro-neuronas (tela de araña)
            for mn_id, similarity in similar_results:
                if mn_id in self.micro_neuronas:
                    mn = self.micro_neuronas[mn_id]
                    # Buscar interconectoras relevantes para esta micro-neurona
                    for inter in self.interconectoras.values():
                        if inter.es_relevante(mn_id):
                            # Ajustar activación base según similitud de embeddings
                            sim = inter.similitud_embedding(mn.embedding)
                            mn.activation_level += 0.2 * sim  # Refuerzo configurable

            # Activate the corresponding micro_neuronas if similarity is above threshold
            for mn_id, similarity in similar_results:
                if similarity >= umbral_mn:
                    if mn_id in self.micro_neuronas and mn_id not in activated_mn_ids:
                        mn = self.micro_neuronas[mn_id]
                        mn.activar(vectores_entrada, frase_original=frase_original, umbral=umbral_mn)
                        if mn.activa:
                            activated_mn_ids.add(mn_id)
                            initial_activations[mn_id] = mn.confianza # Store initial confidence
                            activation_event = NeuralEvent("neuron_activated", {"neuron_id": mn_id, "neuron_type": "micro", "activation_level": mn.confianza})
                            self.event_publisher.publish(activation_event)
                            self.priority_manager.add_item(mn, priority=1) # Assign a default priority for now

        # Identify and register key neurons (e.g., high initial activation, specific types) in thinking memory
        print("DEBUG: Razonador - Identifying and registering key neurons in thinking memory.")
        for mn_id in activated_mn_ids:
            mn = self.micro_neuronas[mn_id]
            # Example criteria for a "key" neuron: high initial activation and is a keyword
            # This logic can be refined based on neuron types (e.g., "question", "request")
            if initial_activations.get(mn_id, 0) >= umbral_mn and mn.tipo == 'palabra_clave':
                key_neuron_info = {
                    "id": mn.id,
                    "tipo": mn.tipo,
                    "initial_activation": initial_activations[mn_id],
                    "metadata": mn.metadata # Include relevant metadata
                }
                registrar_memoria(key_neuron_info, "thinking")
                print(f"DEBUG: Razonador - Registered key neuron {mn.id} in thinking memory.")

        # Process activated micro-neurons from the priority queue (demonstration of priority manager use)
        processed_prioritized_mn_ids = set()
        while not self.priority_manager.is_empty():
            prioritized_mn = self.priority_manager.get_next_item()
            processed_prioritized_mn_ids.add(prioritized_mn.id)

        # Iterative processing loop
        for iteracion in range(num_iteraciones):
            print(f"DEBUG: Razonador - Iteración {iteracion + 1}/{num_iteraciones}")

            # 1. Propagate activation forward (MN -> N -> MN)
            current_mn_activation_state = {mn_id: self.micro_neuronas[mn_id].activation_level for mn_id in self.micro_neuronas}
            activated_n_results = self.evaluar_capa_neuronas_paralelo(current_mn_activation_state)

            # Aprendizaje Hebbiano/adaptativo de pesos en Neuronas
            for n_id, neurona in self.neuronas.items():
                neurona.update_weights(current_mn_activation_state)

            # --- Inhibición lateral: reducir activación de neuronas menos relevantes con alto solapamiento ---
            from collections import defaultdict
            overlap_groups = defaultdict(list)
            for n_id, neurona in self.neuronas.items():
                key = tuple(sorted(neurona.condiciones_mn))
                overlap_groups[key].append(neurona)
            for group in overlap_groups.values():
                if len(group) > 1:
                    max_act = max(n.activation_level for n in group)
                    for n in group:
                        if n.activation_level < max_act:
                            n.activation_level *= 0.7
                            n.activa = n.activation_level >= n.umbral_activacion

            # --- Evaluar MacroNeuronas ---
            ns_activas_ids = {n_id for n_id, n in self.neuronas.items() if n.activa}
            mns_activas_ids = {mn_id for mn_id, mn in self.micro_neuronas.items() if mn.activa}
            macro_activaciones = {}
            for macro_id, macro in self.macro_neuronas.items():
                macro.evaluar(ns_activas_ids, mns_activas_ids)
                macro_activaciones[macro_id] = macro.activa

            # MacroNeuronas pueden inhibir micro y neuronas normales si están activas
            for macro in self.macro_neuronas.values():
                if macro.activa:
                    # Ejemplo: reducir activación de todas las neuronas normales no incluidas en condiciones_n
                    for n_id, n in self.neuronas.items():
                        if n_id not in macro.condiciones_n:
                            n.activation_level *= 0.5
                            n.activa = n.activation_level >= n.umbral_activacion
                    # Ejemplo: reducir activación de micro_neuronas no incluidas en exclusiones
                    for mn_id, mn in self.micro_neuronas.items():
                        if mn_id not in macro.exclusiones_mn:
                            mn.activation_level *= 0.7
                            mn.activa = mn.activation_level >= mn.umbral_activacion

            # Propagate activation from Neuronas back to MicroNeuronas (Feedback)
            self._aplicar_feedback_neuronas_a_micro(self.neuronas)

            # 2. Integrate Memory Retrieval
            memories_retrieved = self._recuperar_memoria_basada_en_activacion(activated_n_results)

            # Incorporate retrieved memories into MicroNeurona activation for the next iteration
            self._incorporar_memorias_recuperadas(memories_retrieved)

            # 3. Apply Decay to all neurons
            self._aplicar_decaimiento()

            # 4. Update history for this iteration
            ciclo_iteracion = {
                'iteracion': iteracion,
                'micro_activas_inicio_iteracion': {mn.id: self.micro_neuronas[mn.id].activa for mn in self.micro_neuronas.values() if self.micro_neuronas[mn.id].activa},
                'micro_activaciones': {mn.id: {
                    'activation_level': mn.activation_level,
                    'historial': mn.historial_activacion[-5:],
                    'umbral': mn.umbral_activacion
                } for mn in self.micro_neuronas.values()},
                'neuronas_activas': activated_n_results,
                'neuronas_detalle': {n.id: {
                    'activation_level': n.activation_level,
                    'activa': n.activa,
                    'pesos': n.weights.copy(),
                    'historial': n.historial_activacion[-5:]
                } for n in self.neuronas.values()},
                'macro_activaciones': macro_activaciones,
                'macro_detalle': {m.id: {
                    'activa': m.activa,
                    'historial': m.historial_activacion[-5:],
                    'condiciones_n': m.condiciones_n
                } for m in self.macro_neuronas.values()},
                'memorias_retrieved': memories_retrieved # Store retrieved memories
            }
            self.historial_ciclos.append(ciclo_iteracion)

        # After iterations, the final state of neuron activations represents the reasoning result.
        # Further processing (e.g., context synthesis, response generation) would use this final state.
        final_activation_state = {
            'micro_neuronas': {mn.id: self.micro_neuronas[mn.id].activa for mn in self.micro_neuronas.values()},
            'neuronas': activated_n_results
        }
        return final_activation_state # Return the final state

    def _aplicar_feedback_neuronas_a_micro(self, activated_n_results, feedback_strength=0.05):
        """Applies feedback from active Neuronas back to their input MicroNeuronas."""
        print("DEBUG: Razonador - Applying feedback from Neuronas to MicroNeuronas.")
        # Iterate through all Neuronas to apply feedback based on their continuous activation level
        for n_id, neurona in self.neuronas.items():
            # Only apply feedback from Neuronas with activation level above a certain point (could be threshold or lower)
            # Using the Neurona's own activation level for feedback strength
            if neurona and neurona.activation_level > 0: # Apply feedback if there's any activation
                # Propagate feedback to input MicroNeuronas
                for mn_id in neurona.condiciones_mn:
                    micro_neurona = self.micro_neuronas.get(mn_id)
                    if micro_neurona:
                        # Increase MicroNeurona activation based on Neurona activation level and feedback strength
                        # This is a simple additive model; more complex models could use weights or other functions
                        micro_neurona.activation_level = min(1.0, micro_neurona.activation_level + neurona.activation_level * feedback_strength)
                        # Re-evaluate MicroNeurona active state based on its threshold
                        # Use the MicroNeurona's own threshold for activation check
                        micro_neurona.activa = micro_neurona.activation_level >= micro_neurona.umbral_activacion if hasattr(micro_neurona, 'umbral_activacion') else micro_neurona.activation_level >= 0.7 # Use default 0.7 if umbral not set
                        # Optional: Log feedback application
                        # print(f"DEBUG: Razonador - Feedback applied to MicroNeurona {mn_id}, new activation: {micro_neurona.activation_level}")

    def _recuperar_memoria_basada_en_activacion(self, activated_n_results):
        """Retrieves memory based on active Neuronas using associative retrieval."""
        # Extract active neuron IDs and their activation states
        active_neurons_info = [(n_id, activation) for n_id, activation in activated_n_results.items() if activation]

        if not active_neurons_info:
            print("DEBUG: Razonador - No active neurons to base memory retrieval on.")
            return {} # Return empty dictionary if no active neurons

        query_concepts = [n_id for n_id, activation in active_neurons_info]
        # Using activation state (True/False) as a basic activation level for now.
        # Could use confidence score if Neurona class is updated to provide it.
        activation_levels = [1.0 if activation else 0.0 for n_id, activation in active_neurons_info]

        print(f"DEBUG: Razonador - Querying memory associatively with concepts: {query_concepts}")

        # Use the new associative retrieval method from Memoria
        retrieved_memories = self.memoria.retrieve_associative(query_concepts, activation_levels)

        return retrieved_memories # Return the dictionary of retrieved memory items

    def _incorporar_memorias_recuperadas(self, memories_retrieved, incorporation_strength=0.1):
        """Incorporates retrieved memories by boosting the activation of associated neurons."""
        print(f"DEBUG: Razonador - Incorporating retrieved memories: {memories_retrieved}")
        if not memories_retrieved:
            print("DEBUG: Razonador - No memories retrieved to incorporate.")
            return

        # Iterate through retrieved memories (concepts)
        for concept_id, memory_info in memories_retrieved.items():
            # Assuming neurons have metadata linking them to memory concepts
            # Find MicroNeuronas associated with this concept
            for mn in self.micro_neuronas.values():
                if mn.metadata.get('memory_concept_id') == concept_id:
                    # Boost activation based on memory relevance/activation and incorporation strength
                    # Boost activation based on memory relevance/activation and incorporation strength
                    # Using memory_info.get('activation', 1.0) as a relevance proxy
                    boost = memory_info.get('activation', 1.0) * incorporation_strength * 0.5 # Reduced boost strength
                    mn.activation_level = min(1.0, mn.activation_level + boost)
                    # Re-evaluate active state
                    # Use the MicroNeurona's own threshold for activation check
                    mn.activa = mn.activation_level >= mn.umbral_activacion if hasattr(mn, 'umbral_activacion') else mn.activation_level >= 0.7 # Use default 0.7 if umbral not set
                    # Optional: Log incorporation
                    # print(f"DEBUG: Razonador - Incorporated memory {concept_id} into MicroNeurona {mn.id}, new activation: {mn.activation_level}")

            # Find Neuronas associated with this concept
            for n in self.neuronas.values():
                if n.metadata.get('memory_concept_id') == concept_id:
                    # Boost activation
                    boost = memory_info.get('activation', 1.0) * incorporation_strength * 0.5 # Reduced boost strength
                    n.activation_level = min(1.0, n.activation_level + boost)
                    # Re-evaluate active state
                    n.activa = n.activation_level >= n.umbral_activacion # Use Neurona's own umbral
                    # Optional: Log incorporation
                    # print(f"DEBUG: Razonador - Incorporated memory {concept_id} into Neurona {n.id}, new activation: {n.activation_level}")

    def _aplicar_decaimiento(self):
        """Applies activation decay to all MicroNeuronas and Neuronas."""
        print("DEBUG: Razonador - Applying activation decay.")
        # Apply decay to MicroNeuronas
        for mn in self.micro_neuronas.values():
            mn.aplicar_decaimiento()

        # Apply decay to Neuronas
        for n in self.neuronas.values():
            n.aplicar_decaimiento()
