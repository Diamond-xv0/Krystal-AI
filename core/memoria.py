# Clase base para la memoria de la IA

from .hierarchical_memory import ShortTermMemory, MediumTermMemory, LongTermMemory, ThinkingMemory
import heapq # Import heapq for priority queue

class Memoria:
    def __init__(self):
        self.short_term_memory = ShortTermMemory()
        self.medium_term_memory = MediumTermMemory()
        self.long_term_memory = LongTermMemory()
        self.thinking_memory = ThinkingMemory() # New thinking memory
        # Puedes expandir esto para guardar relaciones, aprendizajes, etc.

        # Graph representation of memory: {concept_id: {related_concept_id: relationship_type, ...}, ...}
        self.graph = {}
        self.concept_metadata = {} # To store metadata about concepts

    # Placeholder methods to interact with hierarchical memory
    def add_to_memory(self, item, memory_level="short"):
        """Adds an item to the specified memory level."""
        if memory_level == "short":
            self.short_term_memory.add(item)
        elif memory_level == "medium":
            # Medium term memory requires a key, this is a simplified example
            key = str(item) # Using item itself as a simple key for now
            self.medium_term_memory.add(key, item)
        elif memory_level == "long":
            self.long_term_memory.add(item)
        elif memory_level == "thinking":
            self.thinking_memory.add(item)
        else:
            print(f"Warning: Unknown memory level '{memory_level}'")

    def retrieve_from_memory(self, query, memory_level="short"):
        """Retrieves items from the specified memory level based on a query."""
        if memory_level == "short":
            return self.short_term_memory.retrieve(query)
        elif memory_level == "medium":
            # Medium term retrieval by key, simplified
            return self.medium_term_memory.retrieve(str(query))
        elif memory_level == "long":
            return self.long_term_memory.retrieve(query)
        elif memory_level == "thinking":
            return self.thinking_memory.retrieve(query)
        else:
            print(f"Warning: Unknown memory level '{memory_level}' for retrieval")
            return []

    def clear_memory(self, memory_level="all"):
        """Clears the specified memory level."""
        if memory_level == "short" or memory_level == "all":
            self.short_term_memory.clear()
        if memory_level == "medium" or memory_level == "all":
            self.medium_term_memory.clear()
        if memory_level == "long" or memory_level == "all":
            self.long_term_memory.clear()
        if memory_level == "thinking" or memory_level == "all":
            self.thinking_memory.clear()
        elif memory_level != "all":
            print(f"Warning: Unknown memory level '{memory_level}' for clearing")

    def add_concept(self, concept_id, metadata=None):
        """Adds a concept node to the memory graph."""
        if concept_id not in self.graph:
            self.graph[concept_id] = {}
            self.concept_metadata[concept_id] = metadata if metadata is not None else {}
            print(f"DEBUG: Memoria - Added concept: {concept_id}")

    def add_relationship(self, concept1_id, concept2_id, relationship_type, metadata=None):
        """Adds a directed relationship edge between two concepts in the memory graph."""
        if concept1_id in self.graph and concept2_id in self.graph:
            self.graph[concept1_id][concept2_id] = relationship_type
            # Optional: store metadata about the relationship if needed
            print(f"DEBUG: Memoria - Added relationship: {concept1_id} --[{relationship_type}]--> {concept2_id}") # Corrected variable name
        else:
            print(f"Warning: Cannot add relationship, one or both concepts not found: {concept1_id}, {concept2_id}")

    def retrieve_associative(self, query_concepts, activation_levels, depth_limit=3, activation_threshold=0.3): # Adjusted depth limit and activation threshold
        """
        Retrieves related concepts from the memory graph based on initial activated concepts
        and their activation levels, traversing the graph associatively using a priority queue.
        """
        retrieved_info = {}
        # Use a set to keep track of visited concepts to avoid cycles and redundant processing
        visited_concepts = set()
        # Use a dictionary to store the maximum activation level reached for each concept
        max_activation = {}

        # Priority queue stores tuples: (-activation, depth, concept_id)
        # Negative activation is used because heapq is a min-heap, and we want higher activation first
        pq = []

        # Initialize the priority queue with query concepts
        for concept, activation in zip(query_concepts, activation_levels):
            if activation > activation_threshold:
                # Ensure concept exists in the graph before adding to PQ
                if concept in self.graph:
                    heapq.heappush(pq, (-activation, 0, concept))
                    max_activation[concept] = activation
                    visited_concepts.add(concept) # Mark as visited upon adding to PQ

        print(f"DEBUG: Memoria - Starting associative retrieval with {len(pq)} initial concepts.")

        while pq:
            current_neg_activation, current_depth, current_concept = heapq.heappop(pq)
            current_activation = -current_neg_activation # Convert back to positive activation

            # If we found a better path to this concept already, skip
            if current_activation < max_activation.get(current_concept, 0.0):
                 continue

            # Add the current concept and its info to results
            retrieved_info[current_concept] = {
                'metadata': self.concept_metadata.get(current_concept),
                'activation': current_activation,
                'depth': current_depth
            }

            if current_depth >= depth_limit:
                continue

            # Explore connected concepts
            if current_concept in self.graph:
                for related_concept, relationship_type in self.graph[current_concept].items():
                    # Calculate propagated activation
                    # This is a placeholder; propagation rules should be more sophisticated
                    # based on relationship_type and potentially relationship strength/weight
                    propagation_factor = 0.7 # Example factor
                    propagated_activation = current_activation * propagation_factor

                    if propagated_activation > activation_threshold:
                        # If this is a new concept or we found a path with higher activation
                        if related_concept not in max_activation or propagated_activation > max_activation[related_concept]:
                            max_activation[related_concept] = propagated_activation
                            # Add to PQ even if visited, if we found a better path
                            heapq.heappush(pq, (-propagated_activation, current_depth + 1, related_concept))
                            visited_concepts.add(related_concept) # Mark as visited

        print(f"DEBUG: Memoria - Associative retrieval found {len(retrieved_info)} concepts.")
        return retrieved_info # Return dictionary of retrieved concepts and their info