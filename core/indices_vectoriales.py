import math
import sys
from typing import List, Tuple, Dict, Any
import heapq

# Existing helper functions
def dot_product(v1: List[float], v2: List[float]) -> float:
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v: List[float]) -> float:
    """Calculates the magnitude (L2 norm) of a vector."""
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    dot = dot_product(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

# KD-tree Node
class KDNode:
    def __init__(self, vector_id: Any, vector: List[float], metadata: Dict[str, Any], axis: int, left=None, right=None):
        self.vector_id = vector_id
        self.vector = vector
        self.metadata = metadata
        self.axis = axis # Dimension used for splitting at this node
        self.left = left
        self.right = right

class VectorIndex:
    """
    A pure Python vector indexing system supporting grammatical categories
    using a basic KD-tree for O(log n) search capability.
    """
    def __init__(self):
        self.root: KDNode | None = None
        self.dimension: int | None = None
        self._size = 0
        self._removed_ids: set = set() # Set to store IDs of removed vectors

    def add_vector(self, vector_id: Any, vector: List[float], metadata: Dict[str, Any] = None, category: Any = None):
        """Adds a vector and its associated metadata to the index. Optionally accepts a category."""
        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            print(f"Error: Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}.")
            return

        if metadata is None:
            metadata = {}

        if category is not None:
            metadata = dict(metadata)  # avoid mutating input
            metadata['category'] = category

        # If the vector_id was previously removed, unmark it
        if vector_id in self._removed_ids:
            self._removed_ids.remove(vector_id)

        # Simple insertion. If ID exists, it will be added again.
        # For true update/overwrite, a more complex KD-tree implementation is needed.
        # For this basic version, adding an existing ID results in multiple nodes with the same ID.
        # Search will return all of them if they are neighbors.
        self.root = self._insert(self.root, vector_id, vector, metadata, 0)
        self._size += 1
        # --- Temporary Debug Logging ---
        print("\n--- VectorIndex Add Debug ---", file=sys.stderr)
        print(f"Added vector with ID: {vector_id}", file=sys.stderr)
        if self.root:
            print(f"Root node vector: {self.root.vector}, axis: {self.root.axis}", file=sys.stderr)
            if self.root.left:
                print(f"  Root left child vector: {self.root.left.vector}, axis: {self.root.left.axis}", file=sys.stderr)
            if self.root.right:
                print(f"  Root right child vector: {self.root.right.vector}, axis: {self.root.right.axis}", file=sys.stderr)
        print("---------------------------\n", file=sys.stderr)
        # --- End Temporary Debug Logging ---

    def _insert(self, node: KDNode | None, vector_id: Any, vector: List[float], metadata: Dict[str, Any], depth: int) -> KDNode:
        """Recursive helper for inserting a vector into the KD-tree."""
        axis = depth % self.dimension

        if node is None:
            return KDNode(vector_id, vector, metadata, axis)

        if vector[axis] < node.vector[axis]:
            node.left = self._insert(node.left, vector_id, vector, metadata, depth + 1)
        else:
            node.right = self._insert(node.right, vector_id, vector, metadata, depth + 1)

        return node

    def get_vector(self, vector_id: Any) -> List[float] | None:
        """Retrieves a vector by its ID. O(n) in worst case for KD-tree."""
        # Simple tree traversal to find by ID, skipping removed IDs.
        return self._find_by_id(self.root, vector_id)

    def _find_by_id(self, node: KDNode | None, vector_id: Any) -> List[float] | None:
        """Recursive helper to find a vector by ID, skipping removed."""
        if node is None:
            return None
        if node.vector_id == vector_id and node.vector_id not in self._removed_ids:
            return node.vector
        left_result = self._find_by_id(node.left, vector_id)
        if left_result:
            return left_result
        right_result = self._find_by_id(node.right, vector_id)
        return right_result

    def get_metadata(self, vector_id: Any) -> Dict[str, Any] | None:
        """Retrieves metadata by vector ID. O(n) in worst case."""
        # Simple tree traversal to find by ID, skipping removed IDs.
        return self._find_metadata_by_id(self.root, vector_id)

    def _find_metadata_by_id(self, node: KDNode | None, vector_id: Any) -> Dict[str, Any] | None:
        """Recursive helper to find metadata by ID, skipping removed."""
        if node is None:
            return None
        if node.vector_id == vector_id and node.vector_id not in self._removed_ids:
            return node.metadata
        left_result = self._find_metadata_by_id(node.left, vector_id)
        if left_result:
            return left_result
        right_result = self._find_metadata_by_id(node.right, vector_id)
        return right_result


    def search_similar(self, query_vector: List[float], top_k: int = 5, grammar_category: str = None) -> List[Tuple[Any, float]]:
        """
        Searches for vectors similar to the query vector using the KD-tree.
        """
        if self.root is None or self.dimension is None or len(query_vector) != self.dimension:
            return []

        # Min-heap for (distance, vector_id, similarity)
        best_neighbors: List[Tuple[float, Any, float]] = []

        def search_recursive(node: KDNode | None, depth: int):
            if node is None:
                return
            axis = depth % self.dimension

            # Only process if the node hasn't been removed
            if node.vector_id not in self._removed_ids:
                # Calculate similarity and distance
                similarity = cosine_similarity(query_vector, node.vector)
                distance = 1.0 - similarity

                # Add to heap if it's one of the top_k
                if len(best_neighbors) < top_k:
                    heapq.heappush(best_neighbors, (distance, node.vector_id, similarity))
                elif distance < best_neighbors[0][0]:
                    heapq.heapreplace(best_neighbors, (distance, node.vector_id, similarity))

            # Determine which child to search first
            # The splitting decision is based on the node's vector value along the current axis.
            if query_vector[axis] < node.vector[axis]:
                near_child = node.left
                far_child = node.right
            else:
                near_child = node.right
                far_child = node.left

            # Recursively search the near child
            search_recursive(near_child, depth + 1)

            # Check if the other side of the splitting plane could contain a closer point.
            # Simplified pruning check for basic implementation:
            # If the heap is not full, or if the distance from the query to the splitting plane
            # along the current axis is less than the distance of the current furthest neighbor.
            # This is an approximation for cosine similarity.
            axis_distance = abs(query_vector[axis] - node.vector[axis])
            # print(f"DEBUG: VectorIndex.search_similar - Axis distance: {axis_distance}, Current furthest distance: {best_neighbors[0][0] if best_neighbors else 'N/A'}") # Debug log - Removed debug log

            # If the heap is not full, we must explore the far child.
            # If the heap is full, explore the far child only if the splitting plane is within
            # the radius of the current furthest neighbor.
            # The radius is best_neighbors[0][0] (which is 1 - similarity).
            # A point on the other side of the plane at the same axis value as the query
            # would have an axis distance of axis_distance.
            # If axis_distance < best_neighbors[0][0], it's possible a point on the other side
            # is closer. This is a heuristic for cosine similarity.

            # Disable pruning for cosine similarity search to ensure correctness.
            # Always explore the far child after the near child.
            search_recursive(far_child, depth + 1)

        # Perform the initial search to get the top_k nearest neighbors based on vector similarity
        search_recursive(self.root, 0)

        # Convert heap to results list, sorted by similarity descending.
        # Apply grammar_category filter *after* finding nearest neighbors
        filtered_results = []
        for distance, vector_id, similarity in best_neighbors:
            # Retrieve metadata for the vector_id
            metadata = self.get_metadata(vector_id) # Need to retrieve metadata here

            # Apply the grammar_category filter
            if grammar_category is None or (metadata and metadata.get('grammar_category') == grammar_category):
                filtered_results.append((vector_id, similarity))

        # Sort the filtered results by similarity descending
        filtered_results.sort(key=lambda item: item[1], reverse=True)

        return filtered_results

    def remove_vector(self, vector_id: Any):
        """Removes a vector and its metadata from the index by marking its ID as removed."""
        if vector_id in self._removed_ids:
            print(f"Warning: Vector ID {vector_id} already marked as removed.")
            return

        # Check if the ID exists in the tree at all before marking as removed
        # This requires traversing the tree, which is O(n) in worst case.
        # A more efficient check would require a separate dictionary mapping ID to node,
        # but that adds memory and complexity to keep in sync with tree modifications.
        # For this basic implementation, we accept the O(n) check here.
        if self._find_node_by_id(self.root, vector_id):
             self._removed_ids.add(vector_id)
             self._size -= 1
        else:
            print(f"Warning: Vector ID {vector_id} not found in index.")

    def _find_node_by_id(self, node: KDNode | None, vector_id: Any) -> KDNode | None:
        """Recursive helper to find a node by ID (does not skip removed)."""
        if node is None:
            return None
        if node.vector_id == vector_id:
            return node
        left_result = self._find_node_by_id(node.left, vector_id)
        if left_result:
            return left_result
        right_result = self._find_node_by_id(node.right, vector_id)
        return right_result


# Example Usage (for testing purposes) - Update this to reflect the new structure


if __name__ == "__main__":
    index = VectorIndex()

    # Add some vectors
    index.add_vector("doc1", [1.0, 2.0, 3.0], {"grammar_category": "noun"})
    index.add_vector("doc2", [1.1, 2.1, 3.1], {"grammar_category": "noun"})
    index.add_vector("doc3", [-1.0, -2.0, -3.0], {"grammar_category": "verb"})
    index.add_vector("doc4", [10.0, 20.0, 30.0], {"grammar_category": "noun"})
    index.add_vector("doc5", [0.5, 0.6, 0.7], {"grammar_category": "adjective"})

    print(f"Index size after adding: {index._size}")

    # Search for similar vectors
    query = [1.05, 2.05, 3.05]
    similar_docs = index.search_similar(query, top_k=3)
    print(f"Similar documents to {query}: {similar_docs}")

    # Search for similar vectors with grammar category filter
    similar_nouns = index.search_similar(query, top_k=3, grammar_category="noun")
    print(f"Similar nouns to {query}: {similar_nouns}")

    # Remove a vector
    index.remove_vector("doc3")
    print(f"Index size after removing doc3: {index._size}")
    print(f"Removed IDs: {index._removed_ids}")

    # Search again after removal
    similar_docs_after_removal = index.search_similar(query, top_k=3)
    print(f"Similar documents after removal: {similar_docs_after_removal}")

    # Try adding a removed vector back
    index.add_vector("doc3", [-1.0, -2.0, -3.0], {"grammar_category": "verb"})
    print(f"Index size after re-adding doc3: {index._size}")
    print(f"Removed IDs after re-adding doc3: {index._removed_ids}")
    similar_docs_after_readd = index.search_similar(query, top_k=3)
    print(f"Similar documents after re-adding doc3: {similar_docs_after_readd}")

    # Add a vector with existing ID
    index.add_vector("doc1", [1.0, 2.0, 3.0], {"grammar_category": "noun"})
    print(f"Index size after adding doc1 again: {index._size}")
    similar_docs_after_duplicate_add = index.search_similar(query, top_k=5) # Increased top_k to see duplicates
    print(f"Similar documents after adding doc1 again: {similar_docs_after_duplicate_add}")

    # --- Temporary Exact Match Test Case ---
    print("\n--- Exact Match Test ---")
    test_id = "exact_test_vec"
    test_vector = [5.5, 6.6, 7.7]
    test_metadata = {"purpose": "exact_match_test"}

    print(f"Adding test vector with ID: {test_id}, vector: {test_vector}")
    index.add_vector(test_id, test_vector, test_metadata)

    print(f"Searching for exact match of vector: {test_vector}")
    exact_match_results = index.search_similar(test_vector, top_k=1)
    print(f"Exact match search results: {exact_match_results}")
    print("------------------------\n")
    # --- End Temporary Exact Match Test Case ---
    
# Instancia global mínima funcional para importación
embedding_index = VectorIndex()
# (Eliminada la línea duplicada de embedding_index)
# index_manager mínimo para compatibilidad con razonador_optimizado.py
class IndexManager:
    def optimize_all(self):
        # Implementación mínima: no hace nada
        pass

    def get_all_stats(self):
        # Implementación mínima: retorna diccionario vacío
        return {}

index_manager = IndexManager()