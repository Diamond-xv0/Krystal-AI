"""
Pool de Embeddings Optimizado para Krystal AI
Gestiona embeddings pre-calculados y optimiza el uso de memoria.
"""

import threading
import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pickle
import os
import math
from .cache_manager import cache_manager


class EmbeddingPool:
    """Pool optimizado para gestión de embeddings con lazy loading y compresión."""
    
    def __init__(self, cache_dir: str = "cache/embeddings", max_memory_mb: int = 512):
        self.cache_dir = cache_dir
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Almacenamiento en memoria
        self.embeddings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Estadísticas y control
        self.access_count: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        self.memory_usage = 0
        
        # Threading
        self.lock = threading.RLock()
        
        # Configuración
        self.compression_enabled = True
        self.lazy_loading = True
        
        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cargar embeddings persistentes
        self._load_persistent_embeddings()
    
    def _get_cache_path(self, key: str) -> str:
        """Obtiene la ruta del archivo de caché para una clave."""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def _load_persistent_embeddings(self):
        """Carga embeddings persistentes desde disco."""
        if not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                try:
                    filepath = os.path.join(self.cache_dir, filename)
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        key = filename[:-4]  # Remover .pkl
                        
                        if 'embedding' in data and 'metadata' in data:
                            # No cargar en memoria inmediatamente si lazy loading está habilitado
                            if not self.lazy_loading:
                                self.embeddings[key] = data['embedding']
                                self.memory_usage += len(data['embedding']) * 4 # Approximation
                            
                            self.metadata[key] = data['metadata']
                            
                except Exception as e:
                    print(f"Error cargando embedding {filename}: {e}")
    
    def _save_to_disk(self, key: str, embedding: List[float], metadata: Dict):
        """Guarda un embedding a disco."""
        try:
            filepath = self._get_cache_path(key)
            
            compressed_embedding = None
            is_compressed = False
            if self.compression_enabled:
                compressed_embedding = self._compress_embedding(embedding)
                is_compressed = True
                
            data = {
                'embedding': compressed_embedding if is_compressed else embedding,
                'metadata': metadata,
                'timestamp': time.time(),
                'compressed': is_compressed # Add compression flag
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            print(f"Error guardando embedding {key}: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[List[float]]:
        """Carga un embedding desde disco."""
        try:
            filepath = self._get_cache_path(key)
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                embedding_data = data.get('embedding')
                is_compressed = data.get('compressed', False) # Check for compression flag

                if is_compressed and embedding_data is not None:
                    return self._decompress_embedding(embedding_data)
                elif embedding_data is not None:
                    return embedding_data
                else:
                    return None
                
        except Exception as e:
            print(f"Error cargando embedding {key} desde disco: {e}")
            return None

    def _compress_embedding(self, embedding: List[float]) -> Tuple[float, float, List[int]]:
        """Comprime un embedding usando cuantización simple."""
        if not embedding:
            return 0.0, 0.0, []

        min_val = min(embedding)
        max_val = max(embedding)
        
        # Avoid division by zero if all values are the same
        if math.isclose(min_val, max_val):
             # If all values are the same, store the value and a list of zeros
            quantized_embedding = [0] * len(embedding)
            return min_val, max_val, quantized_embedding

        # Scale values to 0-255 range and convert to integers
        scale = 255.0 / (max_val - min_val)
        quantized_embedding = [int((val - min_val) * scale) for val in embedding]

        return min_val, max_val, quantized_embedding

    def _decompress_embedding(self, compressed_embedding: Tuple[float, float, List[int]]) -> List[float]:
        """Descomprime un embedding cuantizado."""
        min_val, max_val, quantized_embedding = compressed_embedding
        
        if not quantized_embedding:
            return []

        # Avoid division by zero if min_val and max_val are the same
        if math.isclose(min_val, max_val):
            # If min and max are the same, all original values were the same
            return [min_val] * len(quantized_embedding)

        # Scale integers back to original approximate float range
        scale = (max_val - min_val) / 255.0
        embedding = [min_val + val * scale for val in quantized_embedding]

        return embedding

    def _evict_least_used(self):
        """Expulsa embeddings menos usados para liberar memoria."""
        if not self.embeddings:
            return
        
        # Ordenar por frecuencia de acceso y tiempo de último acceso
        items = []
        current_time = time.time()
        
        for key in self.embeddings:
            access_count = self.access_count[key]
            last_access = self.last_access.get(key, 0)
            age = current_time - last_access
            
            # Score más bajo = menos importante
            score = access_count / (1 + age / 3600)  # Penalizar por edad en horas
            items.append((score, key))
        
        # Ordenar por score ascendente
        items.sort()
        
        # Expulsar hasta liberar 25% de la memoria
        target_memory = self.max_memory_bytes * 0.75
        
        for score, key in items:
            if self.memory_usage <= target_memory:
                break
            
            # Guardar a disco antes de expulsar
            if key in self.embeddings:
                self._save_to_disk(key, self.embeddings[key], self.metadata.get(key, {}))
                
                # Approximate memory usage of the compressed embedding if compression is enabled
                if self.compression_enabled:
                    # Rough approximation: size of list of ints + size of two floats
                    compressed_size_approx = len(self.embeddings[key]) * 1 + 16
                    self.memory_usage -= compressed_size_approx
                else:
                    self.memory_usage -= len(self.embeddings[key]) * 4 # Original Approximation
                    
                del self.embeddings[key]
    
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Obtiene un embedding del pool."""
        with self.lock:
            # Actualizar estadísticas de acceso
            self.access_count[key] += 1
            self.last_access[key] = time.time()
            
            # Si está en memoria, devolverlo
            if key in self.embeddings:
                return self.embeddings[key][:] # Return a copy
            
            # Si lazy loading está habilitado, intentar cargar desde disco
            if self.lazy_loading and key in self.metadata:
                embedding = self._load_from_disk(key)
                if embedding is not None:
                    # Approximate memory usage of the decompressed embedding
                    embedding_size_approx = len(embedding) * 4 # Approximation for decompressed size

                    # Verificar si hay espacio en memoria
                    if self.memory_usage + embedding_size_approx > self.max_memory_bytes:
                        self._evict_least_used()
                    
                    self.embeddings[key] = embedding[:] # Store a copy
                    self.memory_usage += embedding_size_approx # Approximation
                    return embedding[:] # Return a copy
            
            return None
    
    def store_embedding(self, key: str, embedding: List[float], metadata: Dict = None):
        """Almacena un embedding en el pool."""
        with self.lock:
            if metadata is None:
                metadata = {}
            
            if not isinstance(embedding, list):
                raise ValueError("Embedding must be a list")
            
            # Approximate memory usage based on compression
            if self.compression_enabled:
                 # Rough approximation: size of list of ints + size of two floats
                embedding_size_approx = len(embedding) * 1 + 16
            else:
                embedding_size_approx = len(embedding) * 4 # Original Approximation

            # Verificar espacio en memoria
            if self.memory_usage + embedding_size_approx > self.max_memory_bytes:
                self._evict_least_used()
            
            # Almacenar en memoria
            self.embeddings[key] = embedding[:] # Store a copy
            self.metadata[key] = metadata
            self.memory_usage += embedding_size_approx
            
            # Actualizar estadísticas
            self.access_count[key] = 1
            self.last_access[key] = time.time()
            
            # Guardar a disco para persistencia
            self._save_to_disk(key, embedding, metadata)
    
    def precompute_common_embeddings(self, common_words: List[str],
                                   embedding_func, dim: int = 64):
        """Pre-calcula embeddings para palabras comunes."""
        print(f"Pre-calculando embeddings para {len(common_words)} palabras comunes...")
        
        for i, word in enumerate(common_words):
            if i % 100 == 0:
                print(f"Progreso: {i}/{len(common_words)}")
            
            key = f"word_{word}_{dim}"
            
            # Solo calcular si no existe
            if key not in self.metadata:
                try:
                    embedding = embedding_func(word, dim)
                    if embedding:
                        metadata = {
                            'word': word,
                            'dimension': dim,
                            'type': 'precomputed',
                            'timestamp': time.time()
                        }
                        self.store_embedding(key, embedding, metadata)
                        
                except Exception as e:
                    print(f"Error pre-calculando embedding para '{word}': {e}")
        
        print("Pre-cálculo completado.")
    
    def get_similar_embeddings(self, target_embedding: List[float], 
                               threshold: float = 0.8, max_results: int = 10) -> List[Tuple[str, float]]:
        """Encuentra embeddings similares en el pool."""
        if not isinstance(target_embedding, list):
            raise ValueError("Target embedding must be a list")
        
        results = []
        
        with self.lock:
            for key, embedding in self.embeddings.items():
                # Calcular similitud coseno
                similarity = self._cosine_similarity(target_embedding, embedding)
                
                if similarity >= threshold:
                    results.append((key, similarity))
        
        # Ordenar por similitud descendente
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno entre dos vectores."""
        # Verificar en caché primero
        cached = cache_manager.get_similarity(vec1, vec2)
        if cached is not None:
            return cached
        
        # Calcular similitud
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm1 * norm2)
        
        # Cachear resultado
        cache_manager.cache_similarity(vec1, vec2, similarity)
        
        return similarity
    
    def cleanup_old_embeddings(self, max_age_hours: int = 24):
        """Limpia embeddings antiguos del disco."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        removed_count = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                
                try:
                    # Verificar edad del archivo
                    file_age = current_time - os.path.getmtime(filepath)
                    
                    if file_age > max_age_seconds:
                        os.remove(filepath)
                        removed_count += 1
                        
                        # También remover de memoria si está cargado
                        key = filename[:-4]
                        if key in self.embeddings:
                            self.memory_usage -= len(self.embeddings[key]) * 4 # Approximation
                            del self.embeddings[key]
                        if key in self.metadata:
                            del self.metadata[key]
                            
                except Exception as e:
                    print(f"Error limpiando {filename}: {e}")
        
        print(f"Limpieza completada. Removidos {removed_count} embeddings antiguos.")
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del pool."""
        with self.lock:
            return {
                'embeddings_in_memory': len(self.embeddings),
                'embeddings_on_disk': len(self.metadata),
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self.memory_usage / self.max_memory_bytes,
                'cache_dir': self.cache_dir,
                'compression_enabled': self.compression_enabled,
                'lazy_loading': self.lazy_loading,
                'total_accesses': sum(self.access_count.values())
            }
    
    def optimize(self):
        """Optimiza el pool liberando memoria y limpiando caché."""
        with self.lock:
            # Limpiar embeddings no accedidos recientemente
            current_time = time.time()
            to_remove = []
            
            for key in self.embeddings:
                last_access = self.last_access.get(key, 0)
                if current_time - last_access > 3600:  # 1 hora sin acceso
                    to_remove.append(key)
            
            for key in to_remove:
                self._save_to_disk(key, self.embeddings[key], self.metadata.get(key, {}))
                self.memory_usage -= len(self.embeddings[key]) * 4 # Approximation
                del self.embeddings[key]
            
            print(f"Optimización completada. Liberados {len(to_remove)} embeddings de memoria.")


# Instancia global del pool de embeddings
embedding_pool = EmbeddingPool()