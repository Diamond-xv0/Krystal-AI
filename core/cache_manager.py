"""
Sistema de Caché Inteligente para Krystal AI
Optimiza el rendimiento mediante caché LRU de similitudes y activaciones.
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import pickle


class LRUCache:
    """Implementación de caché LRU thread-safe con TTL opcional."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl  # Time to live en segundos
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        """Verifica si una entrada ha expirado."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamps.get(key, 0) > self.ttl
    
    def _cleanup_expired(self):
        """Limpia entradas expiradas."""
        if self.ttl is None:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché."""
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                self.misses += 1
                if key in self.cache:
                    # Remover entrada expirada
                    del self.cache[key]
                    del self.timestamps[key]
                return None
            
            # Mover al final (más reciente)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
    
    def put(self, key: str, value: Any):
        """Almacena un valor en el caché."""
        with self.lock:
            # Limpiar expirados ocasionalmente
            if len(self.cache) % 100 == 0:
                self._cleanup_expired()
            
            if key in self.cache:
                # Actualizar valor existente
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remover el más antiguo
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Limpia todo el caché."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class CacheManager:
    """Gestor central de caché para el sistema Krystal AI."""
    
    def __init__(self):
        # Caché para similitudes coseno
        self.similarity_cache = LRUCache(max_size=10000, ttl=3600)  # 1 hora TTL
        
        # Caché para embeddings calculados
        self.embedding_cache = LRUCache(max_size=5000, ttl=7200)  # 2 horas TTL
        
        # Caché para activaciones de micro-neuronas
        self.activation_cache = LRUCache(max_size=2000, ttl=1800)  # 30 min TTL
        
        # Caché para resultados de evaluación de neuronas
        self.evaluation_cache = LRUCache(max_size=1000, ttl=1800)  # 30 min TTL
        
        # Estadísticas globales
        self.start_time = time.time()
    
    def _generate_key(self, *args) -> str:
        """Genera una clave única para los argumentos dados."""
        # Serializar argumentos y crear hash
        serialized = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(serialized).hexdigest()
    
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> Optional[float]:
        """Obtiene similitud coseno del caché.
        Asegura que vector1 y vector2 sean secuencias antes de convertirlas a tuple.
        Si alguno es float, lo encapsula en una lista. Si no es secuencia ni float, lanza TypeError.
        """
        def ensure_sequence(vec, name):
            if isinstance(vec, float):
                vec = [vec]
            elif not hasattr(vec, '__iter__'):
                raise TypeError(f"{name} debe ser una secuencia (list, tuple, array) o float, no {type(vec).__name__}")
            return vec
        vector1 = ensure_sequence(vector1, "vector1")
        vector2 = ensure_sequence(vector2, "vector2")
        key = self._generate_key('similarity', tuple(vector1), tuple(vector2))
        return self.similarity_cache.get(key)
    
    def cache_similarity(self, vector1: List[float], vector2: List[float], similarity: float):
        """Almacena similitud coseno en caché."""
        key = self._generate_key('similarity', tuple(vector1), tuple(vector2))
        self.similarity_cache.put(key, similarity)
        
        # También cachear la similitud inversa (vector2, vector1)
        key_inverse = self._generate_key('similarity', tuple(vector2), tuple(vector1))
        self.similarity_cache.put(key_inverse, similarity)
    
    def get_embedding(self, texto: str, dim: int) -> Optional[List[float]]:
        """Obtiene embedding del caché."""
        key = self._generate_key('embedding', texto, dim)
        return self.embedding_cache.get(key)
    
    def cache_embedding(self, texto: str, dim: int, embedding: List[float]):
        """Almacena embedding en caché."""
        key = self._generate_key('embedding', texto, dim)
        self.embedding_cache.put(key, embedding)
    
    def get_activation(self, mn_id: str, vectores_entrada: List[List[float]], 
                      frase_original: str, umbral: float) -> Optional[Tuple[bool, float]]:
        """Obtiene resultado de activación del caché."""
        key = self._generate_key('activation', mn_id, 
                                [tuple(v) for v in vectores_entrada], 
                                frase_original, umbral)
        return self.activation_cache.get(key)
    
    def cache_activation(self, mn_id: str, vectores_entrada: List[List[float]], 
                        frase_original: str, umbral: float, 
                        result: Tuple[bool, float]):
        """Almacena resultado de activación en caché."""
        key = self._generate_key('activation', mn_id, 
                                [tuple(v) for v in vectores_entrada], 
                                frase_original, umbral)
        self.activation_cache.put(key, result)
    
    def get_evaluation(self, neurona_id: str, conceptos_activos: Dict[str, float]) -> Optional[Tuple[bool, float]]:
        """Obtiene resultado de evaluación del caché."""
        # Ordenar conceptos para consistencia en la clave
        conceptos_sorted = tuple(sorted(conceptos_activos.items()))
        key = self._generate_key('evaluation', neurona_id, conceptos_sorted)
        return self.evaluation_cache.get(key)
    
    def cache_evaluation(self, neurona_id: str, conceptos_activos: Dict[str, float], 
                        result: Tuple[bool, float]):
        """Almacena resultado de evaluación en caché."""
        conceptos_sorted = tuple(sorted(conceptos_activos.items()))
        key = self._generate_key('evaluation', neurona_id, conceptos_sorted)
        self.evaluation_cache.put(key, result)
    
    def invalidate_neuron_caches(self, neurona_id: str):
        """Invalida cachés relacionados con una neurona específica."""
        # Para una implementación más sofisticada, podríamos mantener
        # un índice inverso de qué claves están relacionadas con cada neurona
        # Por ahora, simplemente limpiamos los cachés de activación y evaluación
        self.activation_cache.clear()
        self.evaluation_cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales del sistema de caché."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'similarity_cache': self.similarity_cache.get_stats(),
            'embedding_cache': self.embedding_cache.get_stats(),
            'activation_cache': self.activation_cache.get_stats(),
            'evaluation_cache': self.evaluation_cache.get_stats(),
            'total_memory_entries': (
                len(self.similarity_cache.cache) +
                len(self.embedding_cache.cache) +
                len(self.activation_cache.cache) +
                len(self.evaluation_cache.cache)
            )
        }
    
    def clear_all_caches(self):
        """Limpia todos los cachés."""
        self.similarity_cache.clear()
        self.embedding_cache.clear()
        self.activation_cache.clear()
        self.evaluation_cache.clear()
    
    def optimize_memory(self):
        """Optimiza el uso de memoria limpiando cachés según prioridad."""
        # Limpiar primero los cachés menos críticos
        self.activation_cache._cleanup_expired()
        self.evaluation_cache._cleanup_expired()
        self.similarity_cache._cleanup_expired()
        self.embedding_cache._cleanup_expired()


# Instancia global del gestor de caché
cache_manager = CacheManager()