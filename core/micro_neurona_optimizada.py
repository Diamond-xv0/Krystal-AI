"""
MicroNeurona Optimizada para Krystal AI
Versión mejorada con caché, índices vectoriales y paralelización.
"""

import math
import random
import unicodedata
import asyncio
import time
import math
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from .cache_manager import cache_manager
from .embedding_pool import embedding_pool
from .indices_vectoriales import embedding_index


class MicroNeuronaOptimizada:
    """MicroNeurona optimizada con caché, índices y paralelización."""
    
    def __init__(self, id: str, concepto: str, tipo: str, 
                 embedding: Optional[List[float]] = None, metadata: Optional[Dict] = None):
        self.id = id
        self.concepto = concepto
        self.tipo = tipo
        
        # Metadata enriquecida
        self.metadata = metadata if metadata is not None else {}
        
        # Estado de activación
        self.activa = False
        self.confianza = 1.0
        self.historial_activacion = []
        
        # Optimizaciones
        self.embedding_cache_key = f"embedding_{concepto}_{64}"
        self.similarity_cache = {}
        self.last_activation_time = 0
        
        # Memoria episódica optimizada
        self.memoria_episodica = []
        self.max_memoria_episodica = 100
        
        # Inicializar embedding
        self.embedding = self._get_or_compute_embedding(concepto, embedding)
        
        # Registrar en índice vectorial si no existe
        self._register_in_index()
    
    def _get_or_compute_embedding(self, concepto: str, provided_embedding: Optional[List[float]]) -> List[float]:
        """Obtiene embedding del pool o lo calcula si es necesario."""
        if provided_embedding is not None:
            # Cachear el embedding proporcionado
            embedding_pool.store_embedding(
                self.embedding_cache_key, 
                provided_embedding,
                {
                    'concepto': concepto,
                    'tipo': self.tipo,
                    'timestamp': time.time()
                }
            )
            return provided_embedding
        
        # Intentar obtener del pool
        cached_embedding = embedding_pool.get_embedding(self.embedding_cache_key)
        if cached_embedding is not None:
            # Devuelve como lista, sea array o lista nativa
            if hasattr(cached_embedding, "tolist"):
                return cached_embedding.tolist()
            return cached_embedding
        
        # Calcular nuevo embedding
        embedding = self.calcular_embedding(concepto, dim=64)
        
        # Almacenar en pool
        embedding_pool.store_embedding(
            self.embedding_cache_key,
            embedding,
            {
                'concepto': concepto,
                'tipo': self.tipo,
                'timestamp': time.time()
            }
        )
        
        return embedding
    
    def _register_in_index(self):
        """Registra la neurona en el índice vectorial."""
        try:
            # Determinar categoría basada en metadata
            category = self.metadata.get('semantic_field', self.tipo)
            
            # Metadata para el índice
            index_metadata = {
                'neurona_id': self.id,
                'concepto': self.concepto,
                'tipo': self.tipo,
                'categoria': category,
                'timestamp': time.time()
            }
            
            # Añadir al índice
            embedding_index.add_vector(
                self.id,
                self.embedding,
                category=category,
                metadata=index_metadata
            )
            
        except Exception as e:
            print(f"Error registrando neurona {self.id} en índice: {e}")
    
    def normalizar(self, texto: str) -> str:
        """Normaliza texto de forma optimizada."""
        if not texto:
            return ""
        
        # Usar caché para normalizaciones frecuentes
        cache_key = f"normalize_{texto}"
        cached = cache_manager.embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Normalización
        texto = texto.lower()
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) 
                       if unicodedata.category(c) != 'Mn')
        texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
        texto = texto.strip()
        
        # Cachear resultado
        cache_manager.embedding_cache.put(cache_key, texto)
        
        return texto
    
    def calcular_embedding(self, texto: str, dim: int = 64) -> List[float]:
        """Calcula embedding optimizado con caché."""
        # Verificar caché primero
        cached = cache_manager.get_embedding(texto, dim)
        if cached is not None:
            return cached
        
        # Calcular embedding
        embedding = self._compute_embedding_internal(texto, dim)
        
        # Cachear resultado
        cache_manager.cache_embedding(texto, dim, embedding)
        
        return embedding
    
    def _compute_embedding_internal(self, texto: str, dim: int = 64) -> List[float]:
        """Cálculo interno del embedding con optimizaciones."""
        def _generar_ngrams(palabra: str, min_n: int = 2, max_n: int = 5) -> List[str]:
            """Genera n-grams optimizado."""
            if not palabra:
                return []
            
            ngrams = {palabra}  # Usar set para evitar duplicados
            
            # Prefijos y sufijos
            for i in range(1, min(len(palabra), max_n + 1)):
                ngrams.add(palabra[:i])
                ngrams.add(palabra[-i:])
            
            # N-grams internos
            for n in range(min_n, max_n + 1):
                for i in range(len(palabra) - n + 1):
                    ngrams.add(palabra[i:i+n])
            
            return list(ngrams)
        
        texto = self.normalizar(texto)
        if not texto:
            return [0.0] * dim
        
        # Usar listas para cálculos
        vec = [0.0] * dim
        
        ngrams = _generar_ngrams(texto)
        
        for ngram in ngrams:
            # Hash optimizado
            seed = abs(hash(ngram)) % (2**32)
            rnd = random.Random(seed)
            
            # Generar contribución del n-gram
            contribution = [rnd.uniform(-1, 1) for _ in range(dim)]
            for i in range(dim):
                vec[i] += contribution[i]
        
        # Normalización
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        
        return vec
    
    def activar(self, vectores_entrada: List[List[float]], 
               frase_original: Optional[str] = None, umbral: float = 0.7) -> bool:
        """Activación optimizada con caché."""
        current_time = time.time()
        self.last_activation_time = current_time
        
        # Verificar caché de activación
        cached_result = cache_manager.get_activation(
            self.id, vectores_entrada, frase_original or "", umbral
        )
        if cached_result is not None:
            self.activa, self.confianza = cached_result
            self.historial_activacion.append((self.confianza, self.activa, 'CACHED'))
            return self.activa
        
        # Activación por nombre (alta prioridad)
        if frase_original and self.concepto:
            concepto_norm = self.normalizar(self.concepto)
            frase_norm = self.normalizar(frase_original)
            
            if concepto_norm in frase_norm:
                self.activa = True
                self.confianza = 1.0
                self.historial_activacion.append((self.confianza, True, 'NOMBRE'))
                
                # Cachear resultado
                cache_manager.cache_activation(
                    self.id, vectores_entrada, frase_original or "", umbral, 
                    (self.activa, self.confianza)
                )
                return True
        
        # Activación por similitud vectorial
        if not vectores_entrada:
            self.activa = False
            self.confianza = 0.0
            return False
        
        # Usar búsqueda vectorial optimizada
        max_sim = self._compute_max_similarity_optimized(vectores_entrada)
        
        self.confianza = max_sim
        self.activa = self.confianza >= umbral
        self.historial_activacion.append((self.confianza, self.activa, 'VECTORIAL'))
        
        # Cachear resultado
        cache_manager.cache_activation(
            self.id, vectores_entrada, frase_original or "", umbral,
            (self.activa, self.confianza)
        )
        
        # Actualizar memoria episódica
        self._update_episodic_memory(vectores_entrada, frase_original, self.confianza)
        
        return self.activa
    
    def _compute_max_similarity_optimized(self, vectores_entrada: List[List[float]]) -> float:
        """Calcula similitud máxima de forma optimizada."""
        max_sim = 0.0
        
        for vec_entrada in vectores_entrada:
            # Verificar caché de similitud
            cached_sim = cache_manager.get_similarity(self.embedding, vec_entrada)
            if cached_sim is not None:
                sim = cached_sim
            else:
                # Calcular similitud
                sim = self._cosine_similarity(self.embedding, vec_entrada)
                
                # Cachear similitud
                cache_manager.cache_similarity(self.embedding, vec_entrada, sim)
            
            if sim > max_sim:
                max_sim = sim
        
        return max_sim
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Similitud coseno optimizada."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a*a for a in vec1))
        norm2 = math.sqrt(sum(b*b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _update_episodic_memory(self, vectores_entrada: List[List[float]], 
                               frase_original: Optional[str], confianza: float):
        """Actualiza memoria episódica de forma eficiente."""
        if len(self.memoria_episodica) >= self.max_memoria_episodica:
            # Remover el más antiguo
            self.memoria_episodica.pop(0)
        
        episodio = {
            'timestamp': time.time(),
            'confianza': confianza,
            'frase': frase_original,
            'num_vectores': len(vectores_entrada),
            'activada': self.activa
        }
        
        self.memoria_episodica.append(episodio)
    
    async def activar_async(self, vectores_entrada: List[List[float]], 
                           frase_original: Optional[str] = None, umbral: float = 0.7) -> bool:
        """Versión asíncrona de activación para paralelización."""
        # Ejecutar activación en thread pool para operaciones CPU-intensivas
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, self.activar, vectores_entrada, frase_original, umbral
            )
        return result
    
    def buscar_similares(self, k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Busca micro-neuronas similares usando el índice vectorial."""
        return []
    
    def get_activation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de activación."""
        if not self.historial_activacion:
            return {
                'total_activaciones': 0,
                'promedio_confianza': 0.0,
                'tasa_activacion': 0.0,
                'ultimo_acceso': 0
            }
        
        total = len(self.historial_activacion)
        activaciones_exitosas = sum(1 for _, activa, _ in self.historial_activacion if activa)
        promedio_confianza = sum(conf for conf, _, _ in self.historial_activacion) / total
        
        return {
            'total_activaciones': total,
            'activaciones_exitosas': activaciones_exitosas,
            'promedio_confianza': promedio_confianza,
            'tasa_activacion': activaciones_exitosas / total,
            'ultimo_acceso': self.last_activation_time
        }
    
    def optimize_memory(self):
        """Optimiza el uso de memoria de la neurona."""
        # Limpiar historial antiguo
        if len(self.historial_activacion) > 1000:
            self.historial_activacion = self.historial_activacion[-500:]
        
        # Limpiar memoria episódica antigua
        current_time = time.time()
        self.memoria_episodica = [
            ep for ep in self.memoria_episodica 
            if current_time - ep['timestamp'] < 3600  # Mantener solo última hora
        ]
        
        # Limpiar caché de similitudes local
        self.similarity_cache.clear()
    
    def reset(self):
        """Resetea el estado de activación."""
        self.activa = False
        self.confianza = 0.0
        # No limpiar historial para mantener estadísticas
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la neurona a diccionario para serialización."""
        return {
            'id': self.id,
            'concepto': self.concepto,
            'tipo': self.tipo,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'activa': self.activa,
            'confianza': self.confianza,
            'stats': self.get_activation_stats()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MicroNeuronaOptimizada':
        """Crea una neurona desde un diccionario."""
        neurona = cls(
            id=data['id'],
            concepto=data['concepto'],
            tipo=data['tipo'],
            embedding=data.get('embedding'),
            metadata=data.get('metadata', {})
        )
        
        neurona.activa = data.get('activa', False)
        neurona.confianza = data.get('confianza', 0.0)
        
        return neurona
    
    def __repr__(self) -> str:
        return f"MicroNeuronaOpt(id='{self.id}', concepto='{self.concepto}', activa={self.activa})"


# Funciones de utilidad para migración
def migrate_from_old_microneurona(old_mn) -> MicroNeuronaOptimizada:
    """Migra una MicroNeurona antigua a la versión optimizada."""
    return MicroNeuronaOptimizada(
        id=old_mn.id,
        concepto=old_mn.concepto,
        tipo=old_mn.tipo,
        embedding=old_mn.embedding,
        metadata=getattr(old_mn, 'metadata', {})
    )


def batch_activate_neurons(neurons: List[MicroNeuronaOptimizada], 
                          vectores_entrada: List[List[float]],
                          frase_original: Optional[str] = None,
                          umbral: float = 0.7) -> List[bool]:
    """Activa múltiples neuronas en paralelo."""
    async def activate_all():
        tasks = [
            neuron.activar_async(vectores_entrada, frase_original, umbral)
            for neuron in neurons
        ]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(activate_all())