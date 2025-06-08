"""
Razonador Optimizado para Krystal AI
Versión mejorada con paralelización, caché y gestión eficiente de memoria.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import math

from .micro_neurona_optimizada import MicroNeuronaOptimizada, batch_activate_neurons
from .neurona import Neurona
from .macro_neurona import MacroNeurona
from .cache_manager import cache_manager
from .embedding_pool import embedding_pool
from .indices_vectoriales import index_manager
from .MemoryNs import registrar_memoria


class RazonadorOptimizado:
    """Razonador optimizado con paralelización y gestión inteligente de memoria."""
    
    def __init__(self, memoria, personalidad, max_workers: int = 4):
        self.memoria = memoria
        self.personalidad = personalidad
        
        # Almacenamiento neuronal optimizado
        self.micro_neuronas: Dict[str, MicroNeuronaOptimizada] = {}
        self.neuronas: Dict[str, Neurona] = {}
        self.macro_neuronas: Dict[str, MacroNeurona] = {}
        
        # Vocabulario unificado optimizado
        self.vocabulario_palabras_clave: List[MicroNeuronaOptimizada] = []
        
        # Índices para búsquedas rápidas
        self.neuronas_por_categoria: Dict[str, Set[str]] = defaultdict(set)
        self.neuronas_por_tipo: Dict[str, Set[str]] = defaultdict(set)
        
        # Paralelización
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Estadísticas y monitoreo
        self.historial_ciclos = []
        self.stats = {
            'total_ciclos': 0,
            'tiempo_total_activacion': 0.0,
            'tiempo_total_evaluacion': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Control de memoria
        self.max_historial_ciclos = 1000
        self.last_optimization = time.time()
        self.optimization_interval = 300  # 5 minutos
        
        # Threading
        self.lock = threading.RLock()
    
    def registrar_micro_neurona(self, mn: MicroNeuronaOptimizada):
        """Registra una micro-neurona optimizada."""
        with self.lock:
            self.micro_neuronas[mn.id] = mn
            
            # Indexar por categoría y tipo
            categoria = mn.metadata.get('semantic_field', mn.tipo)
            self.neuronas_por_categoria[categoria].add(mn.id)
            self.neuronas_por_tipo[mn.tipo].add(mn.id)
            
            # Añadir al vocabulario si es palabra clave
            if mn.tipo == 'palabra_clave':
                self.vocabulario_palabras_clave.append(mn)
    
    def registrar_neurona(self, n: Neurona):
        """Registra una neurona."""
        with self.lock:
            self.neuronas[n.id] = n
    
    def registrar_macro_neurona(self, macro_n: MacroNeurona):
        """Registra una macro-neurona."""
        with self.lock:
            self.macro_neuronas[macro_n.id] = macro_n
    
    def reset(self):
        """Resetea el estado de activación de todas las neuronas."""
        with self.lock:
            # Reset paralelo de micro-neuronas
            if len(self.micro_neuronas) > 100:
                self._reset_parallel()
            else:
                for mn in self.micro_neuronas.values():
                    mn.reset()
            
            # Reset de neuronas y macro-neuronas
            for n in self.neuronas.values():
                n.reset()
            for macro_n in self.macro_neuronas.values():
                macro_n.reset()
    
    def _reset_parallel(self):
        """Reset paralelo para grandes cantidades de neuronas."""
        def reset_batch(neurons_batch):
            for neuron in neurons_batch:
                neuron.reset()
        
        # Dividir en lotes
        neurons_list = list(self.micro_neuronas.values())
        batch_size = len(neurons_list) // self.max_workers + 1
        batches = [neurons_list[i:i + batch_size] 
                  for i in range(0, len(neurons_list), batch_size)]
        
        # Ejecutar en paralelo
        futures = [self.thread_pool.submit(reset_batch, batch) for batch in batches]
        for future in as_completed(futures):
            future.result()  # Esperar completación
    
    def ciclo_activacion(self, vectores_entrada: List[List[float]], 
                        frase_original: Optional[str] = None, 
                        umbral_mn: float = 0.7) -> Dict[str, Any]:
        """Ciclo de activación optimizado con paralelización."""
        start_time = time.time()
        
        with self.lock:
            self.stats['total_ciclos'] += 1
            
            # Optimización automática periódica
            if time.time() - self.last_optimization > self.optimization_interval:
                self._optimize_memory()
            
            # Activación paralela de micro-neuronas
            if len(self.vocabulario_palabras_clave) > 50:
                resultados = self._activar_paralelo(vectores_entrada, frase_original, umbral_mn)
            else:
                resultados = self._activar_secuencial(vectores_entrada, frase_original, umbral_mn)
            
            # Guardar historial optimizado
            ciclo_info = {
                'timestamp': start_time,
                'micro_activas_iniciales': {mn.id: mn.activa for mn in self.vocabulario_palabras_clave if mn.activa},
                'tiempo_activacion': time.time() - start_time,
                'vectores_entrada_count': len(vectores_entrada),
                'frase_original': frase_original
            }
            
            self._add_to_history(ciclo_info)
            
            # Registrar en memoria
            registrar_memoria({"ciclo_activacion_inicial": ciclo_info})
            
            self.stats['tiempo_total_activacion'] += ciclo_info['tiempo_activacion']
            
            return ciclo_info
    
    def _activar_paralelo(self, vectores_entrada: List[List[float]], 
                         frase_original: Optional[str], umbral_mn: float) -> List[bool]:
        """Activación paralela para grandes vocabularios."""
        # Usar la función batch optimizada
        return batch_activate_neurons(
            self.vocabulario_palabras_clave,
            vectores_entrada,
            frase_original,
            umbral_mn
        )
    
    def _activar_secuencial(self, vectores_entrada: List[List[float]], 
                           frase_original: Optional[str], umbral_mn: float) -> List[bool]:
        """Activación secuencial para vocabularios pequeños."""
        resultados = []
        for mn in self.vocabulario_palabras_clave:
            resultado = mn.activar(vectores_entrada, frase_original, umbral_mn)
            resultados.append(resultado)
        return resultados
    
    def evaluar_neuronas_paralelo(self, conceptos_activos: Dict[str, float]) -> Dict[str, Tuple[bool, float]]:
        """Evaluación paralela de neuronas."""
        start_time = time.time()
        
        if len(self.neuronas) <= 10:
            # Evaluación secuencial para pocos elementos
            resultados = {}
            for n in self.neuronas.values():
                # Verificar caché primero
                cached = cache_manager.get_evaluation(n.id, conceptos_activos)
                if cached is not None:
                    n.activa, n.confianza = cached
                    resultados[n.id] = cached
                    self.stats['cache_hits'] += 1
                else:
                    n.evaluar(conceptos_activos)
                    resultado = (n.activa, n.confianza)
                    cache_manager.cache_evaluation(n.id, conceptos_activos, resultado)
                    resultados[n.id] = resultado
                    self.stats['cache_misses'] += 1
            
            self.stats['tiempo_total_evaluacion'] += time.time() - start_time
            return resultados
        
        # Evaluación paralela
        def evaluar_neurona(neurona):
            # Verificar caché
            cached = cache_manager.get_evaluation(neurona.id, conceptos_activos)
            if cached is not None:
                neurona.activa, neurona.confianza = cached
                return neurona.id, cached, True  # True indica cache hit
            else:
                neurona.evaluar(conceptos_activos)
                resultado = (neurona.activa, neurona.confianza)
                cache_manager.cache_evaluation(neurona.id, conceptos_activos, resultado)
                return neurona.id, resultado, False  # False indica cache miss
        
        # Ejecutar en paralelo
        futures = [self.thread_pool.submit(evaluar_neurona, neurona) 
                  for neurona in self.neuronas.values()]
        
        resultados = {}
        for future in as_completed(futures):
            neurona_id, resultado, cache_hit = future.result()
            resultados[neurona_id] = resultado
            if cache_hit:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
        
        self.stats['tiempo_total_evaluacion'] += time.time() - start_time
        return resultados
    
    def buscar_neuronas_similares(self, concepto: str, k: int = 10, 
                                 threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Busca micro-neuronas similares usando índices vectoriales."""
        if concepto not in self.micro_neuronas:
            return []
        
        mn = self.micro_neuronas[concepto]
        return mn.buscar_similares(k=k, threshold=threshold)
    
    def get_neuronas_por_categoria(self, categoria: str) -> List[MicroNeuronaOptimizada]:
        """Obtiene neuronas por categoría de forma eficiente."""
        with self.lock:
            neurona_ids = self.neuronas_por_categoria.get(categoria, set())
            return [self.micro_neuronas[nid] for nid in neurona_ids 
                   if nid in self.micro_neuronas]
    
    def get_estadisticas_activacion(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de activación."""
        with self.lock:
            total_mn = len(self.micro_neuronas)
            activas_mn = sum(1 for mn in self.micro_neuronas.values() if mn.activa)
            
            total_n = len(self.neuronas)
            activas_n = sum(1 for n in self.neuronas.values() if n.activa)
            
            cache_total = self.stats['cache_hits'] + self.stats['cache_misses']
            cache_hit_rate = self.stats['cache_hits'] / cache_total if cache_total > 0 else 0
            
            avg_activation_time = (self.stats['tiempo_total_activacion'] / 
                                 self.stats['total_ciclos'] if self.stats['total_ciclos'] > 0 else 0)
            
            avg_evaluation_time = (self.stats['tiempo_total_evaluacion'] / 
                                 self.stats['total_ciclos'] if self.stats['total_ciclos'] > 0 else 0)
            
            return {
                'micro_neuronas': {
                    'total': total_mn,
                    'activas': activas_mn,
                    'tasa_activacion': activas_mn / total_mn if total_mn > 0 else 0
                },
                'neuronas': {
                    'total': total_n,
                    'activas': activas_n,
                    'tasa_activacion': activas_n / total_n if total_n > 0 else 0
                },
                'rendimiento': {
                    'total_ciclos': self.stats['total_ciclos'],
                    'tiempo_promedio_activacion_ms': avg_activation_time * 1000,
                    'tiempo_promedio_evaluacion_ms': avg_evaluation_time * 1000,
                    'cache_hit_rate': cache_hit_rate,
                    'cache_hits': self.stats['cache_hits'],
                    'cache_misses': self.stats['cache_misses']
                },
                'memoria': {
                    'historial_ciclos': len(self.historial_ciclos),
                    'categorias': len(self.neuronas_por_categoria),
                    'tipos': len(self.neuronas_por_tipo)
                }
            }
    
    def _add_to_history(self, ciclo_info: Dict[str, Any]):
        """Añade información al historial con gestión de memoria."""
        self.historial_ciclos.append(ciclo_info)
        
        # Mantener solo los últimos N ciclos
        if len(self.historial_ciclos) > self.max_historial_ciclos:
            self.historial_ciclos = self.historial_ciclos[-self.max_historial_ciclos//2:]
    
    def _optimize_memory(self):
        """Optimiza el uso de memoria del razonador."""
        print("Optimizando memoria del razonador...")
        
        # Optimizar micro-neuronas
        for mn in self.micro_neuronas.values():
            mn.optimize_memory()
        
        # Optimizar cachés
        cache_manager.optimize_memory()
        embedding_pool.optimize()
        
        # Optimizar índices
        index_manager.optimize_all()
        
        # Limpiar historial antiguo
        current_time = time.time()
        self.historial_ciclos = [
            ciclo for ciclo in self.historial_ciclos
            if current_time - ciclo['timestamp'] < 3600  # Mantener última hora
        ]
        
        self.last_optimization = current_time
        print("Optimización de memoria completada.")
    
    def precomputar_embeddings_comunes(self, palabras_comunes: List[str]):
        """Pre-computa embeddings para palabras comunes."""
        print(f"Pre-computando embeddings para {len(palabras_comunes)} palabras...")
        
        def compute_embedding(word):
            mn_temp = MicroNeuronaOptimizada(f"temp_{word}", word, "temp")
            return word, mn_temp.embedding
        
        # Computar en paralelo
        futures = [self.thread_pool.submit(compute_embedding, word) 
                  for word in palabras_comunes]
        
        for future in as_completed(futures):
            word, embedding = future.result()
            # El embedding ya se almacena automáticamente en el pool
        
        print("Pre-cómputo completado.")
    
    def exportar_estadisticas(self) -> Dict[str, Any]:
        """Exporta estadísticas completas para análisis."""
        stats = self.get_estadisticas_activacion()
        
        # Añadir estadísticas de componentes
        stats['cache_manager'] = cache_manager.get_global_stats()
        stats['embedding_pool'] = embedding_pool.get_stats()
        stats['indices'] = index_manager.get_all_stats()
        
        return stats
    
    def cleanup(self):
        """Limpia recursos y cierra pools de threads."""
        self.thread_pool.shutdown(wait=True)
        cache_manager.clear_all_caches()
        print("Razonador optimizado limpiado.")
    
    def __del__(self):
        """Destructor para limpiar recursos."""
        try:
            self.cleanup()
        except:
            pass


# Función de migración desde razonador original
def migrar_desde_razonador_original(razonador_original, max_workers: int = 4) -> RazonadorOptimizado:
    """Migra un razonador original al optimizado."""
    razonador_opt = RazonadorOptimizado(
        razonador_original.memoria,
        razonador_original.personalidad,
        max_workers=max_workers
    )
    
    # Migrar micro-neuronas
    for mn_id, mn_original in razonador_original.micro_neuronas.items():
        mn_optimizada = MicroNeuronaOptimizada(
            mn_original.id,
            mn_original.concepto,
            mn_original.tipo,
            mn_original.embedding,
            getattr(mn_original, 'metadata', {})
        )
        razonador_opt.registrar_micro_neurona(mn_optimizada)
    
    # Migrar neuronas y macro-neuronas (sin cambios)
    for n_id, neurona in razonador_original.neuronas.items():
        razonador_opt.registrar_neurona(neurona)
    
    for mn_id, macro_neurona in razonador_original.macro_neuronas.items():
        razonador_opt.registrar_macro_neurona(macro_neurona)
    
    return razonador_opt