# --- ARCHIVO COMPLETO Y FINAL: core/micro_neurona.py ---

import math
import random
import unicodedata
from core.indices_vectoriales import VectorIndex
from core.cache_manager import cache_manager

from typing import Tuple, List, Dict, Any

class MicroNeurona:
    def __init__(self, id, concepto, tipo, embedding=None, metadata=None, decay_rate=0.15, umbral_activacion=0.7): # Added umbral_activacion
        self.id = id
        self.concepto = concepto
        self.tipo = tipo
        # El embedding principal es puramente semántico, basado en el concepto.
        self.embedding = embedding if embedding is not None else self.calcular_embedding(concepto, dim=64)
        # La metadata contiene toda la riqueza dimensional y gramatical.
        self.metadata = metadata if metadata is not None else {}

        self.activa = False # Boolean active state (derived from activation_level)
        self.activation_level = 0.0 # Continuous activation level
        self.decay_rate = decay_rate # Rate at which activation decays per iteration
        self.umbral_activacion = umbral_activacion # Activation threshold for this micro-neuron
        self.historial_activacion = []
        self.confianza = 1.0 # This might become redundant or used differently with activation_level
        self.historial_embeddings = [self.embedding[:]]
        self.mini_chain_of_thought = []
        self.memoria_episodica = []

    def normalizar(self, texto):
        texto = texto.lower()
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
        texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
        texto = texto.strip()
        return texto

    def calcular_embedding(self, texto, dim=64):
        
        def _generar_ngrams(palabra, min_n=2, max_n=5):
            """Genera n-grams de caracteres para una palabra, incluyendo la palabra misma."""
            # Añadimos la palabra original para que siempre tenga un vector base
            ngrams = {palabra}
            # Añadimos prefijos y sufijos para capturar inicios y finales
            for i in range(1, min(len(palabra), max_n + 1)):
                ngrams.add(palabra[:i]) # Prefijos
                ngrams.add(palabra[-i:]) # Sufijos

            for n in range(min_n, max_n + 1):
                for i in range(len(palabra) - n + 1):
                    ngrams.add(palabra[i:i+n])
            return list(ngrams)

        texto = self.normalizar(texto)
        if not texto:
            return [0.0] * dim

        vec = [0.0] * dim
        
        # Usamos n-grams de la palabra completa para capturar subestructuras
        ngrams = _generar_ngrams(texto)
        
        for ngram in ngrams:
            # Usamos el hash del n-gram para la semilla
            seed = abs(hash(ngram)) % (2**32)
            rnd = random.Random(seed)
            for j in range(dim):
                # La contribución de cada n-gram se suma al vector
                vec[j] += rnd.uniform(-1, 1)

        # Normalizamos el vector final para tener una magnitud constante
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def activar(self, vectores_entrada, frase_original=None, umbral=None, activation_fn=None):
        """
        Activa la micro-neurona con función de activación y umbral personalizables.
        activation_fn: callable(float) -> float, e.g. sigmoid, relu, tanh. Por defecto sigmoid.
        umbral: si None, usa self.umbral_activacion.
        """
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        def relu(x):
            return max(0.0, x)
        def tanh(x):
            return math.tanh(x)

        # Selección de función de activación
        fn = activation_fn or sigmoid

        initial_activation = 0.0
        activation_reason = "NONE"

        # Activación por nombre (para comprensión) - alta prioridad
        if frase_original and self.concepto:
            if self.normalizar(self.concepto) in self.normalizar(frase_original):
                initial_activation = 1.0 # High activation for direct match
                activation_reason = 'NM'

        # Activación por Foco de Atención (Embedding) - if not ya altamente activada
        if initial_activation < 1.0 and vectores_entrada:
            max_sim = 0.0
            for i, vec_entrada in enumerate(vectores_entrada):
                sim = self.similitud_coseno(self.embedding, vec_entrada)
                if sim > max_sim:
                    max_sim = sim

            if max_sim > initial_activation: # Use embedding similarity if higher
                initial_activation = max_sim
                activation_reason = 'EMB_FOCUS'

        # Aplicar función de activación personalizada
        activated_value = fn(initial_activation)

        # Determinar umbral dinámico
        threshold = umbral if umbral is not None else self.umbral_activacion

        self.activation_level = activated_value
        self.activa = self.activation_level >= threshold

        # Update history, registrar función usada
        self.historial_activacion.append((self.activation_level, self.activa, activation_reason, fn.__name__))

        print(f"DEBUG: MicroNeurona {self.id} - Activation: {self.activation_level}, Active: {self.activa}, Reason: {activation_reason}, Func: {fn.__name__}, Threshold: {threshold}")
        return self.activa # Return boolean active state
    async def activar_async(self, vectores_entrada, frase_original=None, umbral=0.7):
        """
        Asynchronous method to activate the micro-neuron.
        Wraps the existing activar logic for parallel execution.
        """
        # The existing activar method is synchronous, so we can just call it directly
        # within the async method. If it involved I/O or blocking operations,
        # we would use asyncio.to_thread or similar.
        return self.activar(vectores_entrada, frase_original, umbral)

    @staticmethod
    def similitud_coseno(vec1, vec2):
        # Encapsula floats en listas o lanza error si no es secuencia ni float
        def ensure_sequence(v, name):
            if isinstance(v, float) or isinstance(v, int):
                return [v]
            elif hasattr(v, "__len__") and hasattr(v, "__getitem__"):
                return v
            else:
                raise TypeError(f"{name} debe ser una secuencia (list, tuple, etc.) o float/int, no {type(v)}")
        vec1 = ensure_sequence(vec1, "vec1")
        vec2 = ensure_sequence(vec2, "vec2")

        # Check cache first
        cached_sim = cache_manager.get_similarity(vec1, vec2)
        if cached_sim is not None:
            return cached_sim

        if vec1 is None or vec2 is None or len(vec1) != len(vec2):
            # Cache the zero result for consistency, though unlikely to be hit with None inputs
            cache_manager.cache_similarity(vec1, vec2, 0.0)
            return 0.0

        dot = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a*a for a in vec1))
        norm2 = math.sqrt(sum(b*b for b in vec2))

        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            # Cache the zero result
            cache_manager.cache_similarity(vec1, vec2, 0.0)
            return 0.0

        sim = dot / (norm1 * norm2)
        # Cache the calculated similarity
        cache_manager.cache_similarity(vec1, vec2, sim)
        return sim

    def reset(self):
        """Resetea el estado de activación, nivel de activación y historial."""
        self.activa = False
        self.activation_level = 0.0 # Reset activation level
        self.historial_activacion.clear()
        # Keep historial_embeddings, mini_chain_of_thought, memoria_episodica

    def aplicar_decaimiento(self, refuerzo=1.0, window=10):
        """
        Decaimiento no lineal/contextual: si la activación reciente es alta, el decaimiento es menor.
        refuerzo: factor multiplicativo (1.0 por defecto, <1.0 decae más, >1.0 decae menos)
        window: tamaño de ventana para calcular activación reciente.
        """
        # Calcular score de refuerzo según activación reciente
        if self.historial_activacion:
            recientes = self.historial_activacion[-window:]
            freq = sum(1 for h in recientes if h[1]) / len(recientes)
            refuerzo = 1.0 + 0.5 * freq  # Si freq=1, refuerzo=1.5; si freq=0, refuerzo=1.0
        else:
            refuerzo = 1.0
        decay = self.decay_rate / refuerzo
        self.activation_level = max(0.0, self.activation_level - decay)
        self.activa = self.activation_level >= self.umbral_activacion
        # print(f"DEBUG: MicroNeurona {self.id} - Applied contextual decay, new activation: {self.activation_level}")

    def get_index_data(self) -> Tuple[Any, List[float], Dict[str, Any]]:
        """Returns data needed to add this neuron to a VectorIndex."""
        return self.id, self.embedding, self.metadata
