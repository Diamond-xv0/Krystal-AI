import math
from core.micro_neurona import MicroNeurona

class Neurona:
    def __init__(self, id, nombre, condiciones_mn, umbral=0.5, exclusiones_mn=None, metadata=None, decay_rate=0.25):
        self.id = id
        self.nombre = nombre
        self.condiciones_mn = condiciones_mn # List of input micro-neuron IDs
        self.exclusiones_mn = set(exclusiones_mn) if exclusiones_mn is not None else set()
        self.umbral_activacion = umbral
        self.metadata = metadata if metadata is not None else {}
        self.activa = False # Boolean active state (can be derived from activation_level)
        self.activation_level = 0.0 # Current continuous activation level
        self.decay_rate = decay_rate # Rate at which activation decays per iteration
        self.historial_activacion = []
        self.historial_pesos = []

        # Initialize weights for connections to input micro-neurons
        import random
        self.weights = {mn_id: random.uniform(-1, 1) for mn_id in self.condiciones_mn}

        self.embedding = MicroNeurona('tmp', nombre, 'tmp').calcular_embedding(nombre, dim=64)

    def update_weights(self, input_activations, learning_rate=0.05):
        """
        Hebbian/adaptive learning: refuerza pesos si hay co-activación, debilita si no.
        input_activations: dict {mn_id: activation_level}
        """
        for mn_id in self.condiciones_mn:
            if mn_id in input_activations:
                # Hebbian: Δw = lr * (pre * post)
                delta = learning_rate * input_activations[mn_id] * self.activation_level
                self.weights[mn_id] += delta
        # Guardar historial para trazabilidad
        self.historial_pesos.append(self.weights.copy())

    def evaluar(self, input_activations, umbral=None, activation_fn=None, micro_neuronas_dict=None, attention_window=10):
        """
        Evalúa la neurona basada en las activaciones de las micro-neuronas de entrada,
        aplicando pesos y una función de activación no lineal personalizable.
        input_activations: dict {mn_id: activation_level}
        umbral: umbral dinámico (float), si None usa self.umbral_activacion
        activation_fn: callable(float) -> float, e.g. sigmoid, relu, tanh. Por defecto sigmoid.
        micro_neuronas_dict: dict opcional {mn_id: MicroNeurona} para atención contextual.
        attention_window: int, tamaño de ventana para score de atención.
        """
        # <<< LA LÓGICA DE EXCLUSIÓN EN ACCIÓN >>>
        if self.exclusiones_mn.intersection(input_activations.keys()):
            self.activa = False
            self.activation_level = 0.0
            self.historial_activacion.append((self.activa, self.activation_level, "EXCLUIDA"))
            return

        if not self.condiciones_mn:
            self.activa = False
            self.activation_level = 0.0
            return

        weighted_sum = 0.0
        for mn_id in self.condiciones_mn:
            if mn_id in input_activations:
                weight = self.weights.get(mn_id, 0.0)
                # Score de atención: frecuencia de activación reciente
                attention = 1.0
                if micro_neuronas_dict and mn_id in micro_neuronas_dict:
                    mn = micro_neuronas_dict[mn_id]
                    # Tomar últimas N activaciones
                    recent = mn.historial_activacion[-attention_window:]
                    if recent:
                        freq = sum(1 for h in recent if h[1]) / len(recent)
                        attention = 0.5 + 0.5 * freq  # [0.5,1.0]: más reciente = más atención
                weighted_sum += input_activations[mn_id] * weight * attention

        # Funciones de activación disponibles
        def sigmoid(x):
            try:
                return 1 / (1 + math.exp(-x))
            except OverflowError:
                return 0.0
        def relu(x):
            return max(0.0, x)
        def tanh(x):
            return math.tanh(x)

        fn = activation_fn or sigmoid
        activated_value = fn(weighted_sum)
        threshold = umbral if umbral is not None else self.umbral_activacion

        self.activation_level = activated_value
        self.activa = self.activation_level >= threshold

        # Update history, registrar función usada y score de atención promedio
        self.historial_activacion.append((self.activa, self.activation_level, fn.__name__, threshold, "attention"))

    def reset(self):
        """Resetea el estado de activación y el nivel de activación de la neurona."""
        self.activa = False
        self.activation_level = 0.0 # Reset activation level
        self.historial_activacion.clear()

    def aplicar_decaimiento(self, refuerzo=1.0, window=10):
        """
        Decaimiento no lineal/contextual: si la activación reciente es alta, el decaimiento es menor.
        refuerzo: factor multiplicativo (1.0 por defecto, <1.0 decae más, >1.0 decae menos)
        window: tamaño de ventana para calcular activación reciente.
        """
        old_activation_level = self.activation_level
        if self.historial_activacion:
            recientes = self.historial_activacion[-window:]
            freq = sum(1 for h in recientes if h[0]) / len(recientes)
            refuerzo = 1.0 + 0.5 * freq
        else:
            refuerzo = 1.0
        decay = self.decay_rate / refuerzo
        self.activation_level = max(0.0, self.activation_level - decay)
        self.activa = self.activation_level >= self.umbral_activacion
        print(f"DEBUG: Neurona {self.id} - Applied contextual decay, old activation: {old_activation_level:.4f}, new activation: {self.activation_level:.4f}")
