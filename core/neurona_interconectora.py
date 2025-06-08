import numpy as np

class NeuronaInterconectora:
    """
    Neurona que no participa directamente en el razonamiento, sino que conecta conceptos/neuronas.
    Cada instancia tiene su propio embedding de relación y reglas de conexión.
    """
    def __init__(self, id, neuronas_conectadas, embedding=None, reglas=None):
        self.id = id
        self.neuronas_conectadas = neuronas_conectadas  # lista de IDs de neuronas/conceptos
        self.embedding = embedding if embedding is not None else self.generar_embedding()
        self.reglas = reglas if reglas is not None else {}

    def generar_embedding(self, dim=64):
        # Embedding aleatorio para la relación, puede ser reemplazado por lógica más avanzada
        return np.random.uniform(-1, 1, dim)

    def es_relevante(self, concepto):
        """
        Determina si la interconectora conecta con el concepto dado, usando reglas o embeddings.
        """
        return concepto in self.neuronas_conectadas

    def similitud_embedding(self, otro_embedding):
        """
        Calcula la similitud (coseno) entre embeddings de relación.
        """
        a = self.embedding
        b = otro_embedding
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)