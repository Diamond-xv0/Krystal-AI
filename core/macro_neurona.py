from core.micro_neurona import MicroNeurona

class MacroNeurona:
    def __init__(self, id, nombre, condiciones_n, umbral=0.5, exclusiones_mn=None, metadata=None):
        self.id = id
        self.nombre = nombre
        self.condiciones_n = condiciones_n
        # <<< NUEVO >>> Acepta y almacena las MNs que la excluyen
        self.exclusiones_mn = set(exclusiones_mn) if exclusiones_mn is not None else set()
        self.umbral_activacion = umbral
        self.metadata = metadata if metadata is not None else {}
        self.activa = False
        self.historial_activacion = []
        self.embedding = MicroNeurona('tmp', nombre, 'tmp').calcular_embedding(nombre, dim=64)

    # <<< IMPORTANTE: La firma del método ha cambiado >>>
    # Ahora necesita saber tanto las Neuronas activas (para sus condiciones)
    # como las MicroNeuronas activas (para sus exclusiones).
    def evaluar(self, ns_activas_ids, mns_activas_ids):
        # Primero, la lógica de exclusión
        if self.exclusiones_mn.intersection(mns_activas_ids):
            self.activa = False
            self.historial_activacion.append((self.activa, 0.0, "EXCLUIDA POR MN"))
            return

        # Si no fue excluida, procede con la evaluación normal basada en Neuronas
        if not self.condiciones_n:
            self.activa = False
            return
        
        activas_count = sum(1 for n_id in self.condiciones_n if n_id in ns_activas_ids)
        ratio = activas_count / len(self.condiciones_n)
        self.activa = ratio >= self.umbral_activacion
        self.historial_activacion.append((self.activa, ratio))

    def reset(self):
        self.activa = False
        self.confianza = 0.0
        self.historial_activacion.clear()
