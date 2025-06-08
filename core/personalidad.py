# Clase para gestionar la personalidad de la IA
import yaml

class Personalidad:
    def __init__(self, ruta_yaml):
        with open(ruta_yaml, 'r', encoding='utf-8') as f:
            datos = yaml.safe_load(f)
        self.nombre = datos.get('nombre', 'IA')
        self.rasgos = datos.get('rasgos', {})
        self.estado_emocional = datos.get('estado_emocional', {'humor': 'neutral', 'intensidad': 0.5})