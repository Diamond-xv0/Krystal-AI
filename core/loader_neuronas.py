"""
Loader y gestor de neuronas para arquitectura modular:
- Carga neuronas básicas, aprendidas y personalizadas.
- Prioriza: personalizadas > aprendidas > base.
- Permite fácil edición y limpieza de neuronas aprendidas.
"""

from core.MNs_01 import poblar_modelo_base
from core.razonador import Razonador

def cargar_neuronas():
    """
    Carga y clasifica todas las neuronas del sistema en categorías:
    - micro_neuronas: lista de MicroNeurona
    - neuronas: lista de Neurona
    - macro_neuronas: lista de MacroNeurona
    - interconectoras: lista de NeuronaInterconectora
    """
    from core.micro_neurona import MicroNeurona
    from core.neurona import Neurona
    from core.macro_neurona import MacroNeurona
    from core.neurona_interconectora import NeuronaInterconectora

    # 1. Cargar neuronas básicas
    from core.memoria import Memoria
    from core.personalidad import Personalidad
    memoria = Memoria()
    personalidad = Personalidad()
    razonador = Razonador(memoria, personalidad)
    neuronas_base = poblar_modelo_base(razonador)

    # 2. Cargar neuronas aprendidas (si existen)
    try:
        from core.MNs_aprendidas import poblar_neuronas_aprendidas
        neuronas_aprendidas = poblar_neuronas_aprendidas()
    except ImportError:
        neuronas_aprendidas = []

    # 3. Cargar neuronas personalizadas (si existen)
    try:
        from core.MNs_personalizadas import poblar_neuronas_personalizadas
        neuronas_personalizadas = poblar_neuronas_personalizadas()
    except ImportError:
        neuronas_personalizadas = []

    # 4. Priorizar: personalizadas > aprendidas > base
    todas = {}
    for n in neuronas_base + neuronas_aprendidas + neuronas_personalizadas:
        todas[n.id] = n  # Sobrescribe si ya existe

    micro_neuronas = []
    neuronas = []
    macro_neuronas = []
    interconectoras = []

    for n in todas.values():
        if isinstance(n, MicroNeurona):
            micro_neuronas.append(n)
        elif isinstance(n, Neurona):
            neuronas.append(n)
        elif isinstance(n, MacroNeurona):
            macro_neuronas.append(n)
        elif isinstance(n, NeuronaInterconectora):
            interconectoras.append(n)
        else:
            # Si no se reconoce el tipo, intentar clasificar por atributo
            tipo = getattr(n, "tipo", None)
            if tipo == "micro":
                micro_neuronas.append(n)
            elif tipo == "neurona":
                neuronas.append(n)
            elif tipo == "macro":
                macro_neuronas.append(n)
            elif tipo == "interconectora":
                interconectoras.append(n)

    return {
        "micro_neuronas": micro_neuronas,
        "neuronas": neuronas,
        "macro_neuronas": macro_neuronas,
        "interconectoras": interconectoras,
        "todas": list(todas.values())
    }

# --- Utilidad para guardar una neurona aprendida en MNs_aprendidas.py ---

import json

def guardar_neurona_aprendida(neurona, archivo='core/MNs_aprendidas.py'):
    """
    Serializa la neurona a dict y la añade a la lista en MNs_aprendidas.py.
    (En producción, mejor usar JSON/YAML por neurona.)
    """
    # Asegura que el archivo existe y tiene la estructura básica
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
    except FileNotFoundError:
        contenido = "from core.micro_neurona import MicroNeurona\n\nneuronas_aprendidas = []\n\ndef poblar_neuronas_aprendidas():\n    return neuronas_aprendidas\n"

    if "neuronas_aprendidas" not in contenido:
        contenido += "\nneuronas_aprendidas = []\n"

    # Serializa la neurona
    neurona_dict = neurona.to_dict() if hasattr(neurona, "to_dict") else neurona
    nueva_linea = f"\nneuronas_aprendidas.append(MicroNeurona(**{json.dumps(neurona_dict, ensure_ascii=False, indent=2)}))\n"

    # Añade la nueva neurona al final
    with open(archivo, 'a', encoding='utf-8') as f:
        f.write(nueva_linea)

    print(f"[INFO] Neurona aprendida guardada en {archivo}: {neurona_dict.get('id', 'sin_id')}")