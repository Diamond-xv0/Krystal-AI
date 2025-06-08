from .memoria import Memoria

# Instancia de la clase Memoria que coordina los niveles jerárquicos
memoria_principal = Memoria()

def registrar_memoria(dato, memory_level="short"):
    """Registra un dato en el nivel de memoria especificado."""
    memoria_principal.add_to_memory(dato, memory_level)

def buscar_en_memoria(query, memory_level="short"):
    """Busca un dato en el nivel de memoria especificado."""
    return memoria_principal.retrieve_from_memory(query, memory_level)

# Note: The original buscar_en_memoria searched across all levels.
# The new implementation requires specifying a level.
# A new function could be added to search across levels if needed.
# For now, the function is simplified to search a specific level.