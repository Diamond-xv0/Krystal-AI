from .syntax_engine import SyntaxEngine
from .semantic_validator import SemanticValidator

class GrammarAdjudicator:
    """
    Gestiona las reglas de secuencia gramatical para la generación de respuestas.
    Funciona como un árbitro que, dada una palabra, decide qué tipos de palabras
    pueden venir a continuación.
    """
    def __init__(self):
        """
        Initializes the GrammarAdjudicator with SyntaxEngine and SemanticValidator instances.
        """
        self.syntax_engine = SyntaxEngine()
        self.semantic_validator = SemanticValidator()

    def get_valid_next_words(self, vocabulario, mn_palabra_anterior=None, macro_activas=None, interconectoras=None):
        """
        Devuelve una lista de todas las MicroNeuronas del vocabulario
        que son gramaticalmente válidas para ir después de la palabra anterior.
        Si hay MacroNeuronas activas, prioriza palabras alineadas con ellas.
        Si se proveen interconectoras, prioriza candidatas con fuerte conexión semántica.
        """
        # --- CASO 1: Inicio de la frase ---
        if mn_palabra_anterior is None:
            candidatas_iniciales = []
            for mn in vocabulario:
                gramatica_mn = mn.metadata.get('GRAMATICA', {})
                requisitos = gramatica_mn.get('REQUIERE', [])
                if not requisitos or 'inicio_frase' in requisitos:
                    candidatas_iniciales.append(mn)
            candidatas = candidatas_iniciales
        else:
            # --- CASO 2: Continuación de la frase ---
            gramatica_anterior = mn_palabra_anterior.metadata.get('GRAMATICA', {})
            tipos_permitidos = set(gramatica_anterior.get('PERMITE_DESPUES', ['fin_frase']))
            tipos_prohibidos = set(gramatica_anterior.get('PROHIBE_DESPUES', []))
            candidatas_validas = []
            for mn_candidata in vocabulario:
                gramatica_candidata = mn_candidata.metadata.get('GRAMATICA', {})
                tipo_candidata = gramatica_candidata.get('TIPO')
                if tipo_candidata in tipos_permitidos and tipo_candidata not in tipos_prohibidos:
                    requisitos_candidata = gramatica_candidata.get('REQUIERE')
                    tipo_anterior = gramatica_anterior.get('TIPO')
                    if not requisitos_candidata or tipo_anterior in requisitos_candidata:
                        enriched_metadata = self.syntax_engine.apply_rules(gramatica_candidata.copy())
                        if enriched_metadata.get("syntax_valid", True):
                            if self.semantic_validator.validate(enriched_metadata):
                                candidatas_validas.append(mn_candidata)
            candidatas = candidatas_validas

        # --- MacroNeuronas activas: filtrar/priorizar ---
        if macro_activas:
            macro_ids = [m.id if hasattr(m, "id") else m for m in macro_activas]
            candidatas_macro = [mn for mn in candidatas if any(mid in mn.metadata.get('macro_tags', []) for mid in macro_ids)]
            if candidatas_macro:
                return candidatas_macro
        # Si hay interconectoras y palabra anterior, ponderar por fuerza de conexión semántica
        if interconectoras and mn_palabra_anterior is not None:
            conexiones = []
            for mn in candidatas:
                fuerza = 0.0
                for inter in interconectoras.values():
                    if inter.es_relevante(mn_palabra_anterior.id) and inter.es_relevante(mn.id):
                        fuerza = max(fuerza, inter.similitud_embedding(mn.embedding))
                conexiones.append((mn, fuerza))
            # Ordenar por fuerza de conexión descendente
            conexiones.sort(key=lambda x: x[1], reverse=True)
            candidatas = [mn for mn, _ in conexiones]
        return candidatas
