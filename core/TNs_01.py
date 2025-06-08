import collections

# --- 1. La Estructura de Datos del Pensamiento ---
class PropuestaPensamiento:
    """Un objeto que encapsula la sugerencia de una única ThinkingNeuron."""
    def __init__(self, fuente, plan_conceptual, confianza, razonamiento, metadata=None):
        self.fuente = fuente.__class__.__name__ # El nombre de la TN que la generó
        self.plan_conceptual = plan_conceptual
        self.confianza = confianza # De 0.0 a 1.0
        self.razonamiento = razonamiento # Explicación en texto de por qué se propone esto
        self.metadata = metadata if metadata is not None else {}

# --- 2. La Plantilla para Todos los Pensadores ---
class BaseThinkingNeuron:
    """Clase base abstracta para todas las neuronas de pensamiento."""
    def __init__(self):
        if self.__class__ == BaseThinkingNeuron:
            raise TypeError("No se puede instanciar BaseThinkingNeuron directamente.")
        self.nombre = self.__class__.__name__

    def proponer(self, neural_state, retrieved_memories, context_hypotheses):
        """
        Este método debe ser implementado por cada TN hija.
        Recibe el estado neuronal completo, memorias recuperadas y hipótesis de contexto.
        Debe devolver un objeto PropuestaPensamiento o None.
        """
        raise NotImplementedError("Cada ThinkingNeuron debe implementar su propio método 'proponer'.")

# --- 3. El Consejo de Especialistas (Ejemplos) ---

class TN_ProtocoloSocial(BaseThinkingNeuron):
    """Punto de vista: ¿Cuál es la respuesta socialmente esperada y cortés?"""
    def proponer(self, neural_state, retrieved_memories, context_hypotheses):
        escenarios = context_hypotheses
        razonador = getattr(self, "razonador", None)
        interconectoras = getattr(razonador, "interconectoras", {}) if razonador else {}

        for escenario in escenarios:
            if escenario.get('tipo') == 'patron_aprendido':
                patron_id = escenario.get('elementos_clave', [None])[0]
                explicacion = escenario.get('explicacion', '')
                confianza = escenario.get('confianza', 0.8)
                if "saludo" in escenario.get('soporte_evidencia', {}).get('patron', []):
                    plan = ['concepto_saludo', 'concepto_pregunta_bienestar']
                    razonamiento = f"Detecté el patrón aprendido '{patron_id}'. {explicacion} Respondo con saludo y pregunta de cortesía."
                    confianza = self._ajustar_confianza_por_interconectoras(plan, confianza, interconectoras, razonador)
                    return PropuestaPensamiento(self, plan, confianza, razonamiento)

            if escenario.get('tipo') == 'hub_activo':
                hub_id = escenario.get('elementos_clave', [None])[0]
                confianza = escenario.get('confianza', 0.7)
                plan = [hub_id]
                razonamiento = f"Hub semántico activo '{hub_id}', propongo respuesta alineada a ese campo."
                confianza = self._ajustar_confianza_por_interconectoras(plan, confianza, interconectoras, razonador)
                return PropuestaPensamiento(self, plan, confianza, razonamiento)

            if escenario.get('tipo') == 'interaccion_social':
                subtipo = escenario.get('subtipo')
                confianza = escenario.get('confianza', 0.8)

                if subtipo == 'saludo_informal':
                    plan = ['concepto_saludo', 'concepto_pregunta_bienestar']
                    razonamiento = "Detecté un saludo. La norma social es devolver el saludo e iniciar una pregunta de cortesía."
                    confianza = self._ajustar_confianza_por_interconectoras(plan, confianza, interconectoras, razonador)
                    return PropuestaPensamiento(self, plan, confianza, razonamiento)
                
                if subtipo == 'pregunta_bienestar_directa':
                    plan = ['concepto_respuesta_bienestar', 'concepto_pregunta_reciproca']
                    razonamiento = "El usuario ha preguntado por mi estado. Debo responder y mostrar interés recíproco."
                    confianza = self._ajustar_confianza_por_interconectoras(plan, confianza, interconectoras, razonador)
                    return PropuestaPensamiento(self, plan, confianza, razonamiento)

                if subtipo == 'despedida':
                    plan = ['concepto_despedida']
                    razonamiento = "Detecté una despedida. La respuesta socialmente apropiada es despedirse también."
                    confianza = self._ajustar_confianza_por_interconectoras(plan, confianza, interconectoras, razonador)
                    return PropuestaPensamiento(self, plan, confianza, razonamiento)

        return None

    def _ajustar_confianza_por_interconectoras(self, plan, confianza, interconectoras, razonador):
        if not interconectoras or not razonador:
            return confianza
        fuerza_total = 0.0
        count = 0
        for i in range(len(plan)):
            for j in range(i+1, len(plan)):
                for inter in interconectoras.values():
                    if inter.es_relevante(plan[i]) and inter.es_relevante(plan[j]):
                        emb_i = getattr(razonador.micro_neuronas.get(plan[i], None), "embedding", None)
                        fuerza = inter.similitud_embedding(emb_i) if emb_i is not None else 0.0
                        fuerza_total += fuerza
                        count += 1
        if count > 0:
            confianza = min(1.0, confianza + 0.2 * (fuerza_total / count))
        return confianza

class TN_AnalistaLogico(BaseThinkingNeuron):
    """Punto de vista: ¿Se me ha hecho una pregunta directa que requiere una respuesta fáctica?"""
    def proponer(self, neural_state, retrieved_memories, context_hypotheses):
        # Accedemos a las hipótesis de contexto directamente del argumento
        escenarios = context_hypotheses
        for escenario in escenarios:
            if escenario.get('tipo') == 'pregunta_factica':
                subtipo = escenario.get('subtipo')
                confianza = escenario.get('confianza', 0.9)

                if subtipo == 'pregunta_identidad':
                    plan = ['concepto_auto_revelacion_identidad']
                    razonamiento = "Se me ha preguntado directamente por mi identidad. Debo proporcionar una respuesta fáctica."
                    return PropuestaPensamiento(self, plan, confianza, razonamiento)
            
                if subtipo == 'pregunta_capacidades':
                    plan = ['concepto_auto_revelacion_capacidad']
                    razonamiento = "Se me ha preguntado por mis capacidades. La respuesta lógica es enumerarlas."
                    return PropuestaPensamiento(self, plan, confianza, razonamiento)

        return None

class TN_DetectorDeAmbiguedad(BaseThinkingNeuron):
    """Punto de vista: ¿Es clara la entrada del usuario o es confusa y contradictoria?"""
    def proponer(self, neural_state, retrieved_memories, context_hypotheses):
        # Accedemos a las hipótesis de contexto directamente del argumento
        escenarios = context_hypotheses
        num_escenarios = len(escenarios)
        
        if num_escenarios > 1:
            confianza_promedio = sum(s.get('confianza', 0) for s in escenarios) / num_escenarios
            confianza_ambiguedad = min(1.0, (confianza_promedio * (num_escenarios / 2))) # Heurística
            
            if confianza_ambiguedad > 0.6: # Umbral para actuar
                plan = ['concepto_clarificacion']
                razonamiento = (f"Se detectaron {num_escenarios} interpretaciones posibles, "
                                f"indicando ambigüedad con una confianza de {confianza_ambiguedad:.2f}.")
                return PropuestaPensamiento(self, plan, confianza_ambiguedad, razonamiento)
            
        return None

# --- 4. El Sintetizador Maestro ---

class MacroTN:
    def __init__(self, neuronas_thinking):
        self.neuronas_thinking = neuronas_thinking

    def ciclo_razonamiento(self, neural_state, retrieved_memories, context_hypotheses):
        """
        Orquesta el proceso de pensamiento, recolectando propuestas de las Thinking Neurons
        basadas en el estado neuronal, memorias y contexto, y sintetizando un plan.
        Ahora prioriza propuestas alineadas con MacroNeuronas activas.
        """
        # 1. Recolectar propuestas de todos los especialistas, pasando el estado completo
        propuestas = [p for n in self.neuronas_thinking if (p := n.proponer(neural_state, retrieved_memories, context_hypotheses)) is not None]

        if not propuestas:
            return None, [], [] # No hay nada que pensar

        # --- MacroNeuronas activas ---
        razonador = getattr(self, "razonador", None)
        macro_activas = []
        if razonador:
            macro_activas = [m for m in razonador.macro_neuronas.values() if getattr(m, "activa", False)]
        else:
            # Buscar en neural_state si está disponible
            macro_activas = [k for k, v in neural_state.get("macro_activaciones", {}).items() if v]

        # Si hay MacroNeuronas activas, priorizar propuestas alineadas
        if macro_activas:
            macro_ids = [m.id if hasattr(m, "id") else m for m in macro_activas]
            propuestas_macro = [p for p in propuestas if any(concepto in macro_ids for concepto in getattr(p, "plan_conceptual", []))]
            if propuestas_macro:
                # Elegir la de mayor confianza
                propuesta_top = max(propuestas_macro, key=lambda p: p.confianza)
                return {"plan_conceptual": propuesta_top.plan_conceptual}, propuestas, [propuesta_top]

        # Prioridad 1: Manejar la ambigüedad. Si la TN de ambiguedad está segura, su plan gana.
        for p in propuestas:
            if p.fuente == 'TN_DetectorDeAmbiguedad' and p.confianza > 0.8:
                return {"plan_conceptual": p.plan_conceptual}, propuestas, [p]

        # Agrupar conceptos y puntuar por consenso y confianza
        votos_conceptuales = collections.defaultdict(float)
        for p in propuestas:
            if isinstance(p.plan_conceptual, list):
                for concepto in p.plan_conceptual:
                    votos_conceptuales[concepto] += p.confianza

        conceptos_ordenados = sorted(votos_conceptuales.keys(), key=lambda c: votos_conceptuales[c], reverse=True)

        plan_final = []
        if conceptos_ordenados:
            plan_final.append(conceptos_ordenados[0])
            if conceptos_ordenados[0] == 'concepto_saludo' and 'concepto_pregunta_bienestar' in conceptos_ordenados:
                plan_final.append('concepto_pregunta_bienestar')
            elif len(conceptos_ordenados) > 1:
                if conceptos_ordenados[1] not in plan_final:
                    plan_final.append(conceptos_ordenados[1])

        if not plan_final:
            if propuestas:
                propuesta_mas_segura = max(propuestas, key=lambda p: p.confianza)
                plan_final = propuesta_mas_segura.plan_conceptual
            else:
                plan_final = []

        plan_ganador_objeto = {"plan_conceptual": plan_final}
        return plan_ganador_objeto, propuestas, propuestas

# --- 5. Poblar el Consejo Cognitivo ---

def poblar_tns(memoria=None, sistema_neuronal=None, interconectoras=None):
    """
    Inicializa las TNs y MacroTN con acceso a memoria, sistema neuronal bruto e interconectoras.
    """
    neuronas_thinking = [
        TN_ProtocoloSocial(),
        TN_AnalistaLogico(),
        TN_DetectorDeAmbiguedad(),
        # ... aquí añadirías instancias de tus otras 12+ neuronas
    ]
    # Inyectar dependencias a cada TN
    for tn in neuronas_thinking:
        tn.memoria = memoria
        tn.sistema_neuronal = sistema_neuronal
        tn.interconectoras = interconectoras
    macro_tn = MacroTN(neuronas_thinking)
    macro_tn.memoria = memoria
    macro_tn.sistema_neuronal = sistema_neuronal
    macro_tn.interconectoras = interconectoras
    print(f"[DEBUG] Pobladas {len(neuronas_thinking)} neuronas de pensamiento especializado.")
    return neuronas_thinking, macro_tn
