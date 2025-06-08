import re

def normalizar_texto(texto):
    """Normaliza el texto a minúsculas y elimina caracteres no relevantes."""
    return texto.lower().strip()

def distancia_levenshtein(a, b):
    """Calcula la distancia de Levenshtein entre dos cadenas."""
    if len(a) < len(b):
        return distancia_levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def corregir_palabra(palabra, vocabulario, max_dist=2):
    """
    Corrige una palabra usando distancia de Levenshtein sobre el vocabulario dado.
    Si no hay corrección cercana, retorna la palabra original.
    """
    palabra = palabra.lower()
    if palabra in [v.lower() for v in vocabulario]:
        return palabra  # No corregir si ya está en el vocabulario
    mejor = palabra
    mejor_dist = max_dist + 1
    candidatos = []
    for v in vocabulario:
        dist = distancia_levenshtein(palabra, v.lower())
        if dist < mejor_dist:
            mejor = v
            mejor_dist = dist
            candidatos = [v]
        elif dist == mejor_dist:
            candidatos.append(v)
    # Solo corregir si la distancia es 1, o 2 si la palabra es larga (>6)
    if mejor_dist == 1 or (mejor_dist == 2 and len(palabra) > 6):
        if len(candidatos) == 1:
            return mejor
    return palabra

def corregir_frase(frase, vocabulario):
    """
    Corrige cada palabra de la frase usando el vocabulario dado.
    """
    palabras = re.findall(r'\w+', frase, flags=re.UNICODE)
    palabras_corregidas = [corregir_palabra(p, vocabulario) for p in palabras]
    # Reconstruir la frase con espacios
    return ' '.join(palabras_corregidas) 