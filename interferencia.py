import numpy as np

def generar_vector_jones(tipo, theta=0):
    """
    Genera el vector de Jones para una onda que se propaga en el plano xz
    a un ángulo theta/2 respecto al eje z.
    """
    t2 = theta / 2.0
    if tipo == 's' or tipo == 'H':
        return np.array([1], dtype=complex)
    
    if tipo == 'p1' or tipo == 'V1':
        # Onda 1: inclinada hacia +x
        return np.array([np.cos(t2), 0, -np.sin(t2)], dtype=complex)
    
    if tipo == 'p2' or tipo == 'V2':
        # Onda 2: inclinada hacia -x
        return np.array([np.cos(t2), 0, np.sin(t2)], dtype=complex)
    
    if tipo == 'L1':
        p = generar_vector_jones('p1', theta)
        s = generar_vector_jones('s', theta)
        return (p + 1j * s) / np.sqrt(2)
    
    if tipo == 'L2':
        p = generar_vector_jones('p2', theta)
        s = generar_vector_jones('s', theta)
        return (p + 1j * s) / np.sqrt(2)

    if tipo == 'R2':
        p = generar_vector_jones('p2', theta)
        s = generar_vector_jones('s', theta)
        return (p - 1j * s) / np.sqrt(2)
    
    return None

def calcular_visibilidad(J1, J2):
    """
    Calcula la visibilidad (V) basada en el producto interno de Jones.
    V = |J1 · J2*| para intensidades iguales.
    """
        # Convertimos a arreglos de numpy y aplanamos para evitar el error de reshape
    vec1 = np.array(J1).flatten()
    vec2 = np.array(J2).flatten()
    
    # np.vdot conjuga automáticamente el primer argumento
    return np.abs(np.vdot(vec1, vec2))

def grado_polarizacion(stokes_vector):
    """
    Calcula el grado de polarización P = sqrt(S1^2 + S2^2 + S3^2) / S0.
    """
    s = stokes_vector
    return np.sqrt(s[1]**2 + s[2]**2 + s[3]**2) / s