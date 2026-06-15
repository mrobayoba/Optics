import numpy as np
import matplotlib.pyplot as plt

def calcular_theta_t(ni, nt, theta_i_deg):
    """Calcula el ángulo de refracción usando la Ley de Snell."""
    theta_i_rad = np.radians(theta_i_deg)
    sin_theta_t = (ni / nt) * np.sin(theta_i_rad)
    if sin_theta_t > 1.0:
        return None # Caso Reflexión Interna Total (TIR)
    return np.arcsin(sin_theta_t)

def coeficientes_intensidad(ni, nt, theta_i_deg):
    """Calcula la Reflectancia (R) y Transmitancia (T)."""
    theta_i = np.radians(theta_i_deg)
    theta_t = calcular_theta_t(ni, nt, theta_i_deg)
    
    if theta_t is None:
        return {'Rs': 1.0, 'Rp': 1.0, 'Ts': 0.0, 'Tp': 0.0}

    # Ecuaciones de Fresnel para amplitudes (Hecht)
    rs = -np.sin(theta_i - theta_t) / np.sin(theta_i + theta_t)
    rp =  np.tan(theta_i - theta_t) / np.tan(theta_i + theta_t)
    
    Rs = rs**2
    Rp = rp**2
    
    return {'Rs': Rs, 'Rp': Rp, 'Ts': 1-Rs, 'Tp': 1-Rp}

def graficar_reflectancia(ni, nt):
    """Genera las curvas de reflectancia en función del ángulo de incidencia."""
    angulos = np.linspace(0, 90, 500)
    rs_vals = []
    rp_vals = []

    for ang in angulos:
        res = coeficientes_intensidad(ni, nt, ang)
        rs_vals.append(res['Rs'])
        rp_vals.append(res['Rp'])

    # Cálculo de puntos críticos para anotación
    brewster = np.degrees(np.arctan(nt/ni))
    critico = np.degrees(np.arcsin(nt/ni)) if ni > nt else None

    plt.figure(figsize=(10, 6))
    plt.plot(angulos, rs_vals, 'b-', label='Reflectancia s (TE)', linewidth=2)
    plt.plot(angulos, rp_vals, 'r--', label='Reflectancia p (TM)', linewidth=2)
    
    # Marcar Ángulo de Brewster
    plt.axvline(brewster, color='gray', linestyle=':', alpha=0.7)
    plt.text(brewster+1, 0.1, f'Ángulo Brewster\n({brewster:.2f}°)', color='gray')

    # Marcar Ángulo Crítico si existe
    if critico:
        plt.axvline(critico, color='green', linestyle=':', alpha=0.7)
        plt.text(critico-10, 0.5, f'Ángulo Crítico\n({critico:.2f}°)', color='green')

    plt.title(f'Reflectancia en Interfase (ni={ni} -> nt={nt})', fontsize=14)
    plt.xlabel('Ángulo de Incidencia (grados)', fontsize=12)
    plt.ylabel('Reflectancia (R)', fontsize=12)
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()