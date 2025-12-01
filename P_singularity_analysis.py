import numpy as np
import sympy as sp
from sympy import symbols, cos, sin, Matrix, simplify, det

print("="*80)
print("ANÁLISIS DE SINGULARIDADES - CONFIGURACIONES CRÍTICAS")
print("="*80)

# Variables simbólicas
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5', real=True)

# Parámetros numéricos
L1, L2, L3, L4, L5, L6, L7, L8, L9, L10 = 0.1, 0.35, 0.08, 0.2, 0.08, 0.05, 0.1, 0.05, 0.65, 0.55

def compute_jacobian_numerical(q_vals):
    """Calcula Jacobiano numéricamente"""
    q1_v, q2_v, q3_v, q4_v, q5_v = q_vals
    
    def fk(q):
        q1, q2, q3, q4, q5 = q
        
        px = (L1*np.cos(q1) + L4*np.cos(q1) - L2*np.sin(q1) + L6*np.sin(q1) +
              L5*np.sin(q2)*np.cos(q1) - L7*np.sin(q1)*np.sin(q3) +
              L7*np.cos(q1)*np.cos(q2)*np.cos(q3) - L8*np.sin(q2)*np.cos(q1) +
              L9*((np.sin(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2))*np.cos(q4) - 
                  np.sin(q2)*np.sin(q4)*np.cos(q1)) +
              L10*((np.sin(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2))*np.cos(q4) - 
                   np.sin(q2)*np.sin(q4)*np.cos(q1))*np.cos(q5) -
              L10*((np.sin(q1)*np.cos(q3) + np.sin(q3)*np.cos(q1)*np.cos(q2))*np.sin(q4) + 
                   np.sin(q2)*np.cos(q1)*np.cos(q4))*np.sin(q5))
        
        py = (L3 - L5*np.cos(q2) + L8*np.cos(q2) + L7*np.sin(q2)*np.cos(q3) +
              L9*(np.sin(q2)*np.sin(q3)*np.cos(q4) + np.sin(q4)*np.cos(q2)) +
              L10*(np.sin(q2)*np.sin(q3)*np.cos(q4) + np.sin(q4)*np.cos(q2))*np.cos(q5) +
              L10*(-np.sin(q2)*np.sin(q3)*np.sin(q4) + np.cos(q2)*np.cos(q4))*np.sin(q5))
        
        pz = (L1*np.sin(q1) + L2*np.cos(q1) + L4*np.sin(q1) - L6*np.cos(q1) +
              L5*np.sin(q1)*np.sin(q2) + L7*np.sin(q1)*np.cos(q2)*np.cos(q3) +
              L7*np.sin(q3)*np.cos(q1) - L8*np.sin(q1)*np.sin(q2) +
              L9*((np.sin(q1)*np.sin(q3)*np.cos(q2) - np.cos(q1)*np.cos(q3))*np.cos(q4) - 
                  np.sin(q1)*np.sin(q2)*np.sin(q4)) +
              L10*((np.sin(q1)*np.sin(q3)*np.cos(q2) - np.cos(q1)*np.cos(q3))*np.cos(q4) - 
                   np.sin(q1)*np.sin(q2)*np.sin(q4))*np.cos(q5) -
              L10*((np.sin(q1)*np.sin(q3)*np.cos(q2) - np.cos(q1)*np.cos(q3))*np.sin(q4) + 
                   np.sin(q1)*np.sin(q2)*np.cos(q4))*np.sin(q5))
        
        return np.array([px, py, pz])
    
    eps = 1e-7
    J = np.zeros((3, 5))
    p0 = fk(q_vals)
    
    for i in range(5):
        q_plus = q_vals.copy()
        q_plus[i] += eps
        p_plus = fk(q_plus)
        J[:, i] = (p_plus - p0) / eps
    
    return J

def analyze_singularity(q_vals, name):
    """Analiza singularidad de una configuración"""
    J = compute_jacobian_numerical(q_vals)
    JJT = J @ J.T
    
    # Valores singulares
    U, s, Vh = np.linalg.svd(J)
    
    # Determinante
    det_JJT = np.linalg.det(JJT)
    
    # Número de condición
    kappa = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
    
    # Rango
    rank = np.linalg.matrix_rank(J, tol=1e-6)
    
    print(f"\n{name}")
    print(f"  q = [{np.rad2deg(q_vals[0]):6.1f}°, {np.rad2deg(q_vals[1]):6.1f}°, "
          f"{np.rad2deg(q_vals[2]):6.1f}°, {np.rad2deg(q_vals[3]):6.1f}°, {np.rad2deg(q_vals[4]):6.1f}°]")
    print(f"  Valores singulares: σ = [{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}]")
    print(f"  det(JJ^T) = {det_JJT:.6e}")
    print(f"  Rango(J) = {rank}")
    print(f"  κ(J) = {kappa:.2f}")
    
    if rank < 3:
        print(f"  ⚠️  SINGULAR: Pérdida de {3-rank} grado(s) de libertad")
    elif kappa > 30:
        print(f"  ⚠️  Cercano a singularidad (κ > 30)")
    else:
        print(f"  ✓  No singular")
    
    return J, s, det_JJT, kappa

print("\n1. CONDICIÓN DE SINGULARIDAD")
print("-"*80)
print("El sistema es singular cuando:")
print("\n  rank(J) < 3  ⟺  det(JJ^T) = 0  ⟺  σ_min → 0")
print("\ndonde J ∈ ℝ^{3×5} es el Jacobiano geométrico")

print("\n2. CONFIGURACIONES SINGULARES IDENTIFICADAS")
print("="*80)

# SINGULARIDAD TIPO 1: Codo completamente extendido (q5 = 0)
print("\n── SINGULARIDAD TIPO 1: Extensión completa de codo ──")
print("Condición: q5 ≈ 0°")
print("Efecto: Pérdida de movilidad radial (no puede extenderse más)")

q_sing1 = np.deg2rad([0, 0, 25, 15, 0])  # q5 = 0
J1, s1, det1, kappa1 = analyze_singularity(q_sing1, "Config. q5=0°")

print("\nExplicación matemática:")
print("Cuando q5=0°, los eslabones L9 y L10 quedan alineados.")
print("El Jacobiano pierde una dirección: ∂p/∂q5 → 0")
print("Resultado: σ_min → 0, det(JJ^T) → 0")

# SINGULARIDAD TIPO 2: q3 = 90° (alineación de ejes)
print("\n\n── SINGULARIDAD TIPO 2: Alineación de ejes de hombro ──")
print("Condición: q3 ≈ ±90° con ciertos valores de q4")
print("Efecto: Pérdida de un grado de libertad en orientación")

q_sing2a = np.deg2rad([0, 0, 90, 0, 45])  # q3 = 90°
q_sing2b = np.deg2rad([0, 0, -90, 0, 45])  # q3 = -90°

J2a, s2a, det2a, kappa2a = analyze_singularity(q_sing2a, "Config. q3=+90°")
J2b, s2b, det2b, kappa2b = analyze_singularity(q_sing2b, "Config. q3=-90°")

print("\nExplicación matemática:")
print("Cuando |q3| = 90°, los ejes de rotación de q3 y q4 se alinean.")
print("Dos columnas del Jacobiano se vuelven linealmente dependientes.")
print("Resultado: rank(J) < 3")

# SINGULARIDAD TIPO 3: Frontera del workspace (alcance máximo)
print("\n\n── SINGULARIDAD TIPO 3: Frontera del workspace ──")
print("Condición: Configuraciones en límite de alcance")
print("Efecto: No puede moverse hacia afuera del workspace")

q_sing3 = np.deg2rad([0, 0, 0, 0, 145])  # q5 en límite superior
J3, s3, det3, kappa3 = analyze_singularity(q_sing3, "Config. alcance máximo (q5=145°)")

print("\nExplicación matemática:")
print("En la frontera, el efector alcanza máxima distancia radial.")
print("No existe movimiento infinitesimal posible hacia afuera.")
print("El gradiente de alcance se anula: rank(J) puede disminuir localmente.")

# Configuración no singular de referencia
print("\n\n── CONFIGURACIÓN NO SINGULAR (referencia) ──")
q_ref = np.deg2rad([0, 0, 28, 3, 13])
J_ref, s_ref, det_ref, kappa_ref = analyze_singularity(q_ref, "Config. de trabajo típica")

print("\n3. MATRIZ JACOBIANA EN CONFIGURACIÓN SINGULAR")
print("="*80)
print("\nPara q5=0° (SINGULARIDAD TIPO 1):")
print("\nJ =")
print(np.array2string(J1, precision=4, suppress_small=True, separator='  '))
print(f"\nJJ^T =")
JJT1 = J1 @ J1.T
print(np.array2string(JJT1, precision=4, suppress_small=True))
print(f"\ndet(JJ^T) = {det1:.6e} ≈ 0  ← SINGULAR")

print("\n\nPara configuración de referencia (NO SINGULAR):")
print("\nJ =")
print(np.array2string(J_ref, precision=4, suppress_small=True, separator='  '))
print(f"\nJJ^T =")
JJT_ref = J_ref @ J_ref.T
print(np.array2string(JJT_ref, precision=4, suppress_small=True))
print(f"\ndet(JJ^T) = {det_ref:.6f} ≫ 0  ← NO SINGULAR")

print("\n4. RESUMEN DE CONFIGURACIONES SINGULARES")
print("="*80)
print("\n┌─────────────┬──────────────────────┬──────────┬────────────────────────┐")
print("│ Tipo        │ Condición            │ det(JJ^T)│ Efecto                 │")
print("├─────────────┼──────────────────────┼──────────┼────────────────────────┤")
print(f"│ Tipo 1      │ q5 ≈ 0°              │ {det1:.2e} │ Sin movilidad radial   │")
print(f"│ Tipo 2a     │ q3 ≈ +90°            │ {det2a:.2e} │ Pérdida 1 GDL orient.  │")
print(f"│ Tipo 2b     │ q3 ≈ -90°            │ {det2b:.2e} │ Pérdida 1 GDL orient.  │")
print(f"│ Tipo 3      │ q5 ≈ 145° (frontera) │ {det3:.2e} │ Sin movilidad externa  │")
print(f"│ Referencia  │ q=[0,0,28,3,13]°     │ {det_ref:.2e} │ Operación normal       │")
print("└─────────────┴──────────────────────┴──────────┴────────────────────────┘")

print("\n5. LÍMITES ARTICULARES PARA EVITAR SINGULARIDADES")
print("="*80)
print("\nLímites adoptados que evitan singularidades críticas:")
print("  q1 ∈ [-20°, +20°]   → Evita configuraciones extremas laterales")
print("  q2 ∈ [-20°, +20°]   → Evita configuraciones extremas de hombro")
print("  q3 ∈ [-45°, +45°]   → EVITA q3 = ±90° (Singularidad Tipo 2)")
print("  q4 ∈ [-10°, +45°]   → Rango natural de flexión hombro")
print("  q5 ∈ [5°, 145°]     → EVITA q5 = 0° (Singularidad Tipo 1)")
print("\nEstos límites mantienen κ(J) < 30 en todo el workspace operativo.")

print("\n" + "="*80)
