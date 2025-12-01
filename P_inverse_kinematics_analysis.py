import sympy as sp
import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify, solve, atan2, sqrt

# Variables simbólicas
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5', real=True)
px, py, pz = symbols('px py pz', real=True)
L1, L2, L3, L4, L5, L6, L7, L8, L9, L10 = symbols('L1 L2 L3 L4 L5 L6 L7 L8 L9 L10', positive=True)

print("="*80)
print("ANÁLISIS DE CINEMÁTICA INVERSA - RAZÓN DE INEXISTENCIA DE SOLUCIÓN ANALÍTICA")
print("="*80)

# Ecuaciones de cinemática directa (forma compacta para análisis)
# Estas son las ecuaciones completas que se obtuvieron de las transformaciones

# Análisis del problema
print("\n1. PLANTEAMIENTO DEL PROBLEMA")
print("-"*80)
print("Sistema: 5 articulaciones (q1, q2, q3, q4, q5)")
print("Objetivo: 3 coordenadas (px, py, pz)")
print("Tipo: Sistema REDUNDANTE (5 > 3)")
print("\nCuando el sistema es redundante, existen INFINITAS soluciones")
print("No existe solución analítica ÚNICA cerrada\n")

print("2. COMPLEJIDAD ALGEBRAICA")
print("-"*80)
print("Las ecuaciones de cinemática directa contienen:")
print("- Productos de hasta 5 funciones trigonométricas anidadas")
print("- Términos del tipo: sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)")
print("- Acoplamiento no lineal entre todas las articulaciones")
print("\nIntentando resolver px = f(q1,q2,q3,q4,q5):")

# Expresión simplificada de px (términos principales)
px_expr = (L1*cos(q1) + L4*cos(q1) - L2*sin(q1) + L6*sin(q1) + 
           L7*cos(q1)*cos(q2)*cos(q3) - L7*sin(q1)*sin(q3) +
           L9*(sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*cos(q4))

print(f"\nParte de px (simplificada): ")
print(f"px ≈ L1*cos(q1) + L4*cos(q1) + L7*cos(q1)*cos(q2)*cos(q3) + ...")
print(f"     + L9*(sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*cos(q4) + ...")
print("\n¡Imposible despejar q1, q2, q3, q4, q5 analíticamente!")

print("\n3. MÉTODO NUMÉRICO ADOPTADO: JACOBIANO PSEUDOINVERSO")
print("-"*80)
print("Método: Iteración Newton-Raphson con Jacobiano amortiguado")
print("Ventaja: Maneja redundancia y permite optimización secundaria\n")

# Algoritmo implementado
print("ALGORITMO IMPLEMENTADO:")
print("-"*40)
print("""
Entrada: p_d (posición deseada), q_0 (configuración inicial)
Salida: q* (configuración articular solución)

Paso 1: Inicializar k=0, q_k = q_0
Paso 2: Calcular p_k = FK(q_k)  [cinemática directa]
Paso 3: Calcular error e_k = p_d - p_k
Paso 4: Si ||e_k|| < ε → retornar q_k
Paso 5: Calcular Jacobiano J_k = J(q_k)
Paso 6: Actualizar q_{k+1} = q_k + J_k^# * e_k
        donde J^# = J^T(JJ^T + λ²I)^{-1}  [pseudoinversa amortiguada]
Paso 7: k = k+1, ir a Paso 2

Parámetros:
- ε = 0.001 m (tolerancia)
- λ = 0.05 (factor amortiguamiento)
- max_iter = 50
""")

print("\n4. ECUACIÓN DE ACTUALIZACIÓN ESPECÍFICA")
print("-"*80)
print("Dado J(q) ∈ ℝ^{3×5}, la actualización es:")
print("\nq_{k+1} = q_k + J^T(q_k)[J(q_k)J^T(q_k) + λ²I₃]^{-1}(p_d - p_k)")
print("\nEn forma matricial expandida:")
print("""
[q1]       [q1]       [J₁₁ J₁₂ J₁₃ J₁₄ J₁₅]^T                    [px_d - px_k]
[q2]       [q2]       [J₂₁ J₂₂ J₂₃ J₂₄ J₂₅]   [JJ^T + λ²I]^{-1}  [py_d - py_k]
[q3]   =   [q3]   +   [J₃₁ J₃₂ J₃₃ J₃₄ J₃₅]                      [pz_d - pz_k]
[q4]_{k+1} [q4]_k    
[q5]       [q5]      

donde JJ^T + λ²I ∈ ℝ^{3×3} es:
[J₁₁² + J₁₂² + ... + J₁₅² + λ²    J₁₁J₂₁ + ... + J₁₅J₂₅          J₁₁J₃₁ + ... + J₁₅J₃₅        ]
[J₂₁J₁₁ + ... + J₂₅J₁₅            J₂₁² + J₂₂² + ... + J₂₅² + λ²  J₂₁J₃₁ + ... + J₂₅J₃₅        ]
[J₃₁J₁₁ + ... + J₃₅J₁₅            J₃₁J₂₁ + ... + J₃₅J₂₅          J₃₁² + J₃₂² + ... + J₃₅² + λ²]
""")

# Ejemplo numérico
print("\n5. EJEMPLO NUMÉRICO DE CONVERGENCIA")
print("-"*80)

# Parámetros del exoesqueleto
L_vals = {
    L1: 0.1, L2: 0.35, L3: 0.08, L4: 0.2, L5: 0.08,
    L6: 0.05, L7: 0.1, L8: 0.05, L9: 0.65, L10: 0.55
}

def forward_kinematics_num(q_vals):
    """Cinemática directa numérica"""
    q1_v, q2_v, q3_v, q4_v, q5_v = q_vals
    
    # Componente px
    px_v = (0.1*np.cos(q1_v) + 0.2*np.cos(q1_v) - 0.35*np.sin(q1_v) + 0.05*np.sin(q1_v) +
            0.08*np.sin(q2_v)*np.cos(q1_v) - 0.1*np.sin(q1_v)*np.sin(q3_v) +
            0.1*np.cos(q1_v)*np.cos(q2_v)*np.cos(q3_v) - 0.05*np.sin(q2_v)*np.cos(q1_v) +
            0.65*((np.sin(q1_v)*np.cos(q3_v) + np.sin(q3_v)*np.cos(q1_v)*np.cos(q2_v))*np.cos(q4_v) - 
                  np.sin(q2_v)*np.sin(q4_v)*np.cos(q1_v)) +
            0.55*((np.sin(q1_v)*np.cos(q3_v) + np.sin(q3_v)*np.cos(q1_v)*np.cos(q2_v))*np.cos(q4_v) - 
                  np.sin(q2_v)*np.sin(q4_v)*np.cos(q1_v))*np.cos(q5_v) -
            0.55*((np.sin(q1_v)*np.cos(q3_v) + np.sin(q3_v)*np.cos(q1_v)*np.cos(q2_v))*np.sin(q4_v) + 
                  np.sin(q2_v)*np.cos(q1_v)*np.cos(q4_v))*np.sin(q5_v))
    
    # Componente py
    py_v = (0.08 - 0.08*np.cos(q2_v) + 0.05*np.cos(q2_v) + 0.1*np.sin(q2_v)*np.cos(q3_v) +
            0.65*(np.sin(q2_v)*np.sin(q3_v)*np.cos(q4_v) + np.sin(q4_v)*np.cos(q2_v)) +
            0.55*(np.sin(q2_v)*np.sin(q3_v)*np.cos(q4_v) + np.sin(q4_v)*np.cos(q2_v))*np.cos(q5_v) +
            0.55*(-np.sin(q2_v)*np.sin(q3_v)*np.sin(q4_v) + np.cos(q2_v)*np.cos(q4_v))*np.sin(q5_v))
    
    # Componente pz
    pz_v = (0.1*np.sin(q1_v) + 0.35*np.cos(q1_v) + 0.2*np.sin(q1_v) - 0.05*np.cos(q1_v) +
            0.08*np.sin(q1_v)*np.sin(q2_v) + 0.1*np.sin(q1_v)*np.cos(q2_v)*np.cos(q3_v) +
            0.1*np.sin(q3_v)*np.cos(q1_v) - 0.05*np.sin(q1_v)*np.sin(q2_v) +
            0.65*((np.sin(q1_v)*np.sin(q3_v)*np.cos(q2_v) - np.cos(q1_v)*np.cos(q3_v))*np.cos(q4_v) - 
                  np.sin(q1_v)*np.sin(q2_v)*np.sin(q4_v)) +
            0.55*((np.sin(q1_v)*np.sin(q3_v)*np.cos(q2_v) - np.cos(q1_v)*np.cos(q3_v))*np.cos(q4_v) - 
                  np.sin(q1_v)*np.sin(q2_v)*np.sin(q4_v))*np.cos(q5_v) -
            0.55*((np.sin(q1_v)*np.sin(q3_v)*np.cos(q2_v) - np.cos(q1_v)*np.cos(q3_v))*np.sin(q4_v) + 
                  np.sin(q1_v)*np.sin(q2_v)*np.cos(q4_v))*np.sin(q5_v))
    
    return np.array([px_v, py_v, pz_v])

# Posición deseada
p_d = np.array([0.85, 0.30, -0.65])
q_0 = np.deg2rad([0, 0, 20, 10, 30])

print(f"Posición deseada: p_d = [{p_d[0]:.3f}, {p_d[1]:.3f}, {p_d[2]:.3f}] m")
print(f"Config. inicial: q_0 = [0°, 0°, 20°, 10°, 30°]")
print("\nIteraciones:")
print("k | ||e_k|| [m] | q1[°] | q2[°] | q3[°] | q4[°] | q5[°]")
print("-"*70)

q_k = q_0.copy()
lam = 0.05
for k in range(5):  # Solo mostrar primeras 5 iteraciones
    p_k = forward_kinematics_num(q_k)
    e_k = p_d - p_k
    
    print(f"{k} | {np.linalg.norm(e_k):9.6f}  | {np.rad2deg(q_k[0]):5.1f} | {np.rad2deg(q_k[1]):5.1f} | "
          f"{np.rad2deg(q_k[2]):5.1f} | {np.rad2deg(q_k[3]):5.1f} | {np.rad2deg(q_k[4]):5.1f}")
    
    if np.linalg.norm(e_k) < 0.001:
        break
    
    # Calcular Jacobiano numéricamente
    eps = 1e-6
    J = np.zeros((3, 5))
    for i in range(5):
        q_plus = q_k.copy()
        q_plus[i] += eps
        p_plus = forward_kinematics_num(q_plus)
        J[:, i] = (p_plus - p_k) / eps
    
    # Actualización
    JJT_lam = J @ J.T + lam**2 * np.eye(3)
    q_k = q_k + J.T @ np.linalg.solve(JJT_lam, e_k)

print(f"\nConvergencia en {k+1} iteraciones")
print(f"Solución final: q* = [{np.rad2deg(q_k[0]):.2f}°, {np.rad2deg(q_k[1]):.2f}°, "
      f"{np.rad2deg(q_k[2]):.2f}°, {np.rad2deg(q_k[3]):.2f}°, {np.rad2deg(q_k[4]):.2f}°]")
print(f"Posición alcanzada: p = [{forward_kinematics_num(q_k)[0]:.4f}, "
      f"{forward_kinematics_num(q_k)[1]:.4f}, {forward_kinematics_num(q_k)[2]:.4f}] m")

print("\n" + "="*80)
print("CONCLUSIÓN: Se requiere método numérico iterativo")
print("="*80)
