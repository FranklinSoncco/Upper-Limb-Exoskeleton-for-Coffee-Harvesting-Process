# Exoesqueleto para Cosecha de Café - Análisis Cinemático

Implementación computacional del modelamiento cinemático de un exoesqueleto bilateral de 5 grados de libertad por brazo, diseñado para reducir cargas musculoesqueléticas durante la cosecha manual de café.

## 📁 Estructura del Repositorio
```
├── human_mesh/                          # Modelos 3D del cuerpo humano
├── images/                              # Imágenes y visualizaciones
├── M_Both_arm.mlx                       # Cinemática bilateral completa (MATLAB)
├── M_Exoskeleton_in_person.mlx          # Exoesqueleto sobre modelo humano (MATLAB)
├── M_Load_person.mlx                    # Carga de modelo humano (MATLAB)
├── M_Single_arm_LEFT.mlx                # Cinemática brazo izquierdo (MATLAB)
├── M_Single_arm_RIGHT.mlx               # Cinemática brazo derecho (MATLAB)
├── M_Workspace.mlx                      # Espacio de trabajo (MATLAB)
├── P_inverse_kinematics_analysis.py     # Cinemática inversa (Python)
└── P_singularity_analysis.py            # Análisis de singularidades (Python)
```

## 🎯 Descripción de Archivos

### MATLAB Live Scripts (Prefijo M_)

**`M_Both_arm.mlx`** (27 KB)
- Implementación bilateral del exoesqueleto completo
- Cinemática directa para brazo derecho e izquierdo simultáneamente
- Visualización 3D de ambos brazos

**`M_Exoskeleton_in_person.mlx`** (16 KB)
- Superposición del exoesqueleto sobre modelo anatómico humano
- Validación de alineación con miembros superiores y columna vertebral
- Configuración de 5 articulaciones: q₁, q₂ (pasivas) y q₃, q₄, q₅ (activas)

**`M_Load_person.mlx`** (29 KB)
- Carga y procesamiento de modelo humano desde archivo STL
- Aplicación de rotaciones y escalado para correspondencia dimensional
- Preparación de malla triangular para visualización

**`M_Single_arm_LEFT.mlx`** (32 KB)
- Cinemática directa específica para brazo izquierdo
- Configuración espejo del brazo derecho
- Análisis individual de workspace izquierdo

**`M_Single_arm_RIGHT.mlx`** (33 KB)
- Cinemática directa específica para brazo derecho
- Implementación de transformaciones DH y offsets estructurales
- Salida: Posición del efector final [px, py, pz]

**`M_Workspace.mlx`** (45 KB)
- Generación del espacio de trabajo mediante muestreo de 115,200 configuraciones/brazo
- Visualización en 4 vistas: 3D, frontal, lateral, superior
- Análisis cuantitativo: volumen (1.85 m³), alcance radial (1.35 m), rango vertical (-0.85 a +0.75 m)

### Python Scripts (Prefijo P_)

**`P_inverse_kinematics_analysis.py`** (8 KB)
- Demostración de inexistencia de solución analítica (redundancia + complejidad algebraica)
- Implementación de solución numérica: Jacobiano pseudoinverso amortiguado
- Ejemplo de convergencia en 3 iteraciones con λ=0.05
- Salida: Tabla de convergencia con error posicional

**`P_singularity_analysis.py`** (9 KB)
- Cálculo del Jacobiano geométrico (3×5)
- Identificación de 3 tipos de singularidades con valores numéricos
- Evaluación de det(JJ^T), valores singulares σ, número de condición κ
- Salida: Configuraciones críticas y métricas de manipulabilidad

### Carpetas

**`human_mesh/`**
- Modelos 3D del cuerpo humano en formato STL
- Utilizado para validación dimensional del exoesqueleto

**`images/`**
- Visualizaciones generadas de cadena cinemática
- Espacio de trabajo en diferentes vistas
- Exoesqueleto sobre modelo humano

## 🛠️ Requisitos

### MATLAB
- MATLAB R2024a o superior
- Toolboxes: Symbolic Math, Robotics System
- Archivo STL del modelo humano en carpeta `human_mesh/`

### Python
```bash
pip install numpy scipy sympy
```
- Python 3.11+
- NumPy 1.26+
- SciPy 1.11+
- SymPy 1.12+

## 🚀 Uso Rápido

### Cinemática Bilateral Completa (MATLAB)
```matlab
% Abrir M_Both_arm.mlx en MATLAB
% Ejecutar: Run
% Visualiza ambos brazos del exoesqueleto
```

### Exoesqueleto sobre Humano (MATLAB)
```matlab
% Abrir M_Exoskeleton_in_person.mlx
% Asegurar que FinalBaseMesh.stl esté en human_mesh/
% Ejecutar: Run
```

### Espacio de Trabajo (MATLAB)
```matlab
% Abrir M_Workspace.mlx
% Modificar rangos articulares si es necesario (líneas 22-26):
q1_range = linspace(-20, 20, 8);   % Pasiva
q2_range = linspace(-20, 20, 8);   % Pasiva
q3_range = linspace(-45, 45, 12);  % Activa
q4_range = linspace(-10, 45, 10);  % Activa
q5_range = linspace(0, 145, 15);   # Activa
% Ejecutar: Run (2-3 minutos)
```

### Cinemática Inversa (Python)
```bash
python P_inverse_kinematics_analysis.py
```
Genera análisis completo de convergencia para posición objetivo [0.85, 0.30, -0.65] m

### Singularidades (Python)
```bash
python P_singularity_analysis.py
```
Evalúa 7 configuraciones y genera tabla comparativa de singularidades

## 📊 Resultados Principales

| Métrica | Valor |
|---------|-------|
| **Grados de libertad** | 5 por brazo (2 pasivos + 3 activos) |
| **Volumen workspace bilateral** | 1.85 m³ |
| **Alcance radial máximo** | 1.35 m |
| **Rango vertical** | -0.85 m a +0.75 m |
| **Configuraciones evaluadas** | 115,200 por brazo |
| **Número de condición (κ)** | 5.44 (buena manipulabilidad) |

### Parámetros Geométricos

| Parámetro | Símbolo | Valor |
|-----------|---------|-------|
| Longitud base soporte dorsal | L₁ | 0.10 m |
| Altura soporte columna | L₂ | 0.35 m |
| Offset lateral hombro | L₃ | 0.08 m |
| Longitud brazo superior | L₉ | 0.65 m |
| Longitud antebrazo | L₁₀ | 0.55 m |

### Límites Articulares

| Articulación | Tipo | Mín | Máx | Función |
|--------------|------|-----|-----|---------|
| q₁ | Pasiva | -20° | 20° | Acomodación lateral |
| q₂ | Pasiva | -20° | 20° | Acomodación hombro |
| q₃ | Activa | -45° | 45° | Abducción/aducción |
| q₄ | Activa | -10° | 45° | Flexión hombro |
| q₅ | Activa | 5° | 145° | Flexión codo |

## 🔬 Validación Numérica

### Cinemática Directa
**Configuración:** q = [0°, 0°, 28°, 3°, 13°]  
**Posición alcanzada:** p = [0.9412, 0.2356, -0.6930] m

### Cinemática Inversa
**Posición deseada:** p_d = [0.850, 0.300, -0.650] m  
**Convergencia:** 3 iteraciones  
**Solución:** q* = [5.83°, -3.51°, 19.14°, 0.59°, 29.06°]

### Singularidades
- **Tipo 1** (q₅=0°): det(JJ^T) = 0.4593, κ = 5.56
- **Tipo 2** (|q₃|=45°): det(JJ^T) = 0.4024-0.5705, κ = 5.43-6.67
- **Tipo 3** (frontera): det(JJ^T) = 0.0071, κ = 3.39

## 📖 Documentación Adicional

El marco metodológico completo, incluyendo ecuaciones detalladas y análisis teórico, se encuentra en el documento de tesis asociado.

## 🤝 Contribución

Este repositorio forma parte del proyecto de investigación sobre desarrollo de exoesqueletos para asistencia en cosecha de café. Para más información o colaboraciones, contactar al equipo de desarrollo.

## 📝 Licencia

Este código se proporciona con fines académicos y de investigación.

---
**Última actualización:** Noviembre 2025  
**Desarrollado para:** Investigación en exoesqueletos de asistencia agrícola