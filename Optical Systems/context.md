De acuerdo con el archivo **Óptica.pdf**, el formalismo matricial en la óptica paraxial permite describir la evolución de un rayo de luz a través de un sistema óptico mediante vectores de estado y matrices de transformación.

A continuación, se extraen los conceptos y fórmulas fundamentales:

### 1. El Vector de Rayo
En la aproximación paraxial (donde los ángulos $\alpha$ son pequeños, $\sin \alpha \approx \alpha$), un rayo se identifica por su altura $x$ respecto al eje óptico y el producto del índice de refracción por el ángulo de inclinación ($n\alpha$):
$$\mathbf{r = \begin{pmatrix} n \alpha \\ x \end{pmatrix}}$$

### 2. Matrices de Transformación Básicas
*   **Refracción (Dioptrio/SRE):** Describe el cambio de dirección al cruzar una superficie esférica de radio $R$.
    $$\mathbf{R_a = \begin{pmatrix} 1 & -\mathcal{P} \\ 0 & 1 \end{pmatrix}}$$
    Donde $\mathcal{P}$ es el **poder de la interfase**: $\mathcal{P} = \frac{n_t - n_i}{R}$.
*   **Traslación:** Describe el avance del rayo una distancia $D_{12}$ en un medio de índice $n$.
    $$\mathbf{T_{12} = \begin{pmatrix} 1 & 0 \\ \frac{D_{12}}{n} & 1 \end{pmatrix}}$$.

### 3. Modelado de Lentes
*   **Lente Gruesa:** Se obtiene multiplicando las matrices de las dos superficies (dioptrios) y la traslación interna a través del espesor de la lente $D_l$:
    $$\mathbf{M_{Lgruesa} = R_{aV'} \cdot T_{VV'} \cdot R_{aV} = \begin{pmatrix} 1 - \frac{\mathcal{P}_V D_l}{n_l} & -\mathcal{P}_{sistema} \\ \frac{D_l}{n_l} & 1 - \frac{\mathcal{P}_{V'} D_l}{n_l} \end{pmatrix}}$$.
    *   **Poder del sistema:** $\mathcal{P}_{sistema} = \mathcal{P}_V + \mathcal{P}_{V'} - \frac{\mathcal{P}_V \mathcal{P}_{V'} D_l}{n_l}$.
*   **Lente Delgada ($D_l = 0$):** La matriz se simplifica, recuperando la **Ecuación del fabricante de lentes**:
    $$\mathbf{M_{Ldelgada} = \begin{pmatrix} 1 & -\mathcal{P}_{Ldelgada} \\ 0 & 1 \end{pmatrix}}, \quad \mathcal{P}_{Ldelgada} = (n_l - n) \left( \frac{1}{R} - \frac{1}{R'} \right)$$.

### 4. Teoría de Planos Conjugados y Principales
*   **Planos Conjugados:** Dos planos son imagen uno del otro si cualquier rayo que sale de un punto en el plano objeto llega al mismo punto en el plano imagen, independientemente de su ángulo $\alpha$. Esto exige que el elemento **$M_{21}$ de la matriz del sistema sea cero**.
*   **Planos Principales ($H$ y $H'$):** Son un par de planos conjugados especiales donde la **magnificación lateral es la unidad** ($m_x = 1$). La matriz de transferencia entre estos planos tiene la misma estructura que la de una lente delgada:
    $$\mathbf{M_{HH'} = \begin{pmatrix} 1 & -\mathcal{P}_{sistema} \\ 0 & 1 \end{pmatrix}}$$
*   **Localización de Planos Principales:** Se definen las distancias $D$ (de $H$ a $V$) y $D'$ (de $V'$ a $H'$):
    *   $D = \frac{n}{M_{12}}(1 - M_{11})$
    *   $D' = \frac{n'}{M_{12}}(1 - M_{22})$.

### 5. Magnificaciones y Formación de Imágenes
Al medir las distancias objeto ($s$) e imagen ($s'$) desde los planos principales, se cumple la fórmula general de formación de imágenes:
$$\mathbf{\frac{n'}{s'} + \frac{n}{s} = \mathcal{P}_{sis}}$$

*   **Magnificación Lateral ($m_x$):** Relación de alturas $x'/x$. En términos de la matriz del sistema:
    $$\mathbf{m_x = M_{22} = -\frac{n \cdot s'}{n' \cdot s}}$$.
*   **Magnificación Angular ($m_\alpha$):** Relación de ángulos $\alpha'/\alpha$.
    $$\mathbf{m_\alpha = M_{11} \left( \frac{n}{n'} \right) = -\frac{s}{s'}}$$.
*   **Ecuación de Lagrange:** Una consecuencia fundamental es que el producto de las magnificaciones cumple $n x \Delta \alpha = n' x' \Delta \alpha'$, lo que implica que $m_x m_\alpha \frac{n'}{n} = 1$.

Tienes razón, para completar el estudio de la óptica paraxial es fundamental integrar el tratamiento de los **espejos** y la rigurosidad de los **criterios de signos**, los cuales definen si una imagen puede proyectarse (real) o es solo una ilusión óptica tras el elemento (virtual).

A continuación, detallo estos conceptos basándome en **Óptica.pdf**, el libro de **Hecht** y **K.K. Sharma**.

---

### 1. La Matriz de Reflexión ($\mathcal{R}_e$)
En el formalismo matricial, un espejo se trata como una interfase que cambia la dirección del rayo pero mantiene el rayo en el mismo medio inicial ($n_i$). Existen dos formas de abordarlo según la fuente:

*   **Enfoque de Óptica.pdf:** Se define la matriz de reflexión para una superficie de radio $R$ como [22, 2.4.A.2]:
    $$\mathbf{\mathcal{R}_e = \begin{pmatrix} 1 & \frac{2n_i}{R} \\ 0 & 1 \end{pmatrix}}$$

---

### 2. Teoría de Espejos Esféricos y Fórmula de Gauss
La relación entre la distancia objeto ($s_o$), la distancia imagen ($s_i$) y el radio de curvatura ($R$) sigue la **Fórmula de Gauss para espejos** [22, 2.3.b.1; 12, 5.48]:
$$\mathbf{\frac{1}{s_o} + \frac{1}{s_i} = -\frac{2}{R}}$$
Donde la **distancia focal ($f$)** es la mitad del radio: $\mathbf{f = -R/2}$. Por lo tanto, la ecuación simplificada es:
$$\mathbf{\frac{1}{s_o} + \frac{1}{s_i} = \frac{1}{f}}$$

---

### 3. Criterios de Signos (Esencial para el examen)
Para determinar las características de la imagen, debemos seguir estrictamente la convención de signos (Light entering from the left) [12, Table 5.4; 22, 2.1.b.4]:

| Parámetro | Signo Positivo (+) | Signo Negativo (-) |
| :--- | :--- | :--- |
| **Distancia Objeto ($s_o$)** | Objeto **Real** (a la izquierda de V). | Objeto **Virtual** (a la derecha de V). |
| **Distancia Imagen ($s_i$)** | Imagen **Real** (a la izquierda de V). | Imagen **Virtual** (a la derecha de V/detrás). |
| **Radio de Curvatura ($R$)** | Centro $C$ a la derecha (**Espejo Convexo**). | Centro $C$ a la izquierda (**Espejo Cóncavo**). |
| **Distancia Focal ($f$)** | Espejo **Cóncavo** (convergente). | Espejo **Convexo** (divergente). |
| **Alturas ($y_o, y_i$)** | Por encima del eje óptico [22, 2.1.b.4]. | Por debajo del eje óptico [22, 2.1.b.4]. |

---

### 4. Magnificación y Naturaleza de la Imagen
*   **Magnificación Lateral ($m_x$):** Define el tamaño y orientación. Se calcula como $m_x = y_i/y_o = -s_i/s_o$ [12, 100; 22, 2.3.b.2].
    *   Si $m_x > 0$: La imagen es **derecha** (erect).
    *   Si $m_x < 0$: La imagen es **invertida**.
    *   Si $|m_x| > 1$: La imagen es **aumentada**.
*   **Imagen Real vs. Virtual:**
    *   **Real:** Se forma cuando los rayos realmente convergen en un punto del espacio (frente al espejo, $s_i > 0$). Puede ser recogida en una pantalla [12, 91; 22, 2.3.a].
    *   **Virtual:** Se forma por la prolongación de rayos divergentes (detrás del espejo, $s_i < 0$). No se puede proyectar, pero el ojo la percibe [12, 91; 22, 2.3.a].

**Contexto del Tutor:** Un espejo **convexo** siempre producirá imágenes virtuales, derechas y disminuidas, sin importar la posición del objeto. En cambio, un espejo **cóncavo** puede producir imágenes reales e invertidas si el objeto está más allá del foco, o imágenes virtuales y aumentadas (como un espejo de maquillaje) si el objeto está entre el vértice y el foco.

**Fuentes relacionadas:**
*   **Hecht, Optics (Fuente 12):** Tablas 5.4 y 5.5 para comparativa de imágenes.
*   **Óptica .pdf (Fuente 22):** Sección 2.3.b para deducción de fórmulas de espejos.
*   **Handbook of Physics (Fuente 11):** Sección 11.1.2 para construcción de rayos principales.

La **teoría de planos conjugados** es fundamental para entender cómo los sistemas ópticos transportan la información de un objeto para formar una imagen nítida. Según el archivo **Óptica .pdf**, este concepto se formaliza mediante el método matricial, permitiendo predecir la posición y el tamaño de las imágenes sin importar la complejidad interna del sistema.

A continuación, detallo los puntos clave de esta teoría paso a paso:

### 1. Definición y Concepto Físico
Se dice que dos planos son **conjugados** cuando uno es la imagen del otro. Físicamente, esto significa que cualquier rayo que parte de un punto $P$ en el plano objeto llegará necesariamente al punto correspondiente $P'$ en el plano imagen, sin importar el ángulo $\alpha$ con el que haya salido del objeto.

### 2. La Condición Matemática ($M_{21} = 0$)
En el formalismo de la óptica paraxial, la transformación de un rayo entre dos planos se describe mediante una matriz de transferencia $\mathbf{M}$:
$$\begin{pmatrix} n' \alpha' \\ x' \end{pmatrix} = \begin{pmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{pmatrix} \begin{pmatrix} n \alpha \\ x \end{pmatrix}$$
Para que la posición final del rayo ($x'$) sea independiente del ángulo de entrada ($\alpha$), el elemento **$M_{21}$ de la matriz debe ser igual a cero**.

Al cumplirse esta condición, las ecuaciones se simplifican a:
*   $x' = M_{22} \cdot x$
*   $n' \alpha' = M_{11} \cdot n \alpha + M_{12} \cdot x$

### 3. Magnificaciones en Planos Conjugados
Bajo la condición de conjugación, se definen dos tipos de magnificaciones fundamentales:

*   **Magnificación Lateral ($m_x$):** Es la relación entre la altura de la imagen y la del objeto.
    $$\mathbf{m_x = \frac{x'}{x} = M_{22}}$$
*   **Magnificación Angular ($m_\alpha$):** Es la relación entre el ángulo de salida y el de entrada (evaluado para un punto sobre el eje, $x=0$).
    $$\mathbf{m_\alpha = \frac{\Delta\alpha'}{\Delta\alpha} = M_{11} \left( \frac{n}{n'} \right)}$$

### 4. La Ecuación de Lagrange
Una consecuencia directa de que el determinante de la matriz del sistema sea unitario ($\text{det}(\mathbf{M}) = m_x m_\alpha \frac{n'}{n} = 1$) es la **Ecuación de Lagrange**:
$$\mathbf{n x \Delta\alpha = n' x' \Delta\alpha'}$$
Esta ecuación es de suma importancia porque establece que el producto del índice de refracción, el tamaño del objeto y la apertura angular es un **invariante** del sistema óptico.

### 5. Planos Principales: Los Conjugados Especiales
Los **Planos Principales ($H$ y $H'$)** son un par de planos conjugados únicos donde la **magnificación lateral es exactamente 1** ($m_x = 1$). 
*   Cualquier rayo que incida en $H$ a una altura $x$, "emerge" de $H'$ a la misma altura $x$.
*   Esto simplifica cualquier sistema complejo (como un objetivo de cámara con múltiples lentes) a una matriz equivalente a la de una lente delgada:
    $$\mathbf{M_{HH'} = \begin{pmatrix} 1 & -\mathcal{P}_{sistema} \\ 0 & 1 \end{pmatrix}}$$

### 6. Formación de Imágenes (Fórmula de Gauss)
Cuando medimos las distancias objeto ($s$) e imagen ($s'$) desde los planos principales, se recupera la forma clásica de la **Fórmula de Gauss** para cualquier sistema:
$$\mathbf{\frac{n'}{s'} + \frac{n}{s} = \mathcal{P}_{sistema}}$$
*   **Imagen Real:** Si $s' > 0$ (la imagen se forma a la derecha del plano principal de salida).
*   **Imagen Virtual:** Si $s' < 0$ (la imagen se forma a la izquierda).

**Fuentes de referencia:**
*   **Óptica .pdf (Fuente 22), Sección 2.4.A.5:** Desarrollo matricial de planos conjugados.
*   **Óptica .pdf (Fuente 22), Sección 2.4.D:** Aplicación de planos principales a la formación de imágenes.
*   **Hecht, Optics (Fuente 12), Sección 6.1:** Descripción de puntos y planos cardinales en lentes gruesas.