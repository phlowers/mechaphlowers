# State change for dead head span

## External loads

![Image not available](./assets/cable_external_loads.drawio.png "External loads on cable image")

There are 3 external loads:

- $Q_w$ the wind load:  
    - $Q_w = P_w (D + 2 \cdot e)$
    - Depending on cable diameter $D$, ice thickness $e$ and wind pressure $P_w$
- $Q_{ice}$ the ice weight:  
    - $Q_{ice} = 6000 \cdot \pi \cdot e (e+D)$
    - Depending on the cable diameter $D$ and ice thickness $e$
- $\lambda$ the cable linear weight

The resultant of forces, R is equal to: $R = \sqrt{(Q_{ice}+\lambda)^2+ Q_w^2}$

## Load coefficient

The load coefficient is then defined as: $m = R/\lambda$

Which has been already defined for the relation between the sagging parameter, $\lambda$ and the horizontal tension:

$$p = \frac{T_h}{m \cdot \lambda}$$

## Load angle

The load angle $\beta$ can be calculated as:

$$ \beta = \arctan \frac{Q_w}{Q_{ice} + \lambda}$$

## State change algorithm

### Definition

The problem to solve is to calculate the new horizontal tension when an additional load is placed on the cable.

1. calculate the new beta and define the new cable plane
2. from $a'$ and $b'$ calculate $L'$
3. There are two ways to calculate the relative elongation of the cable:  
    - From $L_0$ definition: ${\varepsilon_{total}}_L = \frac{\Delta L}{L_0} = \frac{L' - L_0}{L_0}$
    - From elongation relation with $T_{mean}: ${\varepsilon_{total}}_T = \frac{T_{mean}}{E\cdot S} + \theta \cdot \alpha_{th}$

4. ${\varepsilon_{total}}_L$ and ${\varepsilon_{total}}_T$ are depending on $T_h$. An error function on $T_h$ estimation can be written:
    - $f(T_h) = {\varepsilon_{total}}_L - {\varepsilon_{total}}_T$

### Example resolution method: Newton-Raphson schema

Problem: finding the root of $f(T_h) = 0$

5. Defining numeric derivative approximation $f'(T_h0) = \frac{f(T_h0 + \zeta) - f(T_h0)}{\zeta}$ with $\zeta = 10N$

6. The following series is converging to the result:

$${T_h}_{n+1} = {T_h}_{n} - \frac{f({T_h}_{n})}{f'({T_h}_{n})}$$

with ${T_h}_{0} = T_h0$ the horizontal tension in the vertical plane.

