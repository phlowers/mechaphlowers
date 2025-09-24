The cable modeling is divided into two parts:

- The space positioning part: the model used for plotting the 3D curve.
- The physics part: the model used for physics calculations.

This division is for clarity purposes only. The space positioning part is linked to the physics part through the parameter $p$.

## Span cable modeling

### Inputs

![cable plane](./assets/cable_plane.drawio.png "Cable plane image")

- $a$ the span length
- $b$ the elevation difference
- $p$ the sagging parameter

The cable equation is expressed in the cable's frame, depending on $\beta$.  
The cable plane "opens" due to the elevation difference, forming an angle $\alpha$.

$\alpha$ can be expressed as a function of $\beta$:

$$
    \alpha = \arctan \left( \frac{b \cdot \sin \beta}{a} \right) = \arccos \left( \frac{a}{\sqrt{a^2 + (b \cdot \sin \beta)^2}} \right)
$$

In the new cable plane, $a$ and $b$ become $a'$ and $b'$, respectively:

$$
    a' = \sqrt{a^2 + (b \cdot \sin \beta)^2}
$$

$$
    b' = b \cdot \cos \beta
$$

### Catenary model

#### 1. Cable equation

The catenary model can be written as follows:

$$
    z(x) = p \cdot \left( \cosh \left( \frac{x}{p} \right) - 1 \right)
$$

To extract the appropriate curve segment, the extremum abscissa values ($x_M$ and $x_N$) must be calculated as functions of the sagging parameter $p$ within the cable's plane. $x = 0$  represents the center of the cable.  
Let $M$ be the left hanging point and $N$ the right hanging point. The cable plane, as defined in the general concepts, is recalled here.  
$a$ and $b$ are expressed in the vertical plane. The following equations use $a'$ and $b'$, so they can be applied without considering $\beta$:

$$
    x_M = -\frac{a'}{2}+p \cdot asinh \left( \frac{b'}{2 \cdot p \cdot \sinh⁡ \left( \frac{a'}{2 \cdot p} \right)}  \right)
$$

$$
    x_N = a' + x_M
$$

The cable length $L$ can be divided into two parts:

$$
    L = L_M + L_N
$$

where

$$
    L_M = -p \cdot \sinh \left( \frac{x_M}{p} \right)
$$

and 

$$
    L_N = p \cdot \sinh \left( \frac{x_N}{p} \right)
$$

#### 2. Tension

The cable equation has an impact on mechanical tension definition. The mechanical tension is separated into two parts:

* The horizontal component 

$$
    T_h = p \cdot k_{load} \cdot \lambda
$$

With $k_{load}$ the load coefficient. No load on cable means $k_{load} = 1$. It is constant along the cable.

* The vertical component

$$
    T_v(x) = T_h \cdot \sinh \left( \frac{x}{p} \right)
$$

Then, the maximal tension $T_{max}$, function of $x$, is a combination of these two components:

$$
    T_{max}(x) = \sqrt{T_h^2 + T_v(x)^2}
$$

$$
    T_{max}(x) = T_h \cdot \cosh⁡ \left( \frac{x}{p} \right)
$$

To understand the tension distribution along the cable, we calculate the overall mean tension.  
In order to do that, we can separate the cable in two halves, at $x=0$, the lowest point of the cable.  
Then, we can calculate ${T_{mean}}_M$ and ${T_{mean}}_N$, the mean tensions on the left and right parts of the cable respectively. They are given by the following formulas: 

$$
{T_{mean}}_M = \frac{-x_M \cdot T_h + L_M \cdot {T_{max}}_M}{2 \cdot L_M}
$$

and

$$
{T_{mean}}_N = \frac{x_N \cdot T_h + L_N \cdot {T_{max}}_N}{2 \cdot L_N}
$$

where $x_M, x_N$ are the horizontal positions at $M$ and $N$, $T_h$ is the constant horizontal tension,
and $L_M, L_N$ are the cable lengths around the extremities. 

 
The overall mean tension is the weighted average of the mean tensions on the left and on the right:

$$
T_{mean} = \frac{{T_{mean}}_M \cdot L_M + {T_{mean}}_N \cdot L_N}{L}
$$

with $L = L_M + L_N$ the total cable length. These expressions provide a global and local understanding of
how the forces are distributed along the cable, essential for analyzing strain and deformation.

### Additional models

#### Parabola model
(*To be developed later.*)

#### Elastic catenary model
(*To be developed later.*)

## Physics-based cable modeling

### Physical properties of the cable

- $S$ the cross-sectional area, in $mm^2$
- $D$ the diameter, in $mm$
- $\lambda$ the linear weight, in $N/m$
- $E$ the Young's modulus, in $GPa$
- $\alpha_{th}$ the thermal expansion coefficient, in $°C^{-1}$

### Linear elasticity model

In this section, the cable is assumed to exhibit linear elasticity, meaning its strain is directly proportional to
stress.  
More complex behaviors (e.g., plasticity) can be added in the future.

The cable's strain results from two sources, mechanical tensions and temperature changes:

* Mechanical strain due to tension $T_{mean}$:

$$
   \varepsilon_{mecha} = \frac{T_{mean}}{E \cdot S}
$$

* Thermal strain due to temperature $\theta$:

$$
   \varepsilon_{therm} = (\theta - \theta_{ref}) \cdot \alpha_{th}
$$

where $\theta_{ref}$ is the reference temperature used to define the unstressed cable length

The total strain is:

$$
    \varepsilon_{total} = \varepsilon_{mecha} + \varepsilon_{therm} = \frac{\Delta L}{L_{ref}} = \frac{L - L_{ref}}{L_{ref}}
$$

With the unstressed cable length $L_{ref}$:

$$
    L_{ref} = \frac{L}{1 + \varepsilon_{total}}
$$