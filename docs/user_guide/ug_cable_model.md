The cable modelisation is divided in two parts:  

- The space positioning part: the model used for plotting the 3D curve.
- The physic part: the model used for physics calculations.

This division is only for the sake of clarity. Indeed, the space positioning part is linked to the physic part through the parameter p.

## Span cable model

### Input 

![Image not available](./assets/cable_plane.drawio.png "Cable plane image")

- $a$ the span length
- $b$ the elevation difference
- $p$ the sagging parameter

The cable equation is expressed in the cable frame, depending on $\beta$. 
The cable plane is "opening" due to the elevation difference with an angle $\alpha$.  
$\alpha$ can be expressed depending on beta:

$$
    \alpha = \arctan \left( \frac{b \cdot \sin \beta}{a} \right) = \arccos \left( \frac{a}{\sqrt{a^2 + ( b \cdot \sin \beta)^2}} \right)
$$

in the new cable plane, a and b become respectively a' and b'

$$
    a' = \sqrt{a^2+(b \cdot \sin\beta)^2} 
$$

$$
    b' = b \cdot \cos \beta
$$  

Another way to see the cable plane is to rotate the cable plane.


### Catenary model

#### 1. Cable equation
The catenary model can be written as the following:

$$
    z(x) = p \cdot \left( \cosh \left( \frac{x}{p} \right) - 1 \right)
$$

In order to take the right piece of the curve, the extremum abscissa have to be calculated function of the cable parameter, in the cable's plane.
Let M the left hanging point and N the right hanging point. The cable plane defined in the general concepts is recalled here. 

$a$ and $b$ are expressed in the vertical plane. The equations that follow use $a'$ and $b'$ so they can be applied without considering $\beta$.

$$
    x_m = -\frac{a'}{2}+p \cdot asinh \left( \frac{b'}{2 \cdot p \cdot \sinh⁡ \left( \frac{a'}{2 \cdot p} \right)}  \right)
$$

$$
    x_n = a' + x_m
$$

The cable length can be divided into two parts:

$$
    L = L_m + L_n
$$

$$
    L_m = -p \cdot \sinh \left( \frac{x_m}{p} \right)
$$

$$
    L_n = p \cdot \sinh \left( \frac{x_n}{p} \right)
$$



#### 2. Tension

The cable equation has an impact on the definition of the mechanical tensions of the cable:

$$T_h = p \cdot m \cdot \lambda$$

with m the load coefficient. No load on cable means $m = 1$.

$$
    T_v(x) = T_h \cdot \sinh \left( \frac{x}{p} \right)
$$

$$
    T_{max} = T_h \cdot \cosh⁡ \left( \frac{x}{p} \right)
$$

$$
    {T_{mean}}_m = \frac{-x_m \cdot T_h + L_m \cdot {T_{max}}_m}{2 \cdot L_m}
$$

$$
    {T_{mean}}_n = \frac{x_n \cdot T_h + L_n \cdot {T_{max}}_n}{2 \cdot L_n}
$$

$$
    T_{mean} = \frac{{T_{mean}}_{m} \cdot L_m+{T_{mean}}_{n} \cdot L_n}{L}
$$

### Parabola model
..

### Elastic catenary model
..


## Physics-based cable model

### Cable's physics properties

- $S$: section in $mm^2$
- $D$: diameter in $mm$
- $\lambda$: linear weight in $N/m$
- $E$: Young modulus in $GPa$
- $\alpha_{th}$: dilatation coefficient in $°C^{-1}$

### Unstressed length

The cable is strained when subjected to mechanical tensions or thermal changes.

- mechanical part due to tension $T_{mean}$ :
$\varepsilon_{mecha}$

- thermal part due to temperature $\theta$ :
$\varepsilon_{therm} = (\theta - \theta_{ref}) \cdot \alpha_{th}$

$\theta_{ref}$ being the reference temperature chosen to define the unstressed cable length.

The total strain is : 

$$\varepsilon_{total} = \varepsilon_{mecha} + \varepsilon_{therm} = \frac{\Delta L}{L_{ref}} = \frac{L - L_{ref}}{L_{ref}}$$

The unstressed cable length $L_{ref}$ can be then expressed as the following: 

$$L_{ref} = \frac{L}{1+\varepsilon_{total}}$$


#### Linear elasticity model

In this section, we assume the cable exhibits linear elasticity and can be described by a linear relationship between stress $\sigma = T_{mean}/S$ and strain $\varepsilon_{mecha}$.

$$\varepsilon_{mecha} = \varepsilon_{m} = \frac{\sigma}{E} = \frac{T_{mean}}{E\cdot S}$$

#### Polynomial model

In this section, we assume that stress and strain are linked by a polynomial relationship: 

$$\sigma = a0 + a1 \cdot \varepsilon_{m} + a2 \cdot \varepsilon_{m}^2 + a3 \cdot \varepsilon_{m}^3 + a4 \cdot \varepsilon_{m}^4$$

However, this equation is only true if the cable strain $\sigma$ is higher than it has ever been. If $\sigma$ is lower than a previous value, the model is once more linear, but the line starts on the highest value of $\sigma$ that has been reached.


![Image not available](./assets/polynomial_deformation.drawio.png "Polynomial deformation")