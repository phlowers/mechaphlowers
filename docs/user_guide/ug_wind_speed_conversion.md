# Wind speed to wind pressure conversion

The goal of this tool is to convert wind speed into wind pressure. Wind speed is the data usually given by weather forecast,
but our library uses wind pressure for calculations. The wind pressure depends on other parameters: roughness,
studied height, wind angle, force coefficient.


### Inputs

- $th$ the tower height (in $m$)
- $z_0$ the roughness length (in $m$)
- $V_{b,0}$ the basic wind velocity (in $m/s$), defined as the mean wind speed measured over 10 minutes at a height of 10 m in open terrain.
- $\delta$ the angle between the wind direction and the cable (in degrees). The wind has its greatest effect when $\delta = 90^\circ$.

### Variables

- $C_f$ the force coefficient, equals to $1.2$ for weaker winds in working conditions, equals to $1$ for stronger winds.
- $C_SC_D$ the structural factor, equals to $\frac{2}{3}$. It takes into account the fact that the wind does not blow
at max speed on all spans.
- $h$ the studied height (in $m$)
- $k_r$ the terrain factor
- $V_m$ the mean wind velocity at a given height (in $m/s$)
- $I_v$ the turbulence intensity
- $q_p$ the peak wind pressure (in $Pa$)

### Formulas

For 90kV/63kV voltage levels, we take $h = th * \frac{3}{4}$.

For 225kV/400kV voltage levels, we take $h = th * \frac{2}{3}$.

The roughness length $z_0$ depends on the type of terrain:

| terrain category | 0     | II   | III |
| ---------------- | ----- | ---- | --- |
| $z_0$            | 0.003 | 0.05 | 0.3 |

$$ k_r = 0.19 * \ln(\frac{z_0}{z_{0,II}}) ^ {0.07} $$ 

$$ V_m = V_{b,0} * k_r * \ln(\frac{h}{z_0}) * \sin(\delta) $$

$$ I_v = \frac{1}{\ln(\frac{h}{z_0})} $$

Finally, the peak pressure equals to:

$$
    q_p = \frac{1}{2} * \rho * V_{m}^2 * (1 + 7 * I_v) * C_SC_D * C_f
$$


### References

All thoses formulas can be found in the following documents:

EN 1991-1-4, 2005. Eurocode 1: Actions on Structures – Part 1-4: General Actions –
Wind Actions. CEN, European Committee for Standardization, Brussels,
Belgium.

EN 1991-1-4/NA, 2005. Eurocode 1: Actions on Structures – Part 1-4: General Actions –
Wind Actions - National annex to NF EN 1991-1-4:2005. CEN, European Committee for Standardization, Brussels,
Belgium.

Ducloux, Hervé & Figueroa, Lionel. (2016). Background information about the wind action model of CENELEC EN 50341-1 (2012) and associated expected reliability of electrical overhead lines. Journal of Wind Engineering and Industrial Aerodynamics. 157. 104-117. 10.1016/j.jweia.2016.08.006. 

EN 50341-1, 2015. Overhead electrical lines exceeding AC 1 kV - Part 1 : general requirements - Common specifications. CEN, European Committee for Standardization, Brussels,
Belgium.