# General concepts

MechaPhlowers is a library specialized to perform mechanical and geometrical calculus for powerlines. The models are designed to be the simplest and most accurate possible : when possible, calculus are performed in 2D planes.

The granularity is important : the line is composed by several section. For each section, there are several cables. Each cable can be divided into spans, which is the lowest scale of calculus.

The algorithms are switching between the different levels during the resolution.

## Definition

![Image not available](./assets/powerline_definitions.drawio.png "Powerlines definitions")

As described below, the study cable is hanging to different suspension strings and start/stopping to a tension support.

Occasionnaly, some line angles can be found into a section. In this case, we are considering the axe of the pylone (i.e. the arm direction) as the angle bisector.

## Frames

Depending on the physical models used in the algorithms, the 3D objects, coordinates, forces and moments are projected onto different planes.

Understanding these planes is not required to use the package, but it is necessary if you wish to explore the various physical models employed.

However, the different frames defined below are necessary to display the results. Therefore, we are outlining the transformation operations to transition between them.

### Earth frame

#### Base frame

The Earth frame  $\mathcal{R}_{earth}$ is defined as the GPS coordinates system :

- x along the west-east axis, facing east
- y along the north-south axis, facing north
- x and y in the plan, z is orthogonal to this plan
- the origin of the frame is the crossing of equator and prime meridian.

!!! important

    For now, only the local frame is considered (in blue in the figure), with the hypotheses that the origin is a plane translation relative to the O point in the figure.

![Image not available](./assets/earth_frame.drawio.png "Earth frame")


#### Georeferencing

Other coordinates system may be added in the future.

### Support frame

There are two frames associated with the support:

- The tower body frame: $\mathcal{R}_{towerbody}$
- The crossarm frame: $\mathcal{R}_{crossarm}$

Those frames are oriented with z axis up and x axis along the crossarm, pointing outwards from the pylon.

![Image not available](./assets/support_frame.drawio.png "Support frame")

### Cable frame / Span frame

The reference support frame for a span is the left support depending on line direction evolution.

The cable frame $\mathcal{R}_{cable}$ is defined as described in the figure below. This frame is then moving depending on cable loads.

![Image not available](./assets/cable_frame.drawio.png "Cable frame")

### Change between different frames

In this section the rotational and translational mapping of the frames are defined.

Definition of rotation of an angle $\theta$ matrix for a frame $\mathcal{R}(O,x,y,z)$ regarding the 3 different axis:

$${\displaystyle R_{{x} }(\theta )={\begin{pmatrix}1&0&0\\0&\cos \theta &-\sin \theta \\0&\sin \theta &\cos \theta \end{pmatrix}}\qquad R_{ {y} }(\theta )={\begin{pmatrix}\cos \theta &0&\sin \theta \\0&1&0\\-\sin \theta &0&\cos \theta \end{pmatrix}}\qquad R_{ {z} }(\theta )={\begin{pmatrix}\cos \theta &-\sin \theta &0\\\sin \theta &\cos \theta &0\\0&0&1\end{pmatrix}}}$$

The transformation from one frames to another is composed of a rotation and a translation $(R,T)$.

- From $\mathcal{R}_{earth}$ to $\mathcal{R}_{towerbody}$: 

    - $T=\begin{pmatrix}x_{support GPS}\\y_{support GPS}\\h_{crossarm}\end{pmatrix}_{\mathcal{R}_{earth}}$
    - $R=R_z(\theta_{support})$

- From $\mathcal{R}_{towerbody}$ to $\mathcal{R}_{crossarm}$: 

    - $T=\begin{pmatrix}L_{arm}\\y_{support GPS}\\0\end{pmatrix}_{\mathcal{R}_{towerbody}}$
    - $R=R_z(\gamma)$ 

- From $\mathcal{R}_{crossarm}$ to $\mathcal{R}_{cable}$:

    - $T=\begin{pmatrix}x_{O_{low}}\\y_{O_{low}}\\0\end{pmatrix}_{\mathcal{R}_{crossarm}}$
    - $R=R_x(\beta)\cdot R_z(\alpha)$ 










