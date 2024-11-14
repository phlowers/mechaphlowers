# General concepts

MechaPhlowers is a library specialized to perform mechanical and geometrical calculus for powerlines. The models are designed to be the simplest and most accurate possible : when possible, calculus are performed in 2D planes.

The granularity is important : the line is composed by several section. For each section, you have several cables. Each cable can be divided into spans, which is the lowest scale of calculus.

[scheme granularity]
Câble : Une phase peut être composée d’un unique câble ou d’un faisceau de câbles. Dans la suite on parle de câble de manière générique : phase constituée d'un câble simple, phase constituée d'un faisceau de câble, ou câble de garde.

The algorithms are switching between the different levels during the resolution

## Definition

![Image not available](./assets/powerline_definitions.drawio.png "Powerlines definitions")

As you can see below, the study cable is hanging to different suspension strings and start/stopping to a dead end string.

Occasionnaly, some line angles can be found into a section. In this case, we are considering the axe of the pylone (i.e. the arm direction) as the angle bisector.

## Frames

Depending on the physical models used in the algorithms, the 3D objects, coordinates, forces and moments are projected onto different planes.

The comprehension of those planes are not mandatory to use the package, but needed if you want to dive into the different physical models used.

However the different frames defined below are needed to display the result : that is why we are defining the transformation operations to go from one to another.

### Span frame

![Image not available](./assets/span_frame.drawio.png "Span frame")

### Arm pylon frame


### Pylon frame


### Earth frame




