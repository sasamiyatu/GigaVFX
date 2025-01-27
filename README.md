# GigaVFX

This is a hobby project with the goal of exploring GPU-based particle system implementations. Particle emission and simulation both run fully in compute shaders
and all state related to particle systems is resident only on the GPU. Buffers and indirect dispatching is used to communicate between passes. Indirect draw is used to create draw calls for rendering on the GPU.

[Video demonstration](https://www.youtube.com/watch?v=AAf6r7EPoss)

![Alt text](screenshots/armadillo.png?raw=true "Burning armadillo")

## Particle systems

### Simple

Basic particle system using a spherical emitter with randomized direction. Gravity, drag and wind velocity are applied in each simulation step. Particles can collide with the ground plane. 

![Alt text](screenshots/simple.png?raw=true "Simple emitter")

### Smoke

Smoke simulation with volumetric shadows using [half-angle slice rendering](https://developer.download.nvidia.com/compute/DevZone/C/html_x64/5_Simulations/smokeParticles/doc/smokeParticles.pdf) and [curl noise](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf). Particles are sorted on the GPU using radix sort.

![Alt text](screenshots/smoke.png?raw=true "Smoke simulation")

### Disintegrator

Disintegration effect inspired by [God of War](https://gdcvault.com/play/1025973/Disintegrating-Meshes-with-Particles-in). Emits particles from depth prepass. Fragments are discarded based on a noise texture and an animated discard threshold within depth prepass, and for each particle that went from visible to not visible during the frame, a particle is emitted.

![Alt text](screenshots/armadillo.png?raw=true "Disintegrate effect")

### Trails

Uses two particle systems where one acts as the parent and one as the child. The movement is simulated in the parent system using [divergence-free noise](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7bb700941935fb37e14bbd3d39abfd5b8318b470) to generate a velocity field conforming to the surface of a sphere. The child system gets its dispatch size based on the number of the particles in the parent system and its own emission rate. Trails are formed by the child particle system.

![Alt text](screenshots/trails.png?raw=true "Disintegrate effect")

