Implementation of DragGAN in JAX
Source - [Arxiv](https://arxiv.org/pdf/2305.10973)

To do
- [x] Get a StyleGAN in JAX whose intermediate layers are available to observe
- [ ] Investigate quality of StyleGAN feature maps
- [ ] Implement the motion supervised loss as a function
- [ ] Implement the motion tracking algorithm
- [ ] Write a optimisation step which uses the loss to optimise _w_ and motion tracking to get new handle point
- [ ] Put it all together, to get the backend of DragGAN
- [ ] Investigate how to improve StyleGAN implementation and performance
- [ ] Figure out the GUI for this