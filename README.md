Implementation of DragGAN in JAX

Source - [Arxiv](https://arxiv.org/pdf/2305.10973)

To do
- [x] Get a StyleGAN in JAX whose intermediate layers are available to observe
- [x] Investigate quality of StyleGAN feature maps (Preliminary done, but follow up on some observations later)
- [x] Implement the motion supervised loss as a function & test if it is able to update _w_ code
- [x] Implement the motion tracking algorithm
- [x] Write a optimisation step which uses the loss to optimise _w_ and motion tracking to get new handle point
- [x] Put it all together, to get the backend of DragGAN
- [ ] Investigate how to improve DragGAN implementation and performance
- [ ] Figure out the GUI for this