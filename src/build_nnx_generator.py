from jax import Array, config
import os
from flaxmodels.stylegan2.generator import MappingNetwork, SynthesisBlock, RESOLUTION
import jax.random as jrndm
from jax.numpy import ones, repeat, array as jarray, float32 as jf32, max as jmax, min as jmin, log2
from flax.nnx import Rngs, Module, state, jit as njit
from flax.nnx import bridge, List as nnxlist
from h5py import File
from numpy import clip, uint8
from PIL import Image
from functools import partial
from typing import Optional

dataset = 'afhqdog'

param_dict = File(rf'weights/flaxmodels/stylegan2_generator_{dataset}.h5','r')['synthesis_network']
#@njit
def build_mapping_network():
    mapping_network = MappingNetwork(pretrained=dataset,ckpt_dir='weights')
    dummy_z = ones((1, 512))
    nnx_mapping_network = bridge.ToNNX(mapping_network, rngs=Rngs(42))
    bridge.lazy_init(nnx_mapping_network, dummy_z)
    return nnx_mapping_network

#@njit
def build(rng:jrndm.PRNGKey = None):
    """
    Run Synthesis Network.
    Args:
        dlatents_in (tensor): Intermediate latents (W) of shape [N, num_ws, w_dim].
        noise_mode (str): Noise type.
                          - 'const': Constant noise.
                          - 'random': Random noise.
                          - 'none': No noise.
        rng (jrndm.PRNGKey): PRNG for spatialwise noise.
    Returns:
        (tensor): Image of shape [N, H, W, num_channels].
    """
    noise_mode='random'
    num_channels = 3
    fmap_base: int=16384
    fmap_decay: int=1
    fmap_min: int=1
    fmap_max: int=512
    # fmap_const: Optional[int]=None
    activation = 'leaky_relu'
    dtype: str='float32'
    use_noise: bool=True
    resample_kernel: tuple=(1, 3, 3, 1)
    fused_modconv: bool=False
    num_fp16_res: int=0
    clip_conv: Optional[float]=None
    blocks_list = []
    latent_code = jrndm.normal(jrndm.PRNGKey(0), (1, 512))
    nnx_mapping_network = build_mapping_network()
    dlatents_in = nnx_mapping_network(latent_code, skip_w_avg_update=True)
    resolution = RESOLUTION[dataset]
    
    if rng is None:
        rng = jrndm.PRNGKey(42)
    resolution_log2 = log2(resolution).astype(int)
    #assert resolution == int(2 ** resolution_log2) and resolution >= 4
    def nf(stage): return clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

    
    # if param_dict is None:
    #     const = jrndm.normal(rng, (1, 4, 4, fmaps), dtype=dtype)

    x = param_dict['const'][()]
    x = repeat(x, repeats=dlatents_in.shape[0], axis=0)
    y = None
    dlatents_in = dlatents_in.astype(jf32)
    
    init_rng = rng
    for res in range(2, resolution_log2 + 1):
        init_rng, init_key = jrndm.split(init_rng)
        mod = SynthesisBlock(fmaps=nf(res - 1),
                              res=res,
                              num_layers=1 if res == 2 else 2,
                              num_channels=num_channels,
                              activation=activation,
                              use_noise=use_noise,
                              resample_kernel=resample_kernel,
                              fused_modconv=fused_modconv,
                              param_dict=param_dict[f'block_{2 ** res}x{2 ** res}'] if param_dict is not None else None,
                              clip_conv=clip_conv,
                              dtype=dtype if res > resolution_log2 - num_fp16_res else 'float32',
                              rng=init_key)
        nnx_mod = bridge.ToNNX(mod, rngs = Rngs(0)) 
        bridge.lazy_init(nnx_mod,x,y, dlatents_in, noise_mode, rng)
        x, y = nnx_mod(x,y, dlatents_in, noise_mode, rng)
        blocks_list.append(nnx_mod)
        
    return nnxlist(blocks_list)



class StyleGAN_Generator(Module):
    def __init__(self, noise_mode:str = 'random'):
        self.noise_mode = noise_mode
        self.blocks = build()
        self.const = param_dict['const'][()]
        self.resolution = RESOLUTION[dataset]
    
    @njit(static_argnames='cutoff')  
    def __call__(self, w: Array, cutoff = None, rng = None):
        if rng is None:
            rng = jrndm.PRNGKey(42)
        x = self.const
        x = repeat(x, repeats=w.shape[0], axis=0)
        y = None
        for i,layer in enumerate(self.blocks):
            if cutoff and i >= cutoff:
                return x
            x, y = layer(x, y, w, self.noise_mode, rng)
        return y
    

if __name__ =='__main__':      
    
    generator_ = StyleGAN_Generator()
    mapping_network = build_mapping_network()
    rng = jrndm.PRNGKey(42)
    latent_code = jrndm.normal(rng, (3, 512))
    w_code = mapping_network(latent_code)
    generated_images = (generator_(w_code))[0]

    images = (generated_images - jmin(generated_images)) / (jmax(generated_images) - jmin(generated_images))

    for i in range(images.shape[0]):
        Image.fromarray(uint8(images[i] * 255)).save(f'image_{i}.jpg')
