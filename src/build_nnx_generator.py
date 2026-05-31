from jax import Array, config
import os
config.update("jax_compilation_cache_dir", os.path.join(os.path.dirname(__file__), ".jax_cache"))
from flaxmodels.stylegan2.generator import MappingNetwork, SynthesisBlock, RESOLUTION
import jax.random as jrndm
from jax.numpy import ones, repeat, array as jarray, float32 as jf32, max as jmax, min as jmin
from flax.nnx import Rngs, Module, state, jit as njit
from flax.nnx import bridge, List as nnxlist
from h5py import File
from numpy import clip, uint8
from PIL import Image
from functools import partial



mapping_network = MappingNetwork(pretrained='ffhq',ckpt_dir='weights')
dummy_z = ones((1, 512))
nnx_mapping_network = bridge.ToNNX(mapping_network, rngs=Rngs(42))
bridge.lazy_init(nnx_mapping_network, dummy_z)

latent_code = jrndm.normal(jrndm.PRNGKey(0), (1, 512))
w = nnx_mapping_network(latent_code)

param_dict = File(r'weights/flaxmodels/stylegan2_generator_ffhq.h5','r')['synthesis_network']
resolution = RESOLUTION['ffhq']

resolution_ = resolution
param_dict_ = param_dict
num_channels = 3
fmap_base: int=16384
fmap_decay: int=1
fmap_min: int=1
fmap_max: int=512
fmap_const: int=None
activation = 'leaky_relu'
dtype: str='float32'
use_noise: bool=True
resample_kernel: tuple=(1, 3, 3, 1)
fused_modconv: bool=False
num_fp16_res: int=0
clip_conv: float=None

blocks_list = []

def build(dlatents_in, noise_mode='random', rng=None):
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
    if rng is None:
        rng = jrndm.PRNGKey(42)
    resolution_log2 = 10
    assert resolution_ == 2 ** resolution_log2 and resolution_ >= 4
    def nf(stage): return clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    
    num_layers = resolution_log2 * 2 - 2
    
    fmaps = fmap_const if fmap_const is not None else nf(1)
    
    if param_dict_ is None:
        const = jrndm.normal(rng, (1, 4, 4, fmaps), dtype=dtype)
    else:
        const = jarray(param_dict_['const'], dtype=dtype)
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
                              param_dict=param_dict_[f'block_{2 ** res}x{2 ** res}'] if param_dict_ is not None else None,
                              clip_conv=clip_conv,
                              dtype=dtype if res > resolution_log2 - num_fp16_res else 'float32',
                              rng=init_key)
        nnx_mod = bridge.ToNNX(mod, rngs = Rngs(0)) 
        bridge.lazy_init(nnx_mod,x,y, dlatents_in, noise_mode, rng)
        x, y = nnx_mod(x,y, dlatents_in, noise_mode, rng)
        blocks_list.append(nnx_mod)
        
    return y

build(w)

class StyleGAN_Generator(Module):
    def __init__(self, noise_mode:str = 'random'):
        self.mapping_network = nnx_mapping_network
        self.const = param_dict['const'][()]
        self.noise_mode = noise_mode
        self.blocks = nnxlist(blocks_list)
        
    def __call__(self, z: Array, cutoff = None, rng = None):
        outputs_list = []
        if rng is None:
            rng = jrndm.PRNGKey(42)
        w = self.mapping_network(z)
        x = self.const
        x = repeat(x, repeats=z.shape[0], axis=0)
        y = None
        for i,layer in enumerate(self.blocks):
            if cutoff and i >= cutoff:
                break  
            x, y = layer(x, y, w, self.noise_mode, rng)
            outputs_list.append(x)
            
        return y, outputs_list
    
generator_ = StyleGAN_Generator()

if __name__ =='__main__':      
    
    
    rng = jrndm.PRNGKey(42)
    latent_code = jrndm.normal(rng, (3, 512))

    # generated_images = (generator_(latent_code))[0]

    # images = (generated_images - jmin(generated_images)) / (jmax(generated_images) - jmin(generated_images))

    # for i in range(images.shape[0]):
    #     Image.fromarray(uint8(images[i] * 255)).save(f'image_{i}.jpg')
    # state = state(model_)
    # checkpointer = ocp.PyTreeCheckpointer()
    # checkpointer.save('C:/Users/anura/DragGAN-JAX/models/ffhq_styleGAN', state)