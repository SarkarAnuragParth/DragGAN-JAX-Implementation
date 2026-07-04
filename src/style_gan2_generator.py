import jax
import os
cache_dir = os.path.join(os.getcwd(), ".jax_cache")
jax.config.update("jax_compilation_cache_dir", cache_dir)
import math
import flaxmodels.stylegan2 as fstg2
from flaxmodels.stylegan2.ops import conv2d
import jax.numpy as jnp
import flax.nnx as nnx
import h5py
import utils
from PIL import Image
from numpy import uint8

def create_kernel_initializer(param_dict, in_features, out_features, lr_multiplier:float = 1.0, layer_name = None, key = None):
    kernel_weights, bias_weights = fstg2.ops.get_weight(shape = [in_features, out_features], param_dict=param_dict, layer_name=layer_name, key = key)
    
    
    scaled_kernel = utils.equalize_lr_weight(kernel_weights, lr_multiplier)
    scaled_bias = utils.equalize_lr_bias(bias_weights, lr_multiplier)
    
    def kernel_weight_init(key, shape, dtype=None, out_sharding = None):
        return scaled_kernel
    
    def bias_weight_init(key, shape, dtype=None, out_sharding = None):
        return scaled_bias
        
    return kernel_weight_init, bias_weight_init

def leaky_relu_alpha_fixed(x: jax.typing.ArrayLike) -> jax.Array:
    return nnx.leaky_relu(x, 0.2)*jnp.sqrt(2)


class ModConv2D(nnx.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        kernel_size: int, 
        rngs: nnx.Rngs = None,
        kernel_weights: jax.Array = None,
        up: bool = False, 
        down: bool = False, 
        demodulate: bool = True, 
        fused_modconv: bool = False
    ):
        assert not (up and down)
        
        self.out_features = out_features
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.fused_modconv = fused_modconv

        if kernel_weights is not None:
            self.weight = nnx.Param(kernel_weights)
        else:
            w_shape = (kernel_size, kernel_size, in_features, out_features)
            self.weight = nnx.Param(jax.random.normal(rngs.params(), w_shape))

    def __call__(self, x: jax.Array, s: jax.Array, resample_kernel=None) -> jax.Array:
        w = self.weight.value

        if x.dtype in (jnp.float16, jnp.bfloat16) and not self.fused_modconv and self.demodulate:
            w *= jnp.sqrt(1 / math.prod(w.shape[:-1])) / jnp.max(jnp.abs(w), axis=(0, 1, 2))
            
        ww = w[jnp.newaxis]

        if x.dtype in (jnp.float16, jnp.bfloat16) and not self.fused_modconv and self.demodulate:
            s *= 1 / jnp.max(jnp.abs(s))
            
        ww *= s[:, jnp.newaxis, jnp.newaxis, :, jnp.newaxis].astype(w.dtype)

        if self.demodulate:
            d = jax.lax.rsqrt(jnp.sum(jnp.square(ww), axis=(1, 2, 3)) + 1e-8)
            ww *= d[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

        if self.fused_modconv:
            x = jnp.transpose(x, (0, 3, 1, 2))
            x = jnp.transpose(jnp.reshape(x, (1, -1, x.shape[2], x.shape[3])), (0, 2, 3, 1))
            w_conv = jnp.reshape(jnp.transpose(ww, (1, 2, 3, 0, 4)), (ww.shape[1], ww.shape[2], ww.shape[3], -1))
        else:
            x *= s[:, jnp.newaxis, jnp.newaxis].astype(x.dtype)
            w_conv = w.astype(x.dtype)

        # Assumes your custom conv2d is defined in the same scope
        x = conv2d(x, w_conv, up=self.up, down=self.down, resample_kernel=resample_kernel)

        if self.fused_modconv:
            x = jnp.transpose(x, (0, 3, 1, 2))
            x = jnp.transpose(jnp.reshape(x, (-1, self.out_features, x.shape[2], x.shape[3])), (0, 2, 3, 1))
        elif self.demodulate:
            x *= d[:, jnp.newaxis, jnp.newaxis].astype(x.dtype)
        
        return x


class MappingNetwork(nnx.Module):
    def __init__(self, z_dim: int = 512, w_dim: int = 512, hidden_features:int = 512, w_plus_nums: int = 18, pretrained = None, param_dict = None, rngs =  nnx.Rngs(0)):
        self.w_plus = w_plus_nums
        num_layers = fstg2.generator.NUM_MAPPING_LAYERS[pretrained]
        linear_layers=  []
        in_features = z_dim
        out_features = [hidden_features]*(num_layers-1) + [w_dim]
        for i in range(num_layers):
            kernel_init, bias_init = create_kernel_initializer(param_dict,
                                                                    in_features, 
                                                                    out_features[i],
                                                                    lr_multiplier=0.01,
                                                                    layer_name = f'fc{i}')
            layer = nnx.Linear(in_features = in_features,
                               out_features = out_features[i],
                               kernel_init = kernel_init,
                               bias_init = bias_init,
                               rngs = rngs)
            linear_layers.append(layer)
            linear_layers.append(leaky_relu_alpha_fixed)
            in_features = hidden_features
        
        self.model = nnx.Sequential(*linear_layers)
        if param_dict:
            w_avg = param_dict['w_avg']
        else:
            w_avg = jnp.zeros(shape=w_dim)
        
        self.w_avg = jnp.array(w_avg)

        
    @nnx.jit
    def __call__(self, z, truncation_psi: float = 1.0):
        x = fstg2.ops.normalize_2nd_moment(z.astype(jnp.float32))
        x = self.model(x)
        x = jnp.repeat(jnp.expand_dims(x, axis=-2), repeats=self.w_plus, axis=-2)
        x = x.at[:, :].set(truncation_psi * x[:, :] + (1 - truncation_psi) * self.w_avg.astype(jnp.float32))
        return x


class SynthesisLayer(nnx.Module):
    def __init__(self, filter_size:int, n_output_channels:int, n_input_channels: int, up: bool, output_res, layer_idx:int, w_dim: int = 512, pretrained_dataset:str = None, param_dict = None, rng = None):
        
        self.layer_idx = layer_idx
        
        kernel_init, base_bias_init = create_kernel_initializer(param_dict,
                                                           w_dim,
                                                           n_input_channels,
                                                           layer_name = 'affine')
        
        def bias_init_with_offset(key, shape, dtype=None,  out_sharding = None):
            return base_bias_init(key, shape, dtype,  out_sharding) + 1.0
        
        self.style_embedding_layer = nnx.Linear(in_features= w_dim,
                                                out_features = n_input_channels,
                                                kernel_init = kernel_init,
                                                bias_init = bias_init_with_offset,
                                                rngs = rng)
        if param_dict is None:
            noise_strength = jnp.zeros(())
        else:
            noise_strength = jnp.array(param_dict['noise_strength']) 
        self.noise_strength = nnx.Param(noise_strength)
        
        if param_dict is not None:
            noise_const = jnp.array(param_dict['noise_const'], dtype=jnp.float32)
        else:
            noise_const = jax.random.normal(rng.params(), shape=(1, 2 **output_res , 2 **output_res , 1), dtype=jnp.float32)
        self.noise_const = nnx.Variable(noise_const)
        
        w, b = fstg2.ops.get_weight(shape = [filter_size, filter_size, n_input_channels, n_output_channels], param_dict = param_dict, layer_name = 'conv')
        w = utils.equalize_lr_weight(w, lr_multiplier=1.0)
        b = utils.equalize_lr_bias(b, lr_multiplier=1.0)
        self.bias= nnx.Param(b)
        self.modconv_layer = ModConv2D(n_input_channels,
                                             n_output_channels,
                                             3,
                                             kernel_weights=w,
                                             up = up,
                                             fused_modconv=True)
        

    def __call__(self, x, dlatents, resample_kernel = (1,3,3,1), rng = None):
        
        s = self.style_embedding_layer(dlatents[:, self.layer_idx])
        x = self.modconv_layer(x, s, resample_kernel)
        x += self.noise_const.astype(jnp.float32) * self.noise_strength.astype(jnp.float32)
        x += self.bias.astype(x.dtype)
        x = nnx.leaky_relu(x, 0.2)
        x *= jnp.sqrt(2)
        return x
            

class ToRGBLayer(nnx.Module):
    def __init__(self, n_input_channels: int, fmaps: int, layer_idx: int, w_dim: int = 512, kernel_size: int = 1, param_dict=None, rng=None):
        self.layer_idx = layer_idx
        self.fmaps = fmaps

        kernel_init, base_bias_init = create_kernel_initializer(
            param_dict, w_dim, n_input_channels, layer_name='affine'
        )
        
        def bias_init_with_offset(key, shape, dtype=None,  out_sharding = None):
            return base_bias_init(key, shape, dtype, out_sharding) + 1.0

        self.style_embedding_layer = nnx.Linear(
            in_features=w_dim,
            out_features=n_input_channels,
            kernel_init=kernel_init,
            bias_init=bias_init_with_offset,
            rngs = rng
        )

        w, b = fstg2.ops.get_weight(
            shape=[kernel_size, kernel_size, n_input_channels, fmaps], 
            param_dict=param_dict,
            layer_name = 'conv'
        )
        w = utils.equalize_lr_weight(w, lr_multiplier=1.0)
        b = utils.equalize_lr_bias(b, lr_multiplier=1.0)
        
        self.bias = nnx.Param(b)
        
        # CRITICAL: demodulate=False for ToRGB layers
        self.modconv_layer = ModConv2D(
            in_features=n_input_channels,
            out_features=fmaps,
            kernel_size=kernel_size,
            kernel_weights=w,
            demodulate=False,
            fused_modconv=True 
        )

    def __call__(self, x, y, dlatents):
        s = self.style_embedding_layer(dlatents[:, self.layer_idx])
        
        x = self.modconv_layer(x, s)
        x += self.bias.astype(x.dtype)
        
        if y is not None:
            x += y.astype(x.dtype)
            
        return x


class SynthesisBlock(nnx.Module):
    def __init__(
        self, 
        in_channels: int, 
        fmaps: int, 
        res: int, 
        num_layers: int = 2, 
        num_channels: int = 3, 
        w_dim: int = 512, 
        param_dict = None, 
        rngs: nnx.Rngs = None
    ):
        self.res = res
        self.num_layers = num_layers
        
        self.layers = nnx.List()
        curr_in_channels = in_channels
        
        for i in range(num_layers):
            is_up = (i == 0 and res != 2)
            layer_idx = res * 2 - (5 - i) if res > 2 else 0
            layer_param = param_dict[f'layer{i}'] if param_dict is not None else None
            
            self.layers.append(
                SynthesisLayer(
                    filter_size=3,
                    n_output_channels=fmaps,
                    n_input_channels=curr_in_channels,
                    up=is_up,
                    output_res=res,
                    layer_idx=layer_idx,
                    w_dim=w_dim,
                    param_dict=layer_param,
                    rng=rngs
                )
            )
            # After the first layer, the input channels for the next layer will match fmaps
            curr_in_channels = fmaps
            
        torgb_param = param_dict['torgb'] if param_dict is not None else None
        
        self.torgb = ToRGBLayer(
            n_input_channels=fmaps,
            fmaps=num_channels,
            layer_idx=res * 2 - 3,
            w_dim=w_dim,
            kernel_size=1,
            param_dict=torgb_param,
            rng=rngs
        )

    def __call__(self, x, y, dlatents, resample_kernel=(1, 3, 3, 1), rng=None):
        # 1. Run through the synthesis convolution layers
        for layer in self.layers:
            x = layer(x, dlatents, resample_kernel=resample_kernel, rng = rng)
            
        if y is not None and self.res != 2:
            k = fstg2.ops.setup_filter(resample_kernel)
            y = fstg2.ops.upsample2d(y, f=k, up=2)
            
        y = self.torgb(x, y, dlatents)
        
        return x, y

class SynthesisNetwork(nnx.Module):
    def __init__(
        self,
        num_channels: int = 3,
        w_dim: int = 512,
        fmap_base: int = 16384,
        fmap_decay: int = 1,
        fmap_min: int = 1,
        fmap_max: int = 512,
        fmap_const: int = None,
        pretrained_dataset: str = None,
        param_dict = None,
        rngs: nnx.Rngs =  nnx.Rngs(0)
    ):
        self.resolution = fstg2.generator.RESOLUTION[pretrained_dataset]
        resolution_log2 = int(jnp.log2(self.resolution))
        assert self.resolution == 2 ** resolution_log2 and self.resolution >= 4

        def nf(stage):
            return int(jnp.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max))

        # 1. Initialize Constant Input
        fmaps_in = fmap_const if fmap_const is not None else nf(1)
        if param_dict is not None and 'const' in param_dict:
            const = jnp.array(param_dict['const'], dtype=jnp.float32)
        else:
            const = jax.random.normal(rngs.params(), (1, 4, 4, fmaps_in), dtype=jnp.float32)
            
        self.const = nnx.Param(const)

        # 2. Initialize Synthesis Blocks
        self.blocks = nnx.List()
        curr_in_channels = fmaps_in

        for res in range(2, resolution_log2 + 1):
            block_fmaps = nf(res - 1)
            num_layers = 1 if res == 2 else 2
            block_param = param_dict[f'block_{2 ** res}x{2 ** res}'] if param_dict is not None else None
            
            self.blocks.append(
                SynthesisBlock(
                    in_channels=curr_in_channels,
                    fmaps=block_fmaps,
                    res=res,
                    num_layers=num_layers,
                    num_channels=num_channels,
                    w_dim=w_dim,
                    param_dict=block_param,
                    rngs=rngs
                )
            )
            curr_in_channels = block_fmaps

    @nnx.jit(static_argnames='cutoff')
    def __call__(self, dlatents: jax.Array, cutoff: int = None, resample_kernel=(1, 3, 3, 1), rng=nnx.Rngs(0)):
        batch_size = dlatents.shape[0]
        
        # Broadcast constant input across the batch dimension
        x = jnp.repeat(self.const.astype(jnp.float32), repeats=batch_size, axis=0)
        y = None
        
        for i, block in enumerate(self.blocks):
            if cutoff is not None and i >= cutoff:
                return x 
            
            x, y = block(x, y, dlatents, resample_kernel=resample_kernel, rng=rng)
            
        return y
        
if __name__ == '__main__':
    dataset = 'ffhq'
    param_dict = h5py.File(rf'weights/flaxmodels/stylegan2_generator_{dataset}.h5','r')
    mapping_network = MappingNetwork(pretrained = 'ffhq', param_dict=param_dict['mapping_network'])
    synthesis_network = SynthesisNetwork(pretrained_dataset='ffhq', param_dict=param_dict['synthesis_network'])
    print("Models initialized")
    z = jax.random.normal(key = jax.random.PRNGKey(234), shape = (1,512), dtype = jnp.float32)
    w = mapping_network(z)
    generated_images = synthesis_network(w)
    image = (generated_images - jnp.min(generated_images)) / (jnp.max(generated_images) - jnp.min(generated_images))
    image = jnp.clip(image * 255, 0, 255)
    Image.fromarray(uint8(image[0])).save("Original_image_1.jpg")