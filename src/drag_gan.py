import time
start = time.time()
import jax.numpy as jnp
import jax
from build_nnx_generator import build_mapping_network, StyleGAN_Generator
print("Finished importing - Generator at time - ", time.time() - start)
import jax.random as jrndm
import utils
from flax import nnx
import optax
import numpy as np
print("Finished importing other things at - ", time.time() - start)

class DragGan(nnx.Module):
    def __init__(self):
        self.mapping_network = build_mapping_network()
        self.synthesis_network = StyleGAN_Generator()
        self.cutoff_block = 7
        self.resolution = self.synthesis_network.resolution
        
    def get_dlatent_loss(self, w_code, pi, ti, ri = None):
        feature_map = self.synthesis_network(w_code, cutoff = self.cutoff_block)
        return self.motion_supervision_loss(feature_map, pi, ti, self.resolution, ri)
    
    def optimise_dlatent_single_it(self, optimizer: optax.GradientTransformationExtraArgs, opt_state, w_code, pi, ti, ri = None):
        loss, grads = nnx.value_and_grad(DragGan.get_dlatent_loss, argnums = 1)(self, w_code, pi, ti, ri)
        updates, opt_state = optimizer.update(grads, opt_state)
        w_code = optax.apply_updates(w_code, updates)
        return loss, w_code, opt_state
    
    def generate_image(self, rng = None):
        if rng is None:
            rng = jrndm.PRNGKey(42)
            
        latent_code = jrndm.normal(rng, (1, 512))
        w_code = self.mapping_network(latent_code)
        generated_images = (self.synthesis_network(w_code))[0]
        image = (generated_images - jnp.min(generated_images)) / (jnp.max(generated_images) - jnp.min(generated_images))
        image = np.array(image)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image, w_code
    
    
    def motion_supervision_loss(self, feature_map, pi: tuple, ti:tuple, res, ri = None):
        resized_map = jax.image.resize(feature_map, shape=(feature_map.shape[0], res, res, feature_map.shape[-1]), method="bilinear")
        if not ri:
            ri = (res*res)//5e4

        if not len(pi) == 2 or not len(ti) == 2:
            raise ValueError(f"Expected args pi and ti to be tuples of len 2. Got tuples of length {len(pi)} and {len(ti)} instead")

        square_length = int(((2**0.5)*ri)/2.0)
        loss = 0
        denominator = jnp.sqrt((ti[0] - pi[0])**2 + (ti[1] - pi[1])**2)
        di = ((ti[0] - pi[0])/denominator, (ti[1] - pi[1])/denominator)

        for j in range(-square_length, square_length, 1):
            for i in range(-square_length, square_length, 1):
                detached_point = jax.lax.stop_gradient(resized_map[(pi[0]+j, pi[1]+i)])
                loss += jnp.linalg.norm(detached_point - resized_map[(jnp.ceil(pi[0] + j + di[0]).astype(int), jnp.ceil(pi[1] + i + di[1]).astype(int))],ord = 1)
        print("Loss - ",loss)
        return loss

    
    def dummy_try(self, rng = None):
        image, w_code = self.generate_image(rng)
        print("Generated image at - ", time.time() - start)
        pi, ti = utils.get_drag_points(image)
        optimizer = optax.adam(2e-3)
        opt_state = optimizer.init(w_code)
        loss, new_w, state = self.optimise_dlatent_single_it(optimizer, opt_state, w_code, pi, ti)
        print(loss)
        

if __name__ == '__main__':
    model = DragGan()
    print("Finished instantiating generator at - ", time.time() - start)
    model.dummy_try()
    
         
        
        
        


