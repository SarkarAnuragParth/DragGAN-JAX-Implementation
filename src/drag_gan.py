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
from PIL import Image
print("Finished importing other things at - ", time.time() - start)

class DragGan(nnx.Module):
    def __init__(self):
        self.mapping_network = build_mapping_network()
        self.synthesis_network = StyleGAN_Generator()
        self.cutoff_block = 7
        self.resolution = self.synthesis_network.resolution
        self.cache = {}
        
    def get_dlatent_loss(self, w_code, pi, ti, r1 = None):
        feature_map = self.synthesis_network(w_code, cutoff = self.cutoff_block)
        loss = self.motion_supervision_loss(feature_map, pi, ti, r1)
        return loss, feature_map
        
    def optimise_dlatent_single_it(self, optimizer: optax.GradientTransformationExtraArgs, opt_state, w_code, pi, ti, r1 = None):
        (loss, feature_map), grads = nnx.value_and_grad(DragGan.get_dlatent_loss, 
                                                        argnums = 1, 
                                                        has_aux = True)(self, w_code, pi, ti, r1)
        updates, opt_state = optimizer.update(grads, opt_state)
        w_code = optax.apply_updates(w_code, updates)
        
        self.cache[f'feature_map_{self.cutoff_block}_block_old'] = feature_map
        
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
    
    
    def get_boundaries(self, pi, r):
        square_length = jnp.ceil(((2**0.5)*r)/2.0).astype(int)
        x_min_b = -min(square_length,pi[0])
        x_max_b = min(square_length, self.resolution - pi[0])
        y_min_b = -min(square_length, pi[1])
        y_max_b = min(square_length, self.resolution - pi[1])
        return x_min_b, x_max_b, y_min_b, y_max_b
          
    
    def motion_supervision_loss(self, feature_map, pi: tuple, ti:tuple, r1 = None):
        resized_map = jax.image.resize(feature_map, shape=(feature_map.shape[0], self.resolution, self.resolution, feature_map.shape[-1]), method="bilinear")
        if not r1:
            r1 = (3*self.resolution)//512

        if not len(pi) == 2 or not len(ti) == 2:
            raise ValueError(f"Expected args pi and ti to be tuples of len 2. Got tuples of length {len(pi)} and {len(ti)} instead")

        x_min_b, x_max_b, y_min_b, y_max_b = self.get_boundaries(pi, r1)
        loss = 0
        denominator = ((ti[0] - pi[0])**2 + (ti[1] - pi[1])**2)**0.5
        di = ((ti[0] - pi[0]), (ti[1] - pi[1]))

        for j in range(x_min_b, x_max_b, 1):
            for i in range(y_min_b, y_max_b, 1):
                detached_point = jax.lax.stop_gradient(resized_map[(pi[0]+j, pi[1]+i)])
                loss += jnp.linalg.norm(detached_point - resized_map[((pi[0] + j + jnp.sign(di[0]))).astype(int), (pi[1] + i +  jnp.sign(di[1])).astype(int)],ord = 1)
        return loss


    def point_tracking(self, new_w, pi, r2 = None):
        if r2 is None:
            r2 = (12*self.resolution)//512
        new_feature_map = self.synthesis_network(new_w, cutoff = self.cutoff_block)
        old_feature_map = self.cache[f'feature_map_{self.cutoff_block}_block_old']
        resized_old_feature_map = jax.image.resize(old_feature_map, shape=(old_feature_map.shape[0], self.resolution, self.resolution, old_feature_map.shape[-1]), method="bilinear") 
        old_point = resized_old_feature_map[(pi[0],pi[1])]
        self.cache[f'feature_map_{self.cutoff_block}_block_new'] = new_feature_map
        resized_new_feature_map = jax.image.resize(new_feature_map, shape=(new_feature_map.shape[0], self.resolution, self.resolution, new_feature_map.shape[-1]), method="bilinear")
        
        x_min_b, x_max_b, y_min_b, y_max_b = self.get_boundaries(pi, r2)
        min_ = float('inf')
        x_new, y_new = None, None
        for j in range(x_min_b, x_max_b, 1):
            for i in range(y_min_b, y_max_b, 1):
                #print(jnp.linalg.norm(resized_new_feature_map[(pi[0]+j, pi[1]+i)] - old_point, ord=1))
                nrm = (jnp.linalg.norm(resized_new_feature_map[(pi[0]+j, pi[1]+i)] - old_point, ord=1))
                print(f" Min:{min_}   Norm: {nrm}    x,y: {(j,i)}")
                if min_ > nrm:
                    
                    min_ = nrm
                    x_new, y_new = j, i

        x_new_abs, y_new_abs = x_new + pi[0], y_new + pi[1]
        
        if (x_new_abs, y_new_abs) == pi:
            raise ValueError("Not  good")
        
        return (x_new_abs, y_new_abs)
                

    
    def loop(self, rng = None):
        image, w_code = self.generate_image(rng)
        print("Generated image at - ", time.time() - start)
        pi, ti = utils.get_drag_points(image)
        original_pi = pi
        optimizer = optax.adam(2e-3)
        opt_state = optimizer.init(w_code)
        ctr = 0

        while (abs(pi[0] - ti[0]) + abs(pi[1] - ti[1]) > 4) and ctr < 10:
            loss, new_w, opt_state = self.optimise_dlatent_single_it(optimizer, opt_state, w_code, pi, ti)
            print(f"Loss = {loss}         Change in w code: {jnp.linalg.norm(new_w[0] - w_code[0], ord = 1)}")
            new_point = self.point_tracking(new_w, pi)
            print(f"({int(new_point[0])}, {int(new_point[1])}) ---- {pi}")
            w_code = new_w
            pi = (int(new_point[0]), int(new_point[1]))
            ctr+=1

        new_image = self.synthesis_network(w_code)[0]
        new_image = (new_image - jnp.min(new_image)) / (jnp.max(new_image) - jnp.min(new_image))
        Image.fromarray(image).save("Original_image.jpg")
        Image.fromarray(np.uint8(new_image * 255)).save(f'image_modified_down.jpg')

        
        
        


if __name__ == '__main__':
    model = DragGan()
    print("Finished instantiating generator at - ", time.time() - start)
    model.loop()
    
         
        
        
        


