from style_gan2_generator import MappingNetwork, SynthesisNetwork
import jax
import time
import utils
import jax.numpy as jnp
import jax.random as jrndm
from flax import nnx
import optax
import numpy as np
from PIL import Image
import pickle as pkl


class DragGan(nnx.Module):
    def __init__(self, feature_map = 6, w_cutoff = 6, pretrained_dataset = 'ffhq'):
        param_dict = utils.get_file(rf'weights/flaxmodels/stylegan2_generator_{pretrained_dataset}.h5')
        self.mapping_network = MappingNetwork(pretrained = pretrained_dataset, param_dict=param_dict['mapping_network'])
        self.synthesis_network = SynthesisNetwork(pretrained_dataset= pretrained_dataset, param_dict=param_dict['synthesis_network'])
        self.cutoff_block = feature_map
        self.resolution = self.synthesis_network.resolution
        self.cache = {}
        self.lamda = 20
        self.w_cutoff = w_cutoff

    def forward_pass(self, w_code, P, T, mask, offsets):
        feature_map = self.synthesis_network(w_code, cutoff = self.cutoff_block)
        resized_map = jax.image.resize(feature_map, shape=(feature_map.shape[0], self.resolution, self.resolution, feature_map.shape[-1]), method="bilinear")
        loss = self.motion_supervision_loss(resized_map, P, T, offsets)
        C = resized_map.shape[-1]
        mask_loss = jnp.linalg.norm((resized_map[0] - self.resized_original_feature_map[0])*mask, ord= 1, axis = -1 )
        final_loss = loss + self.lamda*jnp.sum(mask_loss)/(jnp.sum(mask)*C + 1e-8)
        return final_loss


    @nnx.jit(static_argnames=['optimizer'])
    def optimise_dlatent_single_it(self, optimizer: optax.GradientTransformationExtraArgs, opt_state, w_code, P, T, mask, offsets):

        loss, grads = nnx.value_and_grad(DragGan.forward_pass,
                                                        argnums = 1,
                                                        has_aux = False)(self, w_code, P, T, mask, offsets)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_w_code = optax.apply_updates(w_code, updates)
        cleaned_w_code = new_w_code.at[:,self.w_cutoff:,:].set(w_code[:,self.w_cutoff:,:])
        return loss, cleaned_w_code, new_opt_state


    def generate_image(self, truncation_psi, rng = None):
        if rng is None:
            rng = jrndm.PRNGKey(42)

        latent_code = jrndm.normal(rng, (1, 512))
        w_code = self.mapping_network(latent_code, truncation_psi=truncation_psi)
        generated_images = (self.synthesis_network(w_code))[0]
        image = (generated_images - jnp.min(generated_images)) / (jnp.max(generated_images) - jnp.min(generated_images))
        image = np.array(image)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image, w_code


    def bilinear_interpolate(self, feature_map, qi_x, qi_y, di_x, di_y):

        xp = qi_x + di_x[:,None,None]
        yp = qi_y + di_y[:,None,None]

        x_low = jnp.floor(xp).astype(jnp.int32)
        y_low = jnp.floor(yp).astype(jnp.int32)
        x_high = jnp.clip(x_low + 1, 0, self.resolution)
        y_high = jnp.clip(y_low + 1, 0, self.resolution)
        x_low = jnp.clip(x_low, 0, self.resolution)
        y_low = jnp.clip(y_low, 0, self.resolution)

        fx = jnp.clip(xp - jnp.floor(xp), 0., 1.)[:,:,:,None]
        fy = jnp.clip(yp - jnp.floor(yp), 0., 1.)[:,:,:,None]

        v00 = feature_map[0, x_low, y_low,:]
        v10 = feature_map[0, x_high, y_low,:]
        v01 = feature_map[0, x_low, y_high,:]
        v11 = feature_map[0, x_high, y_high,:]

        return v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy


    @nnx.jit
    def motion_supervision_loss(self, resized_map, P, T, offsets):
        
        n_points = P.shape[0]
        P_x, P_y = P[:,0], P[:,1]
        
        grid_x, grid_y = jnp.meshgrid(offsets, offsets, indexing='ij')

        denominators = jnp.sqrt(jnp.square(T[:,0] - P[:,0]) + jnp.square(T[:,1] - P[:,1]) + 1e-8)
        d_x = (T[:,0] - P[:,0])*(1/denominators)
        d_y = (T[:,1] - P[:,1])*(1/denominators)
        # jax.debug.print("d_x: {}", d_x)
        # jax.debug.print("d_y: {}", d_y)
        grid_x_3d = jnp.repeat(jnp.expand_dims(grid_x, axis = 0),n_points, axis=0)
        grid_y_3d = jnp.repeat(jnp.expand_dims(grid_y, axis = 0),n_points, axis=0)

        curr_x = P_x[:,None,None] + grid_x_3d
        curr_y = P_y[:,None,None] + grid_y_3d

        is_valid = (curr_x >= 0) & (curr_x < self.resolution) & \
                (curr_y >= 0) & (curr_y < self.resolution)

        safe_x = jnp.clip(curr_x, 0, self.resolution - 1)
        safe_y = jnp.clip(curr_y, 0, self.resolution - 1)
        #after here
        ref_points = resized_map[0, safe_x, safe_y, :]
        detached_points = jax.lax.stop_gradient(ref_points)
        target_points = self.bilinear_interpolate(resized_map, safe_x, safe_y, d_x, d_y)
        # target_points = jnp.array(target_points, dtype = jnp.float32)
        losses = jnp.linalg.norm(detached_points - target_points, ord=1, axis=-1)
        losses = jnp.where(is_valid, losses, 0.0)

        return jnp.sum(losses)

    @nnx.jit
    def point_tracking(self, new_feature_map, P, old_points, offsets, r2):

        resized_new = jax.image.resize(
            new_feature_map,
            shape=(new_feature_map.shape[0], self.resolution, self.resolution, new_feature_map.shape[-1]),
            method="bilinear"
        )
        n_points = P.shape[0]
        P_x, P_y = P[:,0], P[:,1]
        
        grid_x, grid_y = jnp.meshgrid(offsets, offsets, indexing='ij')
        grid_x_3d = jnp.repeat(jnp.expand_dims(grid_x, axis = 0),n_points, axis=0)
        grid_y_3d = jnp.repeat(jnp.expand_dims(grid_y, axis = 0),n_points, axis=0)

        curr_x = P_x[:,None,None] + grid_x_3d
        curr_y = P_y[:,None,None] + grid_y_3d
        is_valid = (curr_x >= 0) & (curr_x < self.resolution) & \
                (curr_y >= 0) & (curr_y < self.resolution)

        safe_x = jnp.clip(curr_x, 0, self.resolution - 1)
        safe_y = jnp.clip(curr_y, 0, self.resolution - 1)
        patch_features = resized_new[0, safe_x, safe_y, :]
        distances = jnp.linalg.norm(patch_features - old_points[:, None, None, :], ord=1, axis=-1)
        distances = jnp.where(is_valid, distances, jnp.inf)
        min_flat_idx = distances.reshape(n_points, -1).argmin(axis=1)
        rows, cols = jnp.unravel_index(min_flat_idx, (2*r2+1, 2*r2+1))

        return P + jnp.array((rows - r2,cols - r2)).transpose()


    @nnx.jit
    def distance_less_than_threshold(self, P, T, threshold):
        return jnp.all((jnp.abs(P[:,0] - T[:,0]) + jnp.abs(P[:,1] - T[:,1])) < threshold)


    def entry(self, truncation_psi = 0.7, code_handles_in = None, mask = None, rng = None, save_image_every_k_it: int = 50):
        image, w_code = self.generate_image(truncation_psi = truncation_psi, rng = rng)
        Image.fromarray(image).save("Original_image.jpg")
        original_feature_map = self.synthesis_network(w_code, cutoff = self.cutoff_block)
        self.resized_original_feature_map = jax.image.resize(original_feature_map, shape=(original_feature_map.shape[0], self.resolution, self.resolution, original_feature_map.shape[-1]), method="bilinear")
        if code_handles_in:
            pairs = code_handles_in
        else:
            pairs, mask = utils.get_drag_points(image)
            with open(r"mask.pkl",'wb') as f:
                pkl.dump(mask, f)
            print("Mask saved")
        
        if mask is None:
            mask = jnp.zeros_like(self.resized_original_feature_map[0], dtype=jnp.float32)
        else:
            mask = 1 - mask[:,:,None]
        
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(w_code)
        ctr = 0
        P = jnp.array([pair[0] for pair in pairs], dtype=jnp.int32)
        T = jnp.array([pair[1] for pair in pairs], dtype=jnp.int32)
        old_points = self.resized_original_feature_map[0,P[:,0],P[:,1],:]
        r2 = (12 * self.resolution) // 512
        r1 = (3*self.resolution)//512
        r2_offsets = jnp.arange(-r2, r2 + 1)
        r1_offsets = jnp.arange(-r1, r1 + 1)
        while not self.distance_less_than_threshold(P, T, 2):
            loss, new_w, opt_state= self.optimise_dlatent_single_it(optimizer, opt_state, w_code, P, T, mask, r1_offsets)
            new_feature_map = self.synthesis_network(new_w, cutoff=self.cutoff_block)
            
            new_P = self.point_tracking(new_feature_map, P, old_points, r2_offsets, r2)
            w_code = new_w
            point_str = P.tolist().__str__() + '\n' + new_P.tolist().__str__() + '\n' + T.tolist().__str__() + '\n\n'
            print(f"{ctr}.  Loss:{loss} \n{point_str}")
            P = new_P
            ctr+=1  
            if ctr%save_image_every_k_it == 0:
                new_image = self.synthesis_network(w_code)[0]
                new_image = (new_image - jnp.min(new_image)) / (jnp.max(new_image) - jnp.min(new_image))
                Image.fromarray(np.uint8(new_image * 255)).save(f'image_modified_{ctr}.jpg')
                

        new_image = self.synthesis_network(w_code)[0]
        new_image = (new_image - jnp.min(new_image)) / (jnp.max(new_image) - jnp.min(new_image))
        Image.fromarray(np.uint8(new_image * 255)).save('image_modified.jpg')

if __name__ == '__main__':
    model = DragGan(pretrained_dataset = 'ffhq')
    model.entry(rng=jrndm.PRNGKey(971))
    
    
 