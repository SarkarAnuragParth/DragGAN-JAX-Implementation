import jax.random as jrndm
from jax.numpy import max as jmax, min as jmin, mean as jmean, median as jmedian
from flax.nnx import eval_shape, split, merge
from PIL import Image 
from numpy import uint8
from src.build_nnx_generator import generator_
import orbax.checkpoint as ocp

print("Finished importing files")


rng = jrndm.PRNGKey(42)
batch_size = 3
latent_code = jrndm.normal(rng, (batch_size, 512))
generated_images, feature_maps = generator_(latent_code)

for j in range(batch_size):
    for i,fm in enumerate(feature_maps):
        print(jmax(fm[j]), jmin(fm[j]))
        grayscaled_feature_map = jmin(fm[j], axis = -1)
        normalized_grfm = (grayscaled_feature_map - jmin(grayscaled_feature_map))/(jmax(grayscaled_feature_map) - jmin(grayscaled_feature_map))
        Image.fromarray(uint8(normalized_grfm * 255), mode='L').save(f'Min_image_{j+1}__feature_map_{i+1}.jpg')