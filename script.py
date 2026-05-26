import jax.random as jrndm
from jax.numpy import max as jmax, min as jmin
from flax.nnx import eval_shape, split, merge
from PIL import Image
from numpy import uint8
from src.build_nnx_generator import StyleGAN_Generator
import orbax.checkpoint as ocp

print("Finished importing files")

checkpointer = ocp.PyTreeCheckpointer()
abstract_model = eval_shape(lambda: StyleGAN_Generator())
graphdef, abstract_state = split(abstract_model)
state_restored = checkpointer.restore(r'C:/Users/anura/DragGAN-JAX/models/ffhq_styleGAN', abstract_state)
nnx_generator = merge(graphdef, state_restored)

rng = jrndm.random.PRNGKey(0)
latent_code = jrndm.random.normal(rng, (3, 512))

generated_images = nnx_generator(latent_code)

images = (generated_images - jmin(generated_images)) / (jmax(generated_images) - jmin(generated_images))

for i in range(images.shape[0]):
    Image.fromarray(uint8(images[i] * 255)).save(f'image_{i}.jpg')