"""
pytest test suite for DragGAN loss/point-tracking functions.

Run with:  pytest test/test_drag_gan.py -v
"""
import os
import jax
import jax.numpy as jnp
import pytest
from PIL import Image
from flax import nnx

from src.drag_gan import DragGan


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def basic_model():
    """Loaded once per test module -- weight loading is expensive."""
    return DragGan(pretrained_dataset='afhqdog')


def make_affine_feature_map(size, b_coeffs, g_coeffs):
    """
    F(x, y, c) = b_coeffs[c]*x + g_coeffs[c]*y

    Affine in (x, y) => bilinear interpolation reproduces it exactly at
    ANY query point (integer or fractional), which is what lets us
    compute the expected motion-supervision loss analytically.
    """
    C = len(b_coeffs)
    xs = jnp.arange(size, dtype=jnp.float32)
    ys = jnp.arange(size, dtype=jnp.float32)
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')
    channels = [b_coeffs[c] * X + g_coeffs[c] * Y for c in range(C)]
    fmap = jnp.stack(channels, axis=-1)
    return fmap[None, ...]


# --------------------------------------------------------------------------
# Rule 1 - motion_supervision_loss correctness
# --------------------------------------------------------------------------

def test_rule1_case1(basic_model):
    """d=(1,0), r1=0 -> offsets=[0]. Basic sanity check."""
    dummy_feature_map = jnp.zeros(shape=(1, 8, 8, 3))
    dummy_feature_map = dummy_feature_map.at[0, 3, 3, :].set((0.02, 0.064, 0.102))
    dummy_feature_map = dummy_feature_map.at[0, 4, 3, :].set((0.017, 0.08, 0.115))
    p = jnp.array([(3, 3)])
    t = jnp.array([(4, 3)])

    r1 = 0
    offsets = jnp.arange(-r1, r1 + 1)   # [0]

    loss = basic_model.motion_supervision_loss(dummy_feature_map, p, t, offsets)
    assert jnp.isclose(loss, 0.032)


def test_rule1_case2(basic_model):
    """d=(0.6,0.8), r1=3 -> 7x7 offset grid. n_points=2, same direction."""
    b = jnp.array([0.1, -0.2, 0.05])
    g = jnp.array([0.05, 0.1, -0.1])
    dummy_feature_map = make_affine_feature_map(16, b, g)

    p = jnp.array([(5, 5), (9, 3)])
    t = jnp.array([(8, 9), (12, 7)])
    r1 = 3
    offsets = jnp.arange(-r1, r1 + 1)

    dx, dy = 0.6, 0.8
    L_point = jnp.sum(jnp.abs(b * dx + g * dy))      # 0.10+0.04+0.05 = 0.19
    n_offsets = (2 * r1 + 1) ** 2                     # 49
    expected_loss = n_offsets * (L_point + L_point)   # 49 * 0.38 = 18.62

    loss = basic_model.motion_supervision_loss(dummy_feature_map, p, t, offsets)
    assert jnp.isclose(loss, expected_loss, atol=1e-4)
    assert jnp.isclose(loss, 18.62, atol=1e-4)


def test_rule1_case3(basic_model):
    """
    n_points=3. One point P==T, one moves along a single axis, one moves
    diagonally with mixed-sign components (dx positive, dy negative).
    """
    b = jnp.array([0.1, -0.2, 0.05])
    g = jnp.array([0.05, 0.1, -0.1])
    dummy_feature_map = make_affine_feature_map(16, b, g)

    # Point A: P == T            -> d=(0, 0)        -> contributes 0
    # Point B: T-P=(0, 3)        -> d=(0, 1)
    # Point C: T-P=(3, -4)       -> d=(0.6, -0.8)    -> dx positive, dy negative
    p = jnp.array([(6, 6), (4, 4), (8, 8)])
    t = jnp.array([(6, 6), (4, 7), (11, 4)])
    r1 = 2
    offsets = jnp.arange(-r1, r1 + 1)

    directions = [(0., 0.), (0., 1.), (0.6, -0.8)]
    L_points = jnp.array([jnp.sum(jnp.abs(b * dx + g * dy)) for dx, dy in directions])
    # L_points = [0.0, 0.25, 0.33]
    n_offsets = (2 * r1 + 1) ** 2          # 25
    expected_loss = n_offsets * jnp.sum(L_points)   # 25 * 0.58 = 14.5

    loss = basic_model.motion_supervision_loss(dummy_feature_map, p, t, offsets)
    assert jnp.isclose(loss, expected_loss, atol=1e-4)
    assert jnp.isclose(loss, 14.5, atol=1e-4)


# --------------------------------------------------------------------------
# Rule 2 - point_tracking should follow a moving point
# --------------------------------------------------------------------------

FEATURE_MAP_PATH = r"/data/feature_map.jpg"


@pytest.mark.skipif(
    not os.path.exists(FEATURE_MAP_PATH),
    reason=f"test fixture image not found at {FEATURE_MAP_PATH}",
)
def test_rule2_case1(basic_model):
    """
    Shift the (pre-resize) feature map by 2 rows / 1 column and check
    point_tracking recovers a correspondingly shifted point. Because the
    shift happens in the *original* (small) resolution, it must be scaled
    by resolution/orig_size to compare against tracked points in the
    upsampled (self.resolution) coordinate space.
    """
    example_feature_map = Image.open(FEATURE_MAP_PATH)
    example_feature_map = jnp.array(example_feature_map)
    example_feature_map = jnp.expand_dims(example_feature_map, axis=(0, -1))
    orig_h, orig_w = example_feature_map.shape[1], example_feature_map.shape[2]

    new_feature_map = jnp.zeros_like(example_feature_map)
    new_feature_map = new_feature_map.at[:, 2:, 1:, :].set(example_feature_map[:, :-2, :-1, :])

    resized_original = jax.image.resize(
        example_feature_map,
        shape=(example_feature_map.shape[0], basic_model.resolution, basic_model.resolution, example_feature_map.shape[-1]),
        method="bilinear",
    )

    P = jnp.array([(21, 92), (200, 300)], dtype=jnp.int32)
    old_points = resized_original[0, P[:, 0], P[:, 1], :]

    r2 = (12 * basic_model.resolution) // 512
    offsets = jnp.arange(-r2, r2 + 1)

    new_points = basic_model.point_tracking(new_feature_map, P, old_points, offsets, r2)

    # Shift applied in original (small) resolution -> scale up to full resolution
    scale_x = basic_model.resolution / orig_h
    scale_y = basic_model.resolution / orig_w
    expected_shift = jnp.array([2 * scale_x, 1 * scale_y])
    expected_points = P + jnp.round(expected_shift).astype(jnp.int32)

    # Allow a couple pixels of slack for bilinear-resize/search-grid rounding
    assert jnp.all(jnp.abs(new_points - expected_points) <= 2)


# --------------------------------------------------------------------------
# Rule 3 - gradient of motion supervision loss matches
#          sign(F(qi) - F(qi+di)) * (-d(F(qi+di))/dw)
# --------------------------------------------------------------------------

def test_rule3_case1(basic_model):
    qx, qy = 56, 64
    tx, ty = 59, 68
    delta = jnp.array([tx - qx, ty - qy], dtype=jnp.float32)
    direction = delta / jnp.linalg.norm(delta)   # (0.6, 0.8)
    dx, dy = direction[0], direction[1]

    z_code = jax.random.normal(jax.random.PRNGKey(42), (1, 512))
    w_code = basic_model.mapping_network(z_code)

    def get_feature_map_point(model, w_code, qx, qy, dx, dy):
        feature_map = model.synthesis_network(w_code, cutoff=model.cutoff_block)
        resized_feature_map = jax.image.resize(
            feature_map,
            shape=(feature_map.shape[0], model.resolution, model.resolution, feature_map.shape[-1]),
            method="bilinear",
        )
        
        d_point = model.bilinear_interpolate(
            resized_feature_map, qx, qy, jnp.array([dx]), jnp.array([dy])
        )[0, 0, 0, :]
        q_point = resized_feature_map[0, qx, qy, :]
        return d_point, q_point

    # forward pass only, to get the sign term -- no grad needed here
    d_point0, q_point0 = get_feature_map_point(basic_model, w_code, qx, qy, dx, dy)
    sign_vec = jax.lax.stop_gradient(jnp.sign(q_point0 - d_point0))

    def scalarized(model, w_code, qx, qy, dx, dy):
        d_point, _ = get_feature_map_point(model, w_code, qx, qy, dx, dy)
        # sum_c sign_c * d_point_c  =>  grad wrt w is sign_vec . d(d_point)/dw
        return jnp.sum(sign_vec * d_point)


    contracted_grad = nnx.grad(scalarized, argnums=1)(basic_model, w_code, qx, qy, dx, dy)
    expected_grads = -contracted_grad

    P = jnp.array([(qx, qy)])
    T = jnp.array([(tx, ty)])
    mask = jnp.zeros(shape=(1, basic_model.resolution, basic_model.resolution, 1))

    feature_map = basic_model.synthesis_network(w_code, cutoff=basic_model.cutoff_block)
    basic_model.resized_original_feature_map = jax.image.resize(
        feature_map,
        shape=(feature_map.shape[0], basic_model.resolution, basic_model.resolution, feature_map.shape[-1]),
        method="bilinear",
    )

    r1 = 0
    offsets = jnp.arange(-r1, r1 + 1)   # [0]

    grads = nnx.grad(DragGan.get_dlatent_loss, argnums=1)(
        basic_model, w_code, P, T, mask, offsets
    )

    assert jnp.allclose(grads, expected_grads, atol=1e-4)