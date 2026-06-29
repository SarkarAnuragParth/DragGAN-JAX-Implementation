import cv2
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import math
from flaxmodels.stylegan2.generator import URLS
from flaxmodels.stylegan2.ops import conv2d
from flaxmodels.utils import download
import os

from h5py import File

def get_file(file_path:str):
    file_name = os.path.basename(file_path)
    dataset = file_name.removeprefix('stylegan2_generator_').removesuffix('.h5')
    return File(download(os.path.dirname(file_path), URLS[dataset]), 'r')

def equalize_lr_weight(w, lr_multiplier=1.0):
    """
    Applies Equalized Learning Rate scaling to a weight tensor.
    JAX-safe: returns a newly scaled array instead of mutating in-place.
    """
    fan_in = jnp.prod(jnp.array(w.shape[:-1]))
    gain = lr_multiplier / jnp.sqrt(fan_in)
    return w * gain

def equalize_lr_bias(b, lr_multiplier=1.0):
    """
    Applies the learning rate multiplier to the bias.
    """
    return b * lr_multiplier


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




def get_drag_points(image_input, brush_size=15):
    """
    Displays an interactive image editor that lets the user:
      1. Select multiple handle/target point pairs (drag-style edits), AND
      2. Paint an optional binary mask, with a live brush-size preview.

    GUI:
        - Trackbar "Brush Size" at the top of the window — drag with mouse.
        - On-canvas buttons: [POINT MODE] [MASK MODE] [CLEAR MASK]
        - Keyboard shortcuts: 'm' toggle mode, 'c' clear mask, 'q'/Esc finish.

    Args:
        image_input (str or numpy.ndarray): Path to image, or an already-loaded array.
        brush_size (int): Initial brush radius in pixels.

    Returns:
        tuple:
            pairs (list): [((hx, hy), (tx, ty)), ...] handle/target pairs.
            mask (numpy.ndarray): uint8 (H, W) array, 1 where painted, 0 elsewhere.
                                   If nothing was painted, returns an all-ones mask.
    """

    # ---- Load the image -----------------------------------------------
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not load image from {image_input}")
    else:
        img = image_input.copy()

    try:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except cv2.error:
        pass  # already BGR / grayscale

    h, w = img.shape[:2]

    # ---- Layout: leave a top strip for buttons --------------------------
    TOP_BAR_H = 50
    canvas_h = h + TOP_BAR_H

    BTN_POINT = (10, 10, 130, 40)   # x1, y1, x2, y2
    BTN_MASK = (140, 10, 260, 40)
    BTN_CLEAR = (270, 10, 390, 40)

    def point_in_rect(px, py, rect):
        x1, y1, x2, y2 = rect
        return x1 <= px <= x2 and y1 <= py <= y2

    # ---- State ------------------------------------------------------------
    selected_points = []
    annotations = []
    mask = np.zeros((h, w), dtype=np.uint8)   # 0 or 255

    state = {
        "mode": "point",          # "point" or "mask"
        "brush": max(1, int(brush_size)),
        "drawing_mask": False,
        "last_mask_pt": None,
        "cursor": None,            # (x, y) in IMAGE coords, or None if off-image
        "base": None,              # cached: img + mask overlay + annotations (no cursor)
    }

    WIN = "DragGAN Editor"

    # ---- Rendering ----------------------------------------------------------
    def rebuild_base():
        """Recompute the cached base layer (image + mask overlay + point annotations)."""
        body = img.copy()

        if mask.any():
            overlay = body.copy()
            overlay[mask > 0] = (0, 255, 255)  # yellow, BGR
            body = cv2.addWeighted(overlay, 0.45, body, 0.55, 0)

        for draw_fn in annotations:
            draw_fn(body)

        # Stitch top bar + body into one canvas
        full = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        full[TOP_BAR_H:, :, :] = body
        state["base"] = full

    def draw_button(canvas, rect, label, active=False):
        x1, y1, x2, y2 = rect
        color = (60, 180, 75) if active else (90, 90, 90)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def render():
        """Draw top bar (buttons) + cached base + live cursor preview, then show."""
        if state["base"] is None:
            rebuild_base()

        canvas = state["base"].copy()

        # Top bar background
        cv2.rectangle(canvas, (0, 0), (w, TOP_BAR_H), (40, 40, 40), thickness=-1)
        draw_button(canvas, BTN_POINT, "POINT MODE", active=(state["mode"] == "point"))
        draw_button(canvas, BTN_MASK, "MASK MODE", active=(state["mode"] == "mask"))
        draw_button(canvas, BTN_CLEAR, "CLEAR MASK")

        hud = f"Brush: {state['brush']}px"
        cv2.putText(canvas, hud, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Live brush-size preview circle, only in MASK mode, only over the image area
        if state["mode"] == "mask" and state["cursor"] is not None:
            cx, cy = state["cursor"]
            preview = canvas.copy()
            cv2.circle(preview, (cx, cy + TOP_BAR_H), state["brush"],
                       (255, 255, 255), thickness=-1)
            canvas = cv2.addWeighted(preview, 0.35, canvas, 0.65, 0)
            cv2.circle(canvas, (cx, cy + TOP_BAR_H), state["brush"],
                       (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow(WIN, canvas)

    # ---- Point-mode annotation helper --------------------------------------
    def add_point_annotation(x, y):
        selected_points.append((x, y))
        pair_idx = len(selected_points) // 2 + (len(selected_points) % 2)

        if len(selected_points) % 2 != 0:
            def draw_handle(c, pt=(x, y), idx=pair_idx):
                cv2.circle(c, pt, radius=5, color=(0, 0, 255), thickness=-1)
                cv2.putText(c, f"H{idx}", (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            annotations.append(draw_handle)
        else:
            handle_pt = selected_points[-2]
            target_pt = selected_points[-1]

            def draw_target(c, h_pt=handle_pt, t_pt=target_pt, idx=pair_idx):
                cv2.circle(c, t_pt, radius=5, color=(0, 255, 0), thickness=-1)
                cv2.putText(c, f"T{idx}", (t_pt[0] + 10, t_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.arrowedLine(c, h_pt, t_pt, color=(255, 0, 0), thickness=2)
            annotations.append(draw_target)

        rebuild_base()

    # ---- Mouse callback -----------------------------------------------------
    def mouse_callback(event, x, y, flags, param):
        # Coordinates are in FULL canvas space; convert to image space when below top bar
        in_top_bar = y < TOP_BAR_H
        img_x, img_y = x, y - TOP_BAR_H

        if event == cv2.EVENT_MOUSEMOVE:
            state["cursor"] = (img_x, img_y) if not in_top_bar else None
            if state["mode"] == "mask" and state["drawing_mask"] and not in_top_bar:
                last_pt = state["last_mask_pt"]
                if last_pt is not None:
                    cv2.line(mask, last_pt, (img_x, img_y), 255,
                             thickness=state["brush"] * 2)
                cv2.circle(mask, (img_x, img_y), state["brush"], 255, thickness=-1)
                state["last_mask_pt"] = (img_x, img_y)
                rebuild_base()
            render()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if in_top_bar:
                if point_in_rect(x, y, BTN_POINT):
                    state["mode"] = "point"
                elif point_in_rect(x, y, BTN_MASK):
                    state["mode"] = "mask"
                elif point_in_rect(x, y, BTN_CLEAR):
                    mask[:] = 0
                    rebuild_base()
                render()
                return

            if state["mode"] == "point":
                add_point_annotation(img_x, img_y)
            else:
                state["drawing_mask"] = True
                state["last_mask_pt"] = (img_x, img_y)
                cv2.circle(mask, (img_x, img_y), state["brush"], 255, thickness=-1)
                rebuild_base()
            render()

        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing_mask"] = False
            state["last_mask_pt"] = None

    # ---- Trackbar (brush size slider) --------------------------------------
    def on_trackbar(val):
        state["brush"] = max(1, val)
        render()

    cv2.namedWindow(WIN)
    cv2.createTrackbar("Brush Size", WIN, state["brush"], 100, on_trackbar)
    cv2.setMouseCallback(WIN, mouse_callback)

    print("GUI Opened.")
    print("  - Click [POINT MODE] / [MASK MODE] buttons, or press 'm', to switch modes.")
    print("  - Drag the 'Brush Size' slider with your mouse to resize the brush.")
    print("  - POINT mode: click for Handle, click again for its Target.")
    print("  - MASK mode : click-and-drag to paint; live circle previews brush size.")
    print("  - [CLEAR MASK] button or 'c' key clears the mask.")
    print("  - 'q' or Esc: finish and close the window.")

    rebuild_base()
    render()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            state["mode"] = "mask" if state["mode"] == "point" else "point"
            render()
        elif key == ord('c'):
            mask[:] = 0
            rebuild_base()
            render()

        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()

    # ---- Build point pairs --------------------------------------------------
    pairs = []
    for i in range(0, len(selected_points) - 1, 2):
        pairs.append((selected_points[i], selected_points[i + 1]))

    if len(selected_points) % 2 != 0:
        print("\nNotice: The last handle point had no target and was discarded.")

    if pairs:
        print(f"\nRecorded {len(pairs)} pair(s):")
        for i, pair in enumerate(pairs, 1):
            print(f"({pair[0]},{pair[1]})", end=", ")
    else:
        print("\nNo complete pairs selected.")
    print("\n")
    # ---- Build binary mask ---------------------------------------------------
    binary_mask = (mask > 0).astype(np.uint8)
    if not binary_mask.any():
        print("No mask drawn — returning a mask full of 1's.")
        binary_mask = np.ones((h, w), dtype=np.uint8)
    else:
        print(f"Mask drawn covering {int(binary_mask.sum())} pixels.")

    return pairs, binary_mask


def draw_arrows_on_image(image_path, point_pairs, output_path="output_with_arrows.jpg"):
    """
    Draws arrows on an image from point p to point t for multiple pairs.
    
    :param image_path: Path to the input image.
    :param point_pairs: List of tuples/lists containing pairs of points, e.g., [((x1, y1), (x2, y2)), ...]
    :param output_path: Path where the resulting image will be saved.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Configuration for the arrows
    # Color is in BGR format (Blue, Green, Red). (0, 0, 255) is pure Red.
    color = (0, 0, 255) 
    thickness = 3
    tip_length = 0.2  # Length of the arrow tip relative to the arrow length

    # Loop through all the provided pairs and draw arrows
    for p, t in point_pairs:
        # cv2.arrowedLine(img, start_point, end_point, color, thickness, tipLength)
        cv2.arrowedLine(image, p, t, color, thickness, tipLength=tip_length)

    # Save the modified image
    cv2.imwrite(output_path, image)
    print(f"Successfully saved the image with arrows to {output_path}")


    
if __name__ == "__main__":
    image_path = 'Original_image.jpg'
    pairs = [((269, 792),(285, 780)),((762, 781),(748, 769))]
    draw_arrows_on_image(image_path, pairs)
    