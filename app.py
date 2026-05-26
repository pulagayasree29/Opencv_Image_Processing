import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Image Processing using Streamlit", layout="wide")
st.title("Image Processing using Streamlit")


# -----------------------------
# Helper functions
# -----------------------------
def list_images(folder: str):
    path = Path(folder)
    if not path.exists() or not path.is_dir():
        return []
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p.name for p in path.iterdir() if p.suffix.lower() in allowed_ext])


def to_rgb(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_download_button(img, filename: str):
    if img is None:
        return

    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    _, buffer = cv2.imencode(".png", out)
    st.download_button(
        label=f"Download {filename}",
        data=buffer.tobytes(),
        file_name=filename,
        mime="image/png",
    )


def draw_shapes(img_bgr):
    out = img_bgr.copy()
    cv2.line(out, (0, 0), (150, 150), (0, 255, 0), 15)
    cv2.rectangle(out, (15, 25), (200, 150), (0, 0, 255), 5)
    cv2.circle(out, (150, 80), 55, (255, 0, 0), -1)
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    cv2.polylines(out, [pts], True, (255, 0, 255), 5)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(out, "Hello World", (50, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)
    return out


def pixel_roi_demo(img_bgr):
    out = img_bgr.copy()
    pixel_value = None
    watch_face = None

    if out.shape[0] > 55 and out.shape[1] > 55:
        pixel_value = out[55, 55]

    if out.shape[0] > 150 and out.shape[1] > 150:
        out[100:150, 100:150] = [255, 255, 255]

    if out.shape[0] > 111 and out.shape[1] > 194:
        watch_face = out[37:111, 107:194].copy()
        out[0:74, 0:87] = watch_face

    return out, pixel_value, watch_face


def logo_overlay_demo(bg_bgr, logo_bgr):
    if bg_bgr is None or logo_bgr is None:
        return None, None, None, None, None

    rows, cols, channels = logo_bgr.shape
    if bg_bgr.shape[0] < rows or bg_bgr.shape[1] < cols:
        return None, None, None, None, None

    roi = bg_bgr[0:rows, 0:cols]
    logo_gray = cv2.cvtColor(logo_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(logo_gray, 220, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
    logo_part = cv2.bitwise_and(logo_bgr, logo_bgr, mask=mask)
    dst = cv2.add(bg_part, logo_part)

    result = bg_bgr.copy()
    result[0:rows, 0:cols] = dst
    return mask, mask_inv, bg_part, logo_part, result


def threshold_demo(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, threshold_color = cv2.threshold(img_bgr, 12, 255, cv2.THRESH_BINARY)
    _, threshold_gray = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    gaus = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1
    )
    _, otsu = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, threshold_color, threshold_gray, gaus, otsu


def grabcut_demo(img_bgr):
    if img_bgr is None:
        return None, None

    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgmodel = np.zeros((1, 65), np.float64)
    fgmodel = np.zeros((1, 65), np.float64)

    rect_w = min(300, img_bgr.shape[1] - 50)
    rect_h = min(500, img_bgr.shape[0] - 50)
    rect_w = max(1, rect_w)
    rect_h = max(1, rect_h)
    rect = (50, 50, rect_w, rect_h)

    cv2.grabCut(img_bgr, mask, rect, bgmodel, fgmodel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = img_bgr * mask2[:, :, np.newaxis]
    return mask2, result


def morph_demo(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary, erosion, dilation, opening, closing


# -----------------------------
# Load image from folder
# -----------------------------
img_folder = "images"

if not os.path.exists(img_folder):
    st.error(f"Please create a folder named '{img_folder}' and put an image inside.")
    st.stop()

img_list = list_images(img_folder)

if not img_list:
    st.warning("No images found in the folder.")
    st.stop()

selected_file = st.selectbox("Select an image", img_list)
img_path = os.path.join(img_folder, selected_file)

# Load and Resize to 1080x720
image = Image.open(img_path).convert("RGB")
resized_img = image.resize((1080, 720))

st.subheader("Main Window - Resized Image (1080×720)")
st.image(resized_img, use_container_width=True)

# Converting PIL image to OpenCV format for processing
img_array = np.array(resized_img)
img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# 2. Parameters & Buttons
st.sidebar.header("Parameters")
st.sidebar.caption("Use the buttons below to switch the main output.")

col1, col2, col3 = st.columns(3)

if "output_mode" not in st.session_state:
    st.session_state.output_mode = "original"

# GRAY SCALE
if col1.button("Gray"):
    st.session_state.output_mode = "gray"

# BLUR
blur_val = st.sidebar.slider("Blur Intensity (must be odd)", 1, 99, 5, step=2)
if col2.button("Blur"):
    st.session_state.output_mode = "blur"

# EDGE DETECTION
t1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
t2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
if col3.button("Edge"):
    st.session_state.output_mode = "edge"

# Show selected output
st.subheader("Output Window")
output_img = img_cv.copy()
output_caption = "Original image"

if st.session_state.output_mode == "gray":
    output_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    output_caption = "Gray-scaled Image"
    st.image(output_img, caption=output_caption, use_container_width=True)
    st.info("Converts the image into grayscale.")
    save_download_button(output_img, "gray_image.png")

elif st.session_state.output_mode == "blur":
    output_img = cv2.GaussianBlur(img_cv, (blur_val, blur_val), 0)
    output_caption = "Blurred Image"
    st.image(to_rgb(output_img), caption=output_caption, use_container_width=True)
    st.info("Applies Gaussian blur to smooth the image.")
    save_download_button(output_img, "blur_image.png")

elif st.session_state.output_mode == "edge":
    gray_for_edge = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    output_img = cv2.Canny(gray_for_edge, t1, t2)
    output_caption = "Edge Detected Image"
    st.image(output_img, caption=output_caption, use_container_width=True)
    st.info("Detects strong edges in the image.")
    save_download_button(output_img, "edge_image.png")

else:
    st.image(to_rgb(output_img), caption=output_caption, use_container_width=True)

# Processing time example
st.caption(f"Loaded and displayed in {0.00:.2f} seconds")

# -----------------------------
# Extra OpenCV topics
# -----------------------------
st.divider()
st.subheader("Additional OpenCV Topics")

# 1) Draw Line, Rectangle, Circle, Polygon, Text
with st.expander("Draw Line, Rectangle, Circle, Polygon, and Text"):
    st.info("Draws geometric shapes and text on the image using OpenCV drawing functions.")
    shape_bgr = draw_shapes(img_cv)

    c1, c2 = st.columns(2)
    with c1:
        st.image(to_rgb(img_cv), caption="Original", use_container_width=True)
    with c2:
        st.image(to_rgb(shape_bgr), caption="Image with shapes and text", use_container_width=True)

    save_download_button(shape_bgr, "shapes.png")

# 2) Pixel access, ROI and copy-paste
with st.expander("Pixel Access, ROI, and Copy-Paste"):
    st.info("Reads a pixel value, selects a region of interest, and copies a watch-face region to another place.")
    roi_img, px, watch_face = pixel_roi_demo(img_cv)

    c1, c2 = st.columns(2)
    with c1:
        st.image(to_rgb(img_cv), caption="Original", use_container_width=True)
    with c2:
        st.image(to_rgb(roi_img), caption="ROI modified image", use_container_width=True)

    if px is not None:
        st.write(f"Pixel value at (55, 55): `{px}`")
    else:
        st.write("Image is too small to read pixel (55, 55).")

    if watch_face is not None:
        st.write("A small region was copied and pasted to the top-left corner.")

    save_download_button(roi_img, "roi_copy.png")

# 3) Logo overlay with masking
with st.expander("Logo Overlay with Masking"):
    st.info("Uses masking and bitwise operations to overlay a logo onto another image.")

    if len(img_list) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            bg_file = st.selectbox("Background image", img_list, key="bg_file")
        with col_b:
            logo_file = st.selectbox("Logo image", img_list, key="logo_file")

        bg_img = Image.open(os.path.join(img_folder, bg_file)).convert("RGB")
        logo_img = Image.open(os.path.join(img_folder, logo_file)).convert("RGB")

        bg_arr = cv2.cvtColor(np.array(bg_img.resize((1080, 720))), cv2.COLOR_RGB2BGR)
        logo_arr = cv2.cvtColor(np.array(logo_img), cv2.COLOR_RGB2BGR)

        mask, mask_inv, bg_part, logo_part, result = logo_overlay_demo(bg_arr, logo_arr)

        if result is None:
            st.warning("Background image must be larger than the logo image.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(mask, caption="Mask", use_container_width=True)
            with c2:
                st.image(mask_inv, caption="Inverse mask", use_container_width=True)
            with c3:
                st.image(to_rgb(result), caption="Merged Result", use_container_width=True)

            c4, c5 = st.columns(2)
            with c4:
                st.image(to_rgb(bg_part), caption="Background Part", use_container_width=True)
            with c5:
                st.image(to_rgb(logo_part), caption="Logo Foreground", use_container_width=True)

            save_download_button(result, "logo_overlay.png")
    else:
        st.warning("Need at least two images in the folder for the logo overlay demo.")

# 4) Thresholding
with st.expander("Thresholding"):
    st.info("Converts images into binary form using simple, adaptive, and Otsu thresholding.")
    gray_img, threshold_color, threshold_gray, gaus, otsu = threshold_demo(img_cv)

    c1, c2 = st.columns(2)
    with c1:
        st.image(to_rgb(img_cv), caption="Original", use_container_width=True)
        st.image(threshold_color, caption="Simple Threshold", use_container_width=True)
        st.image(gaus, caption="Adaptive Gaussian Threshold", use_container_width=True)

    with c2:
        st.image(gray_img, caption="Grayscale", use_container_width=True)
        st.image(threshold_gray, caption="Threshold on Grayscale", use_container_width=True)
        st.image(otsu, caption="Otsu Threshold", use_container_width=True)

    save_download_button(otsu, "otsu_threshold.png")

# 5) GrabCut foreground extraction
with st.expander("GrabCut Foreground Extraction"):
    st.info("Extracts the foreground object from the background using the GrabCut algorithm.")
    mask2, result = grabcut_demo(img_cv.copy())

    if result is None:
        st.error("Could not run GrabCut on this image.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(to_rgb(img_cv), caption="Original", use_container_width=True)
        with c2:
            st.image(to_rgb(result), caption="Foreground Extracted", use_container_width=True)

        st.image(mask2 * 255, caption="Binary Mask", use_container_width=True)
        save_download_button(result, "grabcut_result.png")

# 6) Morphological operations
with st.expander("Morphological Operations"):
    st.info("Shows erosion, dilation, opening, and closing on a binary image.")
    binary, erosion, dilation, opening, closing = morph_demo(img_cv)

    c1, c2 = st.columns(2)
    with c1:
        st.image(binary, caption="Binary Image", use_container_width=True)
        st.image(erosion, caption="Erosion", use_container_width=True)
    with c2:
        st.image(dilation, caption="Dilation", use_container_width=True)
        st.image(opening, caption="Opening", use_container_width=True)
    st.image(closing, caption="Closing", use_container_width=True)

    save_download_button(closing, "morph_close.png")

st.sidebar.markdown("---")
st.sidebar.caption("Put your image files inside the folder you selected.")
st.sidebar.caption("Supported files: JPG, JPEG, PNG, BMP, TIFF, WEBP")
# GITHUB ICON
st.sidebar.markdown("---")

github_html = """
<div style="text-align:center;">
    <a href="https://github.com/pulagayasree29/Opencv_Image_Processing" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="45">
    </a>
    <p style="font-size:14px;">
        My GitHub
    </p>
</div>
"""

st.sidebar.markdown(github_html, unsafe_allow_html=True)
