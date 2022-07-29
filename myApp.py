import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from matplotlib.patches import Rectangle


def read_image(name):
    image = st.file_uploader("Upload an "+ name, type=["png", "jpg", "jpeg"])
    if image:
        im = Image.open(image)
        im.filename = image.name
        return im

def show_image(image, mask):
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
    cnts,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _bbAndMask(image, cnts)
    _heatmap(image, cnts)

def _bbAndMask(image, cnts):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')
    _bbox(image, cnts, ax1)
    _maskOutline(image, cnts, ax2)
    st.pyplot(fig)

def _bbox(image, cnts, ax):
    ax.imshow(image)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        [x, y, w, h] = cv2.boundingRect(c)
        ax.add_patch(Rectangle((x, y), w, h, color = "red", fill = False))
def _maskOutline(image, cnts, ax):
    img = _drawMask(image, cnts, False)
    ax.imshow(img)        

def _drawMask(image, cnts, fill=True):
    image = np.array(image)
    markers = np.zeros((image.shape[0], image.shape[1]))
    heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    t = 2
    if fill:
        t = -1
    cv2.drawContours(markers, cnts, -1, (255, 0, 0), t)
    mask = markers>0
    image[mask,:] = heatmap_img[mask,:]
    return image

def _heatmap(image, cnts):
    fig2 = plt.figure()
    plt.axis('off')
    hm = st.slider("slider for heatmap", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    img = _drawMask(image, cnts)
    plt.imshow(img, alpha=hm)
    plt.imshow(image, alpha=1-hm)
    plt.title("heatmap")
    st.pyplot(fig2)

def main():
    st.set_page_config(page_title='Omdena Envisionit', page_icon=None, layout='centered', initial_sidebar_state='auto')
    st.title('Detecting Pathologies Through Computer Vision in Ultrasound')
    image = read_image('image')
    mask = read_image('mask')
    if image and mask:
        show_image(image, mask)

if __name__ == "__main__":
    main()