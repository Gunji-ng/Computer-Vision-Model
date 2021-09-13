import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

header = st.container()
img_display = st.container()
prediction = st.container()

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./extracted_product_detection_model/')
    model.summary()
    return model

def normalize(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

INPUT_SIZE = 224

about_text = """
Sometimes, consumers see (in public) items that they would love to buy but they might not know the name/model of the product. It would be great if they could just upload an image of the item an have the product identified.
This is a Demo application that allows users to upload pictures of items they are interested in and have the product identified. This would be useful for ecommerce stores as users could get results of products matching the desired item by simply uploading an image.
As this is just a demo, the model was trained on 5 products only:
* iphone12_pro_max
* jbl_charge3
* nintendo_switch
* ps4_controller
* yeezy_boost_350
"""

with header:
    st.title('Product Detection Demo')
    st.markdown(about_text)

with img_display:
    uploaded_image = st.file_uploader('Upload your image here', type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
    else:
        demo_image = './products_images/test/ps4_controller/demo_test.jpg'
        image = Image.open(demo_image)

    st.image(image)

with prediction:
    model = load_model()

    image = np.expand_dims(image, axis=0)
    image = tf.concat(image, axis=0)
    image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])

    image = normalize(image)

    labels = ['iphone12_pro_max', 'jbl_charge3', 'nintendo_switch', 'ps4_controller', 'yeezy_boost_350']
    prediction = model.predict(image)

    if prediction.max() > 0:
        predicted_label = labels[prediction.argmax()]
        st.markdown(f'_Product found in the image:_ **{predicted_label}**')
    else:
        st.markdown("_Sorry we couldn't find any matching products_")
