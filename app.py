import streamlit as st
from PIL import Image
import tempfile
import torch
import os
from ultralytics import YOLO

# Buat direktori jika belum ada
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

st.title('Upload dan Tampilkan Gambar, Video, dan Model')

# Fungsi untuk menampilkan gambar
def display_image(image_file):
    image = Image.open(image_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    return image

# Fungsi untuk menampilkan video
def display_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.video(tfile.name)
    return tfile.name

# Fungsi untuk menampilkan model
def display_model(model_file):
    try:
        model_path = os.path.join("models", model_file.name)
        with open(model_path, 'wb') as f:
            f.write(model_file.getbuffer())
        model = YOLO(model_path)
        st.write("Model berhasil diunggah.")
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Fungsi untuk menyimpan file
def save_file(file, dir_name):
    file_path = os.path.join(dir_name, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

# Fungsi untuk mendeteksi objek pada gambar
def detect_objects_image(model, image_file):
    image_path = save_file(image_file, "images")
    results = model.predict(source=image_path)
    for result in results:
        result_img = result.plot()
        st.image(result_img, caption='Gambar dengan deteksi objek', use_column_width=True)

# Fungsi untuk mendeteksi objek pada video
def detect_objects_video(model, video_file):
    video_path = save_file(video_file, "videos")
    results = model.track(source=video_path, show=True, stream=True, save=True)
    output_video_path = None
    for result in results:
        output_video_path = result.path
    if output_video_path:
        result_video_path = os.path.join("results", os.path.basename(output_video_path))
        os.rename(output_video_path, result_video_path)  # Move the output video to the results folder
        st.video(result_video_path)  # Display the video

# Membuat kolom
col1, col2, col3 = st.columns(3)

# Kolom untuk mengunggah gambar dan video
with col1:
    st.header("Unggah Gambar")
    image_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], key="image")

    st.header("Unggah Video")
    video_file = st.file_uploader("Pilih file video", type=["mp4", "mov", "avi"], key="video")

# Kolom untuk menampilkan gambar dan video
with col2:
    if image_file is not None:
        image = display_image(image_file)

    if video_file is not None:
        video_path = display_video(video_file)

# Kolom untuk mengunggah model
with col3:
    st.header("Unggah Model")
    model_file = st.file_uploader("Pilih file model", type=["pt"], key="model")

    if model_file is not None:
        model = display_model(model_file)
 
# Tombol untuk memproses deteksi objek
if st.button("Execute"):
    if model_file is not None and model is not None:
        if image_file is not None:
            detect_objects_image(model, image_file)

        if video_file is not None:
            detect_objects_video(model, video_file)