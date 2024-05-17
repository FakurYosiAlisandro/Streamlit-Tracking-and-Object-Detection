import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# Buat direktori jika belum ada
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)

# CSS untuk memposisikan judul di tengah
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Menampilkan judul di tengah
st.markdown("<h1 class='title'>YOLOv8n Tracking and Object Detection</h1>", unsafe_allow_html=True)

# Fungsi untuk menampilkan gambar
def display_image(image_file):
    image = Image.open(image_file)
    st.image(image, caption='Gambar yang diunggah.', width=500)
    return image

# Fungsi untuk menampilkan video
def display_video(video_path):
    st.video(video_path)
    return video_path

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
        st.image(result_img, caption='Gambar dengan deteksi objek', width=500)

# Fungsi untuk mendeteksi objek pada video
def detect_objects_video(model, video_file):
    video_path = save_file(video_file, "videos")
    results = model.track(source=video_path, save=True)  # pastikan save=True untuk menyimpan hasil deteksi
    
    # Mengambil path dari direktori hasil deteksi
    output_dir = results[0].save_dir  # Mengambil direktori penyimpanan hasil
    return output_dir

# Fungsi untuk mengkonversi video dari .avi ke .mp4
def convert_video_to_mp4(input_path, output_path):
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264')
        return output_path
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengkonversi video: {e}")
        return None

# Sidebar untuk mengunggah file dan model
st.sidebar.header("Unggah File")
image_file = st.sidebar.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], key="image")
video_file = st.sidebar.file_uploader("Pilih file video", type=["mp4"], key="video")
model_file = st.sidebar.file_uploader("Pilih file model", type=["pt"], key="model")

# Menata layout menggunakan kolom
col1, col2 = st.columns(2)

# Di kolom pertama
with col1:
    st.header("Info Model")
    if model_file is not None:
        model = display_model(model_file)
    else:
        st.write("Tidak ada model yang diunggah.")

    if image_file is not None:
        st.header("Gambar yang Diunggah")
        image = display_image(image_file)

    if video_file is not None:
        st.header("Video yang Diunggah")
        video_path = display_video(video_file)

# Di kolom kedua
with col2:
    st.header("Hasil Deteksi")
    if st.sidebar.button("Execute"):
        if model_file is not None and model is not None:
            if image_file is not None:
                st.write("Hasil Deteksi Gambar:")
                detect_objects_image(model, image_file)

            if video_file is not None:
                st.write("Hasil Deteksi Video:")
                output_dir = detect_objects_video(model, video_file)
                if output_dir:
                    # Mengambil file .avi dari direktori yang benar
                    avi_files = [f for f in os.listdir(output_dir) if f.endswith('.avi')]
                    if avi_files:
                        avi_file_path = os.path.join(output_dir, avi_files[0])
                        output_mp4_path = os.path.join("output_videos", "output.mp4")
                        mp4_path = convert_video_to_mp4(avi_file_path, output_mp4_path)
                        if mp4_path:
                            st.video(mp4_path)