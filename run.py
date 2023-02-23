# 实现web界面
# 2023年2月23日 功能：选择文件、选择转换风格


from PIL import Image
import streamlit as st
from io import BytesIO
from detect import opt, detect
from util import util
"""
# CycleGAN
"""


# 单文件载入
def choose_img_file(label: str, frame=st):
    uploaded_file = frame.file_uploader(label=label)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(BytesIO(bytes_data))
        st.image(img, caption='', use_column_width=True)
        opt.dataroot = img
        fake_img_pil = detect()
        return fake_img_pil


# @st.cache(show_spinner=False, suppress_st_warning=True)
def setting_params_ui(frame=st.sidebar):
    frame.markdown("# 选择风格")
    style_options = util.get_all_weights()
    style = frame.selectbox(label='# 选择风格', options=style_options)

    frame.markdown("# 调整参数")
    # confidence_threshold = frame.slider("C", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = frame.slider("O", 0.0, 1.0, 0.3, 0.01)

    return style, overlap_threshold


setting_params_ui()
img = choose_img_file("选择一张本地图片")
if img is not None:
    st.image(img, caption='', use_column_width=True)

# stringio = bytes_data.decode("utf-8")
# st.write(stringio)

# To convert to a string based IO:
# stringio = StringIO(bytes_data.decode("utf-8"))
# st.write(stringio)

# # To read file as string:
# string_data = stringio.read()
# st.write(string_data)

# # Can be used wherever a "file-like" object is accepted:
# st.write(uploaded_file)
# dataframe = pd.read_csv(uploaded_file)
# st.write(dataframe)

# 多文件载入
# uploaded_files = st.file_uploader(
#     "Choose a CSV file", accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     st.write(bytes_data)
