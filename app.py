import streamlit as st
from app_logic import text2image


from io import BytesIO


def app():
    st.header("Text-to-image Web App")
    st.subheader("Powered by Hugging Face")
    user_input = st.text_area(
        "Enter your text prompt below and click the button to submit."
    )

    option = st.selectbox(
        "Select model (in order of processing time)",
        (
        "mukaist/DALLE-4K",    
        "prithivMLmods/Canopus-Realism-LoRA",
        "black-forest-labs/FLUX.1-dev",
        "SG161222/RealVisXL_V4.0_Lightning",
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "SG161222/RealVisXL_V3.0",
        "CompVis/stable-diffusion-v1-4",
        ),
    )

    with st.form("my_form"):
        submit = st.form_submit_button(label="Submit text prompt")

    if submit:
        with st.spinner(text="Generating image ... It may take up to some time."):
            im, start, end = text2image(prompt=user_input, repo_id=option)

            buf = BytesIO()
            im.save(buf, format="PNG")
            byte_im = buf.getvalue()

            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)

            st.success(
                "Processing time: {:0>2}:{:0>2}:{:05.2f}.".format(
                    int(hours), int(minutes), seconds
                )
            )

            st.image(im)

            st.download_button(
                label="Click here to download",
                data=byte_im,
                file_name="generated_image.png",
                mime="image/png",
            )


if __name__ == "__main__":
    app()