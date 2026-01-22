import streamlit as st
from main import run_pipeline

st.set_page_config(page_title="LLGC Runner", layout="wide")
st.title("🚀 LLGC Pipeline Runner")

uploaded_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])

if uploaded_file:
    dataset_path = "uploaded_dataset.csv"
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Dataset uploaded")

log_box = st.empty()
logs = []

def ui_logger(message):
    logs.append(message)
    log_box.code("\n".join(logs))

if st.button("Run Pipeline"):
    if not uploaded_file:
        st.error("Upload a dataset first.")
    else:
        with st.spinner("Running pipeline..."):
            result = run_pipeline(dataset_path, progress_cb=ui_logger)

            st.subheader("📊 Final Metrics")
            st.json(result["metrics"])

            st.subheader("🖼️ Visualizations")
            for fig in result["figures"]:
                st.image(fig, width=600)

        st.success("✅ Pipeline completed")
