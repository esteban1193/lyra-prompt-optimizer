import streamlit as st
from openai import OpenAI
import pandas as pd

st.set_page_config(
    page_title="Lyra – AI Prompt Optimizer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Theme toggle
theme = st.sidebar.radio("🌓 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)

st.title("🧠 Lyra – AI Prompt Optimizer")

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

if "latest_optimized_prompt" not in st.session_state:
    st.session_state.latest_optimized_prompt = ""

st.sidebar.header("🛠️ Settings")
primary_model = st.sidebar.selectbox("Primary Model", ["gpt-4", "gpt-3.5-turbo", "gpt-4o"])
compare_model = st.sidebar.selectbox("Compare With (optional)", ["None", "gpt-4", "gpt-3.5-turbo", "gpt-4o"])
target_ai = st.sidebar.selectbox("Target AI", ["ChatGPT", "Claude", "Gemini", "Other"])
mode = st.sidebar.radio("Optimization Mode", ["DETAIL", "BASIC"])

st.markdown("### ✍️ Enter your rough prompt")
user_prompt = st.text_area("Prompt", height=200)

def estimate_cost(input_tokens, output_tokens, model):
    prices = {
        "gpt-4": (0.03, 0.06),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "gpt-4o": (0.005, 0.015)
    }
    input_price, output_price = prices.get(model, prices["gpt-4o"])
    return round(input_tokens * input_price / 1000 + output_tokens * output_price / 1000, 4)

client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if st.button("🚀 Optimize Prompt") and client:
    if not user_prompt:
        st.warning("Please enter a prompt.")
    else:
        system_prompt = f"""
You are Lyra, a master-level AI prompt optimization specialist. Your mission: transform any user input into precision-crafted prompts that unlock AI's full potential across all platforms.

Follow the 4-D Methodology: DECONSTRUCT → DIAGNOSE → DEVELOP → DELIVER.
Use advanced optimization techniques: role assignment, context layering, constraint-based design, chain-of-thought, etc.
Always tailor prompts to the selected AI platform and user intent.
"""

        full_prompt = f"""
Target AI: {target_ai}
Mode: {mode}
Rough Prompt: {user_prompt}
"""

        try:
            response = client.chat.completions.create(
                model=primary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            optimized_output = response.choices[0].message.content
            st.success("✅ Optimized Prompt:")
            st.session_state.latest_optimized_prompt = optimized_output
            st.session_state.prompt_history.insert(0, {
                "input": user_prompt,
                "output": optimized_output,
                "model": primary_model
            })
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.latest_optimized_prompt:
    st.markdown("### ✏️ Review and Edit the Optimized Prompt")
    editable_prompt = st.text_area("Edit Optimized Prompt Below:", value=st.session_state.latest_optimized_prompt, height=200)

    if st.button("▶ Try Optimized Prompt") and client:
        def run_model(model_name):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": editable_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content

        try:
            response_main = run_model(primary_model)
            st.markdown("### 🤖 Response from Primary Model")
            st.markdown(response_main)
            tokens_in = len(editable_prompt.split())
            tokens_out = len(response_main.split())
            st.markdown(f"💰 **Estimated Cost:** ${estimate_cost(tokens_in, tokens_out, primary_model)}")

            if compare_model != "None" and compare_model != primary_model:
                response_compare = run_model(compare_model)
                st.markdown("---")
                st.markdown("### 🤖 Response from Comparison Model")
                st.markdown(response_compare)
                tokens_out_compare = len(response_compare.split())
                st.markdown(f"💰 **Estimated Cost:** ${estimate_cost(tokens_in, tokens_out_compare, compare_model)}")

        except Exception as e:
            st.error(f"Error running optimized prompt: {e}")

st.markdown("### 📜 Prompt History")
if st.session_state.prompt_history:
    for i, item in enumerate(st.session_state.prompt_history[:5]):
        with st.expander(f"History #{i+1}"):
            st.markdown("**Input:**")
            st.code(item['input'], language='markdown')
            st.markdown("**Optimized Prompt:**")
            st.markdown(item['output'])

    if st.download_button("📥 Download History as CSV", pd.DataFrame(st.session_state.prompt_history).to_csv(index=False), file_name="lyra_prompt_history.csv"):
        st.success("Download started.")

else:
    st.info("No prompt history in this session.")
