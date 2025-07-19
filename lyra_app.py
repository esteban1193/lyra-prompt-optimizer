import streamlit as st
from openai import OpenAI
import json
import os

st.set_page_config(
    page_title="Lyra – AI Prompt Optimizer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Lyra – AI Prompt Optimizer")

history_file = "prompt_history.json"
if os.path.exists(history_file):
    with open(history_file, "r") as f:
        prompt_history = json.load(f)
else:
    prompt_history = []

st.sidebar.header("🛠️ Settings")
model = st.sidebar.selectbox("Select Model", ["gpt-4", "gpt-3.5-turbo", "gpt-4o"])
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

if st.button("🚀 Optimize Prompt"):
    if not user_prompt:
        st.warning("Please enter a prompt.")
    elif "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OpenAI API Key in secrets.")
    else:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            optimized_output = response.choices[0].message.content
            st.success("✅ Optimized Prompt:")
            st.markdown(optimized_output)

            input_tokens = len(system_prompt.split()) + len(full_prompt.split())
            output_tokens = len(optimized_output.split())
            estimated_cost = estimate_cost(input_tokens, output_tokens, model)

            st.markdown(f"💰 **Estimated Cost:** ${estimated_cost} (approx.)")

            prompt_history.insert(0, {"input": user_prompt, "output": optimized_output})
            with open(history_file, "w") as f:
                json.dump(prompt_history, f)

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("### 📜 Prompt History")
if prompt_history:
    for i, item in enumerate(prompt_history[:5]):
        with st.expander(f"History #{i+1}"):
            st.markdown("**Input:**")
            st.code(item['input'], language='markdown')
            st.markdown("**Output:**")
            st.markdown(item['output'])
else:
    st.info("No prompt history yet.")
