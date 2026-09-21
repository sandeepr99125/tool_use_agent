import streamlit as st
from agent import ToolUseAgent

st.set_page_config(page_title="AI Agent Tool Loop", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Turn Autonomous Tool Agent")
st.write("Calculates order status, executes refund logic, and drafts emails across autonomous tool loops.")

# Sidebar options & instructions
with st.sidebar:
    st.header("⚙️ Configuration")
    max_steps = st.slider("Max Multi-Turn Loop Steps", min_value=1, max_value=8, value=5)
    st.markdown("---")
    st.markdown("**Fallback Sequence:**")
    st.caption("1. Gemini 2.5 Flash Lite\n2. OpenRouter (`openrouter/free`)\n3. NVIDIA Nemotron 3 Ultra\n4. Google Gemma 4 31B\n5. Cohere North Mini Code")

user_prompt = st.text_area(
    "User Request:",
    value="Check orders for customer CUST101, calculate a 10% refund on their order, and send them a confirmation email with the details.",
    height=100
)

if st.button("Execute Agent Task", type="primary"):
    if not user_prompt.strip():
        st.warning("Please enter a valid request.")
    else:
        with st.spinner("Agent running autonomous tool loop..."):
            try:
                agent = ToolUseAgent()
                result = agent.run(user_prompt, max_steps=max_steps)
                
                # --- UI Status Badges & Metrics ---
                st.subheader("Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Status", value="Completed ✅")
                with col2:
                    st.metric(label="Provider / Model", value=result.get("provider", "Unknown"))
                with col3:
                    st.metric(label="Autonomous Loop Turns", value=f"{result.get('steps', 1)} / {max_steps}")
                
                st.markdown("---")
                
                # Render Final Text Output
                st.markdown("### Agent Answer")
                st.info(result.get("answer", "No text answer returned."))
                
                # Render Execution Step Logs in Expander Badge
                tool_calls = result.get("tool_calls", [])
                if tool_calls:
                    with st.expander(f"🛠️ Execution Trace & Tool Call Logs ({len(tool_calls)} total calls)", expanded=True):
                        for log in tool_calls:
                            st.write(f"**Turn {log.get('step', 1)}** — Called function: `{log.get('tool')}`")
                            st.json(log.get("args"))
                else:
                    st.caption("No tools were required to complete this request.")

            except Exception as e:
                st.error(f"Error executing agent: {e}")