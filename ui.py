import streamlit as st
import requests

# 1) Page config & hide menus
st.set_page_config(
    page_title="Census Field Companion",
    layout="wide",
)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 2) Sidebar: Manuals & About
st.sidebar.title("🗄️ Manuals Used")
for m in [
    "HouseListing_Housing_Census_2011.pdf",
    "Abridged_Houselist_Household_Schedule.pdf",
    "Supervisor_Handbook_Flowcharts.pdf",
    "Household_Schedule_Manual.pdf",
    "Houselisting_Housing_Census_Schedule.pdf",
    "Urban_Frame_Jurisdiction/*"
]:
    st.sidebar.write(f"- {m}")

st.sidebar.markdown("""

**Helps**  
- **Enumerators** with SOP guidance  
- **Supervisors** surface field issues  
- **Managers** see aggregate insights  

**Powered by**  
OpenAI GPT-3.5 Turbo 

©VS

""")

# 3) Main UI
st.title("📡 PoC: Census Field Companion for ORGI")
st.markdown("Select your role, review sample questions, and type your own question below.")

role = st.selectbox("👤 Your Role", options=["enumerator","supervisor","manager"])
sample_questions = {
  "enumerator": ["What if a house is locked?", "How to record a vacant dwelling?"],
  "supervisor": ["Show me hotspots of locked houses today.", "What’s the compliance rate?"],
  "manager": ["Aggregate data entry errors?", "Overall completion percentage?"]
}[role]

st.markdown("**Sample questions for your role:**")
for q in sample_questions:
    st.write(f"- {q}")

query = st.text_input("💬 Ask your question")
if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Getting answer..."):
            try:
                api_url = "https://census-field-companion-your-username.streamlit.app/chat"
                res = requests.post(api_url, json={"question": query, "role": role}, timeout=30)
                # Check for HTTP errors
                res.raise_for_status()
                # Attempt to parse JSON
                data = res.json()
                answer = data.get("answer")
                if not answer:
                    st.error("No ‘answer’ field in response JSON.")
                else:
                    st.markdown("**Answer:**")
                    st.write(answer)
            except requests.exceptions.JSONDecodeError:
                st.error(f"Invalid JSON received. Status {res.status_code} – Response text:\n\n{res.text}")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
