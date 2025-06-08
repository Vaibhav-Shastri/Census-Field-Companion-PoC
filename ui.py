import streamlit as st
import requests

st.set_page_config(page_title="Census Field Companion", layout="wide")

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
**Powered by**  
OpenAI GPT-3.5 Turbo  
**Helps**  
- Enumerators: step-by-step SOP guidance  
- Supervisors: surface common field issues  
- Managers: aggregate process insights  
""")

st.title("📡 Census Field Companion")

st.markdown("Welcome… Select your role and ask your question below.")

role = st.selectbox("Your Role", ["enumerator","supervisor","manager"])
samples = {
  "enumerator":["What if a house is locked?","How to record a vacant dwelling?"],
  "supervisor":["Show me hotspots of locked houses today.","What’s the compliance rate?"],
  "manager":["Aggregate data entry errors?","Overall completion percentage?"]
}[role]
st.markdown("**Sample questions:**")
for q in samples: st.write(f"- {q}")

query = st.text_input("Ask your question")
if st.button("Submit"):
    if not query: st.warning("Enter a question.")
    else:
        with st.spinner("Thinking…"):
            api_url = "https://<YOUR-APP>.streamlit.app/chat"  # update later
            r = requests.post(api_url, json={"question":query,"role":role}, timeout=30)
            if r.ok:
                st.markdown("**Answer:**")
                st.write(r.json().get("answer",""))
            else:
                st.error(f"Error {r.status_code}: {r.text}")
