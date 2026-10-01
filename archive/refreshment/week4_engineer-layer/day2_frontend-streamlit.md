Welcome to **Week 4, Day 2: The Frontend (Streamlit).**

You have a backend (Ollama/vLLM) serving your model.
But normal people cannot use `curl` or Python scripts. They need a website.

As a backend engineer, you probably hate writing HTML/CSS/React.
**Good news:** You don't have to.
We use **Streamlit**. It turns a Python script into a full web app automatically.

---

**The Concept: Reactive UI**
Streamlit works differently than Flask/Django.

- **Flask:** You define routes (`/`, `/api`). You handle state in a database.
- **Streamlit:** You write a script that runs from top to bottom. Whenever the user clicks a button, the **entire script runs again**.
  - It feels inefficient, but for internal AI tools, it is incredibly fast to build.

**The Code: Your Own "ChatGPT"**

We will build a web interface that connects to your local `sarcastic-bot` (or any Ollama model).

**Prerequisite:**

```
pip install streamlit
```

Create `week4_day2_ui.py`:

```Python
import streamlit as st
from openai import OpenAI

# 1. Config
st.set_page_config(page_title="Sarcastic Bot", page_icon="🤖")
st.title("🤖 Sarcastic Support Agent")

# 2. Connect to Local Backend (Ollama)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 3. Session State (The Memory)
# Streamlit reruns the script on every click. We need 'session_state' to remember the chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
# Loop through the history and draw the bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("Complain about your computer here..."):
    # A. Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # C. Get Response from AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for the typing effect

        # Call the API
        response = client.chat.completions.create(
            model="sarcastic-bot", # Make sure this matches your Ollama model name
            messages=st.session_state.messages,
        )

        bot_reply = response.choices[0].message.content

        # Display reply
        message_placeholder.markdown(bot_reply)

    # D. Add bot reply to history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
```

**Run It**
In your terminal:

```Bash
streamlit run week4_day2_ui.py
```

A browser tab will open (usually `http://localhost:8501`).
Type "My internet is slow."
It should reply (sarcastically) in a nice chat bubble interface.

---

**Why this stack wins**

- **No JavaScript:** You built a reactive Single Page App (SPA) in 50 lines of Python.

- **State Management:** st.session_state handled the chat history logic that usually requires Redux or React Context.

- **Backend Connection:** It talks directly to your API (Ollama).

---

**Week 4 Day 2 Assignment: "The Parameter Slider"**

1. Add a **Sidebar** to the app (`st.sidebar`).
2. Add a **Slider** for "Creativity" (Temperature) ranging from 0.0 to 1.0.
3. Pass that value into the `client.chat.completions.create(..., temperature=my_slider_val`) call.
   - _Result: You can now tweak how "crazy" the bot is in real-time without restarting the server._
