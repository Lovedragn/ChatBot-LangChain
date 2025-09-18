import os, json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from Services.Database_actions import create_task, update_task, delete_task, fetch_tasks

# Load Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash" ,api_key=GOOGLE_API_KEY)

# Define prompt
prompt = ChatPromptTemplate.from_messages([("system", 
 "You are a task manager assistant. "
 "Always reply ONLY in valid JSON. "
 "Rules: "
 "For 'delete' or 'update' actions, ALWAYS use the numeric 'task_id' (integer). "
 "Never put the title in 'task_id'. "
 "For 'create': include task_date, title, user_email. "
 "For 'fetch': include user_email (optional). "
 "No explanations, no markdown, only JSON."),("human", "{user_input}")])



def ask_bot(user_input: str):
    chain = prompt | chat
    result = chain.invoke({"user_input": user_input})
    raw = result.content or "{}"

    # Clean Markdown code fences
    if raw.startswith("```"):
        raw = raw.strip("`")  # remove all backticks
        if raw.startswith("json"):
            raw = raw[4:].strip()  # remove "json" language tag

    print("CLEANED RESPONSE:", raw)
    return raw

def handle_action(response: str):
    try:
        data = json.loads(response)
    except Exception as e:
        print("JSON Parse Error:", e, "Response:", response)
        return "❌ Invalid response from AI."

    action = data.get("action")
    
    if action == "create":
        task = create_task(data["task_date"], data["title"], data["user_email"])
        return f"✅ Task {task.id} created: {task.title}"

    elif action == "update":
        task = update_task(data["task_id"], data.get("title"), data.get("task_date"), data.get("user_email"))
        return f"🔄 Task {task.id} updated!" if task else "❌ Task not found."

    elif action == "delete":
        task_id = data.get("task_id")
        if not task_id or not str(task_id).isdigit():
            return "❌ Invalid or missing task_id for delete."
        task = delete_task(int(task_id))
        return f"🗑️ Task {task.id} deleted!" if task else "❌ Task not found."


    elif action == "fetch":
        tasks = fetch_tasks(data.get("user_email"))
        return [{"id": t.id, "date": t.task_date, "title": t.title, "email": t.user_email} for t in tasks]

    return "❌ Unknown action."
