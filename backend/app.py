from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph
from typing import TypedDict
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

USER_AVATAR = os.getenv("USER_AVATAR_URL")
AI_AVATAR = os.getenv("AI_AVATAR_URL")

# === Setup Flask ===
app = Flask(__name__)
CORS(app)

# === Gemini API Key ===
genai.configure(api_key="GOOGLE_API_KEY")

# === Load Document ===
with open("document.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(raw_text)
documents = [Document(page_content=chunk) for chunk in chunks]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
doc_texts = [doc.page_content for doc in documents]
doc_vectors = embedding_model.embed_documents(doc_texts)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class GraphState(TypedDict):
    query: str
    context: str
    answer: str

def retrieve_node(state: GraphState) -> GraphState:
    query = state["query"]
    query_vec = embedding_model.embed_query(query)

    scored = [
        (cosine_similarity(query_vec, vec), doc.page_content)
        for vec, doc in zip(doc_vectors, documents)
    ]
    top_context = "\n\n".join([doc for score, doc in sorted(scored, reverse=True)[:3] if score > 0.6])
    return {"query": query, "context": top_context, "answer": ""}

# === RAG Answer Node ===
def answer_node(state: GraphState) -> GraphState:
    query = state["query"]
    context = state["context"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.2)

    if context.strip():
        prompt = f"""Use the following context to answer the question:

Context:
{context}

Question:
{query}
"""
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(query)

    return {"query": query, "context": context, "answer": response.content.strip()}


builder = StateGraph(GraphState)
builder.add_node("RETRIEVE", retrieve_node)
builder.add_node("ANSWER", answer_node)
builder.set_entry_point("RETRIEVE")
builder.add_edge("RETRIEVE", "ANSWER")
builder.set_finish_point("ANSWER")
graph = builder.compile()

@app.route('/send_message', methods=['POST'])
def send_message():
    user_msg = request.json.get('message')

    try:
        result = graph.invoke({"query": user_msg, "context": "", "answer": ""})
        ai_text = result["answer"]
    except Exception as e:
        ai_text = f"Error: {str(e)}"

    return jsonify([
        {
            "sender": "user",
            "text": user_msg,
            "avatar": USER_AVATAR  
        },
        {
            "sender": "ai",
            "text": ai_text,
            "avatar": AI_AVATAR  
        }
    ])


if __name__ == '__main__':
    app.run(debug=True)
