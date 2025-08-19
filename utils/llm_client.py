import os
import google.generativeai as genai
from groq import Groq
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
hf_client = InferenceClient(token=os.getenv("HF_API_KEY"))


def query_llm(prompt: str) -> str:
    """
    Try Gemini first, then Groq, then HuggingFace.
    """
    # --- Try Gemini ---
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"[Gemini Error] {e}")
        print("[Fallback] Gemini failed → Groq...")

    # --- Try Groq ---
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # you can swap to 70B if quota allows
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[Groq Error] {e}")
        print("[Fallback] Groq failed → HuggingFace...")

    # --- Try Hugging Face ---
    try:
        resp = hf_client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            prompt=prompt,
            max_new_tokens=512,
        )
        return resp
    except Exception as e:
        print(f"[HuggingFace Error] {e}")
        raise RuntimeError("All LLM backends failed.")
