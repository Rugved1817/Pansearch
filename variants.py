from openai import OpenAI
import sys

# ---------------------------------------------
# Configure client to use your local Ollama Llama 3.2 model
# ---------------------------------------------
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Use localhost for Ollama
    api_key="ollama" # Use "ollama" as the API key
)

MODEL_NAME = "llama3.2:latest"

# ---------------------------------------------
# System prompt (defines persona, rules, and format)
# ---------------------------------------------
# <-- MODIFIED: This prompt is now highly specific to your example output
SYSTEM_PROMPT = """
You are a bilingual name transliteration expert for Indian languages,
specializing in English (Latin) and Marathi (Devanagari).

Your task is to generate name variants matching the user's example format.

GUIDELINES:
1.  **English Variants List:**
    * Include 3-4 full phonetic spellings (e.g., Rajiv, Rajeev, Raajiv).
    * *Also* include short forms like "Firstname L" (e.g., Rajiv P) and
        "F Lastname" (e.g., R Patel) and initials (e.g., RP).
2.  **Marathi Variants List:**
    * Include 4-5 full name phonetic and script variations
        (e.g., राजीव पटेल, राजिव पटील, रजीव पटेल, राजीव् पटेल).
3.  **Initials List:**
    * *Repeat* the short forms from the English list (e.g., R Patel, Rajiv P, RP).
    * *Add* new initial-only formats like "F.L." (e.g., R.P.) or
        "F M Lastname" (e.g., R J Patel, if a middle name is implied).

FORMATTING (CRITICAL):
You MUST output *only* plain text.
DO NOT use markdown (like **, `, *, etc.).
DO NOT use JSON.
DO NOT add any conversational text, pleasantries, or explanations.
Your output must follow this exact structure:

English Variants:
<list of variants as described in guideline 1>

Marathi Variants:
<list of variants as described in guideline 2>

Initials:
<list of variants as described in guideline 3>
"""

# ---------------------------------------------
# Prompt template
# ---------------------------------------------
def build_prompt(name: str) -> str:
    """Creates the user-facing prompt with the specific name."""
    return f"""Given the name: "{name}"

Generate the variants."""

# ---------------------------------------------
# Core function
# ---------------------------------------------
def generate_variants(name: str):
    prompt = build_prompt(name)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        print(response.choices[0].message.content.strip())
    
    except Exception as e:
        print(f"Error: Could not connect to Ollama or generate response.")
        print(f"Details: {e}")
        print(f"Please ensure Ollama is running and the model '{MODEL_NAME}' is available.")
        sys.exit(1)


# ---------------------------------------------
# CLI entry point
# ---------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python variants_llama.py \"<Full Name>\"")
        print("Example: python variants_llama.py \"Rajiv Patel\"")
        sys.exit(1)

    name = " ".join(sys.argv[1:]).strip()
    
    if not name:
        print("Error: No name provided.")
        sys.exit(1)
        
    generate_variants(name)