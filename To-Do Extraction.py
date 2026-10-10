from llama_cpp import Llama
import spacy
import re
import dateparser
from functools import lru_cache

# Configuration
CONFIG = {
    "model_path": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "n_ctx": 4096,
    "n_gpu_layers": 15,
    "n_threads": 6,
    "n_batch": 512
}

# Load multilingual NLP models
nlp_en = spacy.load("en_core_web_sm")
nlp_ar = spacy.load("ar_core_web_sm")  # Requires separate installation

llm = Llama(**CONFIG)

# Bilingual patterns
PATTERNS = {
    "en": [
        [{"POS": "VERB"}, {"POS": "PART", "OP": "*"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["need", "must"]}}, {"POS": "VERB"}]
    ],
    "ar": [
        [{"POS": "VERB"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["يجب", "ينبغي"]}}, {"POS": "VERB"}]  # 'must', 'should'
    ]
}

def detect_language(text):
    """Simple language detection using Unicode ranges"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars/len(text) > 0.3 else "en"

@lru_cache(maxsize=1000)
def process_text(text, lang):
    """Language-specific text processing"""
    nlp = nlp_ar if lang == "ar" else nlp_en
    doc = nlp(text)
    
    matcher = spacy.matcher.Matcher(nlp.vocab)
    matcher.add("TASKS", PATTERNS[lang])
    matches = matcher(doc)
    
    return list({doc[start:end].text for _, start, end in matches})

def format_prompt(candidates, lang):
    """Bilingual prompt formatting"""
    system_prompt = {
        "en": "Extract tasks from these candidates in English and Arabic:",
        "ar": "استخرج المهام من هذه العناصر باللغتين الإنجليزية والعربية:"
    }[lang]
    
    return f"""<s>[INST] <<SYS>>
{system_prompt}
- Use bullet points
- Format dates as YYYY-MM-DD
- Preserve original language
<</SYS>>

Candidates:
{chr(10).join(candidates)}
Current Date: {datetime.now().strftime('%Y-%m-%d')}
[/INST]"""

def extract_todos(text):
    lang = detect_language(text)
    candidates = process_text(text, lang)
    
    # Bilingual processing prompt
    prompt = format_prompt(candidates, lang)
    response = llm(prompt, max_tokens=1024, temperature=0.2)
    
    return post_process(response['choices'][0]['text'], lang)

def post_process(text, lang):
    """Bilingual post-processing"""
    # Date normalization
    text = re.sub(
        r'(\d{1,2}\s+[\u0621-\u064A]+\s+\d{4})',  # Arabic date pattern
        lambda m: dateparser.parse(m.group(1), languages=[lang]).strftime('%Y-%m-%d'),
        text
    )
    
    # Directionality handling
    if lang == "ar":
        text = '\u202B' + text + '\u202C'  # RTL markers
    
    return text

# Usage
arabic_text = "يجب إصلاح الخطأ في النظام بحلول الجمعة القادمة"
english_text = "Need to update API documentation before 2024-03-15"

print(extract_todos(arabic_text))
print(extract_todos(english_text))