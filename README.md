# Student Helper

Collection of Python scripts for student productivity: summarization, task extraction, and audio transcription.

## Scripts

| Script | Description |
|--------|-------------|
| `Summarize.py` | Text summarization using AI |
| `To-Do Extraction.py` | Extracts action items from text |
| `transcribe.py` | Audio transcription |

## Setup

1. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys to `.env` file (copy from `.env.example`)

## Usage

```bash
python Summarize.py
python "To-Do Extraction.py"
python transcribe.py
```

## Requirements

- Python 3.8+
- OpenAI API key (or other LLM provider)

## License

MIT