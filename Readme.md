# Chat Summarizer

This script is to summarize chat logs by extracting key information such as the number of exchanges, main topics, and most common keywords. It supports both simple keyword extraction and TF-IDF-based keyword ranking.

## Features

- Parse chat logs to separate user and AI messages.
- Extract and rank keywords using:
  - Simple frequency-based method.
  - TF-IDF method.
- Summarize chat logs with key statistics.

## Requirements

- Python 3.11 or higher
- Required Python libraries:
  - `nltk`
  - `scikit-learn`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nafeu-khan/qtech.git
   cd qtech
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage

Run the script with the path to a `.txt` file or a folder containing `.txt` files:

```bash
python chat_summarize.py <path_to_file_or_folder>
```

### Options

- `--tfidf`: Use TF-IDF for keyword extraction.

### Example

```bash
python chat_summarize.py ./chats --tfidf
```

### Sample input/output
#### Sample Input

Sample chat log file (`chat1.txt`):
```
User: Hello!
AI: Hi! How can I assist you today?
User: Can you explain what machine learning is?
AI: Certainly! Machine learning is a field of AI that lets systems learn from data.
User: Nice. What languages are popular for machine-learning work?
AI: Python is the most common because of its rich ecosystem (NumPy, pandas, scikit-learn, TensorFlow).
User: Besides data science, what else can I do with Python?
AI: Web development (Django, FastAPI), automation, scripting, even game development with libraries like Pygame.
User: Thanks. Any resources for a beginner?
AI: The official Python tutorial and “Automate the Boring Stuff” are great starting points.

```

#### Sample Output
Summarizing chat\chat.txt...
Summary:
 - The conversation had 10 exchanges.
 - The user asked mainly about python and its uses.
 - Most common keywords: python, machine, learning, data, development.

---

## Logging

The script uses Python's `logging` module to provide runtime information.

## License

This project is licensed under the MIT License.