# Course Chatbot

A Flask-based intelligent chatbot system that answers questions about courses using natural language processing and semantic search.

## Features

- 🤖 AI-powered question answering using OpenAI's API
- 🔍 Semantic search using Sentence Transformers
- 💨 Fast document retrieval with vector similarity
- 🌐 Web scraping capabilities for course content
- 🚀 RESTful API interface
- 📝 Comprehensive logging system

## Prerequisites

- Python 3.8+
- OpenAI API key
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/course-chatbot.git
cd course-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the server:
```bash
python src/chatbot.py
```

2. The API will be available at `http://localhost:5000/chat`

3. Send POST requests with questions:
```bash
curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What courses are available?"}'
```

## API Documentation

### POST /chat
Send questions to the chatbot.

**Request Body:**
```json
{
    "question": "string"
}
```

**Response:**
```json
{
    "answer": "string",
    "status": "success",
    "timestamp": "ISO-8601 timestamp"
}
```

## Project Structure
```
course-chatbot/
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── config.py
│   └── exceptions.py
├── static/
│   └── index.html
├── tests/
│   └── test_chatbot.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
