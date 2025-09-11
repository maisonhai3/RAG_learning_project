# FastAPI RAG Chatbot - Example Usage Scenarios

## 📋 Table of Contents

1. [Basic Q&A Examples](#basic-qa-examples)
2. [Advanced Search Scenarios](#advanced-search-scenarios)
3. [Developer Workflow Integration](#developer-workflow-integration)
4. [Batch Processing Examples](#batch-processing-examples)
5. [Custom Integration Examples](#custom-integration-examples)
6. [Real-World Use Cases](#real-world-use-cases)

---

## Basic Q&A Examples

### 1. Getting Started with FastAPI

**Question:** "How do I create my first FastAPI application?"

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create my first FastAPI application?",
    "max_chunks": 5
  }'
```

**Expected Response:**
```json
{
  "answer": "To create your first FastAPI application:\n\n1. Install FastAPI: `pip install fastapi uvicorn`\n2. Create a file `main.py`:\n\n```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/\")\ndef read_root():\n    return {\"Hello\": \"World\"}\n```\n\n3. Run the server: `uvicorn main:app --reload`\n4. Visit http://127.0.0.1:8000 to see your API",
  "sources": [
    {
      "title": "First Steps - FastAPI",
      "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
      "relevance_score": 0.92,
      "excerpt": "Let's start with a simple example..."
    }
  ],
  "confidence_score": 0.89,
  "processing_time": 1.2,
  "tokens_used": 145
}
```

### 2. Understanding Path Parameters

**Question:** "How do I use path parameters in FastAPI?"

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I use path parameters in FastAPI?",
    "temperature": 0.3
  }'
```

### 3. Request Body Handling

**Question:** "How do I handle JSON request bodies in FastAPI?"

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I handle JSON request bodies in FastAPI?",
    "max_chunks": 6
  }'
```

---

## Advanced Search Scenarios

### 1. Find Specific Features

**Search for dependency injection:**

```bash
curl "http://localhost:8000/search?query=dependency%20injection%20FastAPI&k=5"
```

**Expected Response:**
```json
[
  {
    "title": "Dependencies - FastAPI",
    "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
    "relevance_score": 0.94,
    "excerpt": "FastAPI has a very powerful but intuitive Dependency Injection system..."
  },
  {
    "title": "Sub-dependencies - FastAPI",
    "url": "https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/",
    "relevance_score": 0.87,
    "excerpt": "You can create dependencies that have sub-dependencies..."
  }
]
```

### 2. Security and Authentication

**Search for authentication methods:**

```bash
curl "http://localhost:8000/search?query=authentication%20security%20oauth&k=3"
```

### 3. Database Integration

**Search for database-related content:**

```bash
curl "http://localhost:8000/search?query=database%20SQLAlchemy%20ORM&k=4"
```

---

## Developer Workflow Integration

### 1. Python Script Integration

Create a helper script for your development workflow:

```python
#!/usr/bin/env python3
"""
FastAPI Documentation Assistant
Usage: python fastapi_helper.py "How do I handle errors?"
"""

import sys
import requests
import json

def ask_fastapi_question(question):
    """Ask a question to the FastAPI documentation chatbot."""
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question, "max_chunks": 5},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to API: {e}"}

def main():
    if len(sys.argv) < 2:
        print("Usage: python fastapi_helper.py \"Your question here\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    print(f"🤔 Question: {question}\n")
    
    result = ask_fastapi_question(question)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("🤖 Answer:")
    print(result.get("answer", "No answer available"))
    
    sources = result.get("sources", [])
    if sources:
        print("\n📚 Sources:")
        for i, source in enumerate(sources[:3], 1):
            print(f"{i}. {source['title']}")
            print(f"   📎 {source['url']}")
            print(f"   📊 Relevance: {source['relevance_score']:.2f}")
    
    print(f"\n⏱️  Response time: {result.get('processing_time', 0):.2f}s")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python fastapi_helper.py "How do I handle file uploads?"
python fastapi_helper.py "What is dependency injection in FastAPI?"
python fastapi_helper.py "How to test FastAPI applications?"
```

### 2. VS Code Integration

Create a VS Code task for quick documentation lookup:

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "FastAPI Docs: Ask Question",
            "type": "shell",
            "command": "python",
            "args": ["fastapi_helper.py", "${input:question}"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        }
    ],
    "inputs": [
        {
            "id": "question",
            "description": "What's your FastAPI question?",
            "default": "How do I...",
            "type": "promptString"
        }
    ]
}
```

### 3. Shell Alias for Quick Access

Add to your `.bashrc` or `.zshrc`:

```bash
# FastAPI documentation helper
alias fastapi-ask='python /path/to/fastapi_helper.py'
alias fastapi-search='curl "http://localhost:8000/search?query='

# Usage examples:
# fastapi-ask "How do I create middleware?"
# fastapi-search authentication&k=3"
```

---

## Batch Processing Examples

### 1. Multiple Questions Script

Process multiple questions efficiently:

```python
#!/usr/bin/env python3
"""
Batch process multiple FastAPI questions
"""

import asyncio
import aiohttp
import json

async def ask_question(session, question):
    """Ask a single question asynchronously."""
    try:
        async with session.post(
            'http://localhost:8000/ask',
            json={'question': question, 'max_chunks': 3}
        ) as response:
            return await response.json()
    except Exception as e:
        return {'error': str(e), 'question': question}

async def batch_questions(questions):
    """Process multiple questions in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [ask_question(session, q) for q in questions]
        return await asyncio.gather(*tasks)

def main():
    questions = [
        "How do I create a FastAPI application?",
        "What is dependency injection?",
        "How do I handle errors in FastAPI?",
        "How to implement authentication?",
        "What are FastAPI middleware?",
        "How to test FastAPI applications?",
        "How to deploy FastAPI to production?",
        "What is the difference between FastAPI and Flask?"
    ]
    
    print("🚀 Processing batch questions...")
    results = asyncio.run(batch_questions(questions))
    
    print("\n📋 Results Summary:")
    for i, (question, result) in enumerate(zip(questions, results), 1):
        print(f"\n{i}. Q: {question}")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            answer = result.get('answer', 'No answer')[:100] + "..."
            print(f"   ✅ A: {answer}")
            print(f"   📊 Confidence: {result.get('confidence_score', 0):.2f}")

if __name__ == "__main__":
    main()
```

### 2. Documentation Coverage Analysis

Analyze documentation coverage for different topics:

```python
#!/usr/bin/env python3
"""
Analyze FastAPI documentation coverage
"""

import requests
import json

def analyze_coverage():
    """Analyze documentation coverage for key FastAPI topics."""
    
    topics = {
        "Getting Started": [
            "create first FastAPI app",
            "install FastAPI",
            "basic routing"
        ],
        "Request Handling": [
            "path parameters",
            "query parameters", 
            "request body",
            "form data",
            "file uploads"
        ],
        "Response Handling": [
            "response models",
            "status codes",
            "headers",
            "cookies"
        ],
        "Advanced Features": [
            "dependency injection",
            "middleware",
            "background tasks",
            "WebSockets"
        ],
        "Security": [
            "authentication",
            "authorization",
            "OAuth2",
            "JWT tokens"
        ],
        "Database": [
            "SQLAlchemy",
            "database connections",
            "ORM models",
            "async database"
        ],
        "Testing": [
            "unit testing",
            "test client",
            "pytest FastAPI"
        ],
        "Deployment": [
            "Docker deployment",
            "production deployment",
            "HTTPS setup"
        ]
    }
    
    coverage_report = {}
    
    for category, keywords in topics.items():
        print(f"🔍 Analyzing {category}...")
        category_results = []
        
        for keyword in keywords:
            try:
                response = requests.get(
                    f"http://localhost:8000/search",
                    params={"query": keyword, "k": 3},
                    timeout=10
                )
                results = response.json()
                
                avg_relevance = (
                    sum(r['relevance_score'] for r in results) / len(results)
                    if results else 0
                )
                
                category_results.append({
                    'keyword': keyword,
                    'results_count': len(results),
                    'avg_relevance': avg_relevance,
                    'status': '✅' if avg_relevance > 0.5 else '⚠️' if avg_relevance > 0.2 else '❌'
                })
                
            except Exception as e:
                category_results.append({
                    'keyword': keyword,
                    'error': str(e),
                    'status': '❌'
                })
        
        coverage_report[category] = category_results
    
    # Print report
    print("\n📊 FASTAPI DOCUMENTATION COVERAGE REPORT")
    print("=" * 60)
    
    for category, results in coverage_report.items():
        print(f"\n🏷️  {category}:")
        for result in results:
            status = result.get('status', '❌')
            keyword = result['keyword']
            if 'error' in result:
                print(f"  {status} {keyword}: Error - {result['error']}")
            else:
                relevance = result['avg_relevance']
                count = result['results_count']
                print(f"  {status} {keyword}: {count} docs (relevance: {relevance:.3f})")
    
    # Summary statistics
    all_results = [r for results in coverage_report.values() for r in results if 'error' not in r]
    good_coverage = len([r for r in all_results if r['avg_relevance'] > 0.5])
    total_topics = len(all_results)
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total topics analyzed: {total_topics}")
    print(f"  Good coverage (>0.5): {good_coverage} ({good_coverage/total_topics*100:.1f}%)")
    print(f"  Average relevance: {sum(r['avg_relevance'] for r in all_results)/len(all_results):.3f}")

if __name__ == "__main__":
    analyze_coverage()
```

---

## Custom Integration Examples

### 1. Slack Bot Integration

Create a Slack bot that answers FastAPI questions:

```python
#!/usr/bin/env python3
"""
Slack bot for FastAPI documentation
"""

import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Initialize Slack app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

@app.message("fastapi")
def handle_fastapi_question(message, say):
    """Handle FastAPI-related questions in Slack."""
    
    question = message['text']
    user = message['user']
    
    # Extract the actual question (remove 'fastapi' trigger)
    question_text = question.replace('fastapi', '').strip()
    
    if not question_text:
        say(f"Hi <@{user}>! Ask me anything about FastAPI. Example: 'fastapi How do I create endpoints?'")
        return
    
    try:
        # Ask the FastAPI documentation chatbot
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question_text, "max_chunks": 4},
            timeout=30
        )
        result = response.json()
        
        if 'error' in result:
            say(f"Sorry <@{user}>, I encountered an error: {result['error']}")
            return
        
        # Format response for Slack
        answer = result.get('answer', 'No answer available')
        sources = result.get('sources', [])
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*FastAPI Question:* {question_text}\n\n*Answer:*\n{answer}"
                }
            }
        ]
        
        if sources:
            source_text = "\n".join([
                f"• <{source['url']}|{source['title']}> (relevance: {source['relevance_score']:.2f})"
                for source in sources[:3]
            ])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Sources:*\n{source_text}"
                }
            })
        
        say(blocks=blocks)
        
    except Exception as e:
        say(f"Sorry <@{user}>, I couldn't process your question: {str(e)}")

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
```

### 2. Discord Bot Integration

```python
#!/usr/bin/env python3
"""
Discord bot for FastAPI documentation
"""

import discord
import requests
import asyncio
from discord.ext import commands

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command(name='fastapi')
async def fastapi_question(ctx, *, question):
    """Answer FastAPI questions in Discord."""
    
    # Show typing indicator
    async with ctx.typing():
        try:
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question, "max_chunks": 4},
                timeout=30
            )
            result = response.json()
            
            if 'error' in result:
                await ctx.send(f"❌ Error: {result['error']}")
                return
            
            # Create Discord embed
            embed = discord.Embed(
                title="🚀 FastAPI Documentation Assistant",
                description=f"**Question:** {question}",
                color=0x009688
            )
            
            answer = result.get('answer', 'No answer available')
            if len(answer) > 1024:
                answer = answer[:1021] + "..."
            
            embed.add_field(
                name="📝 Answer", 
                value=answer, 
                inline=False
            )
            
            sources = result.get('sources', [])
            if sources:
                source_text = "\n".join([
                    f"[{source['title']}]({source['url']}) (relevance: {source['relevance_score']:.2f})"
                    for source in sources[:3]
                ])
                embed.add_field(
                    name="📚 Sources", 
                    value=source_text, 
                    inline=False
                )
            
            embed.set_footer(
                text=f"Response time: {result.get('processing_time', 0):.2f}s"
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error processing question: {str(e)}")

@bot.command(name='search')
async def search_docs(ctx, *, query):
    """Search FastAPI documentation."""
    
    try:
        response = requests.get(
            f"http://localhost:8000/search",
            params={"query": query, "k": 5},
            timeout=10
        )
        results = response.json()
        
        if not results:
            await ctx.send(f"No results found for: {query}")
            return
        
        embed = discord.Embed(
            title="🔍 FastAPI Documentation Search",
            description=f"**Query:** {query}",
            color=0x2196F3
        )
        
        for i, result in enumerate(results[:5], 1):
            embed.add_field(
                name=f"{i}. {result['title']}",
                value=f"[Link]({result['url']}) (relevance: {result['relevance_score']:.2f})\n{result['excerpt'][:100]}...",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error searching: {str(e)}")

# Run bot
bot.run('YOUR_DISCORD_BOT_TOKEN')
```

### 3. Web Interface

Simple HTML interface for the chatbot:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FastAPI Documentation Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .question-box {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background: #009688;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background: #00796b;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .answer-box {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #009688;
        }
        .sources {
            margin-top: 20px;
        }
        .source-item {
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 6px;
            border: 1px solid #eee;
        }
        .source-title {
            font-weight: bold;
            color: #009688;
        }
        .relevance-score {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 FastAPI Documentation Assistant</h1>
            <p>Ask any question about FastAPI and get instant answers from the official documentation!</p>
        </div>
        
        <div class="question-box">
            <textarea id="questionInput" rows="4" placeholder="How do I create a FastAPI application?"></textarea>
            <button onclick="askQuestion()">Ask Question</button>
            <button onclick="searchDocs()" style="background: #2196F3;">Search Docs</button>
        </div>
        
        <div class="loading" id="loading">
            <p>🤔 Thinking... Please wait</p>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('Please enter a question!');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('http://localhost:8000/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        max_chunks: 5
                    })
                });
                
                const result = await response.json();
                displayAnswer(result);
                
            } catch (error) {
                displayError('Failed to get answer: ' + error.message);
            } finally {
                showLoading(false);
            }
        }
        
        async function searchDocs() {
            const query = document.getElementById('questionInput').value.trim();
            if (!query) {
                alert('Please enter a search query!');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(query)}&k=5`);
                const results = await response.json();
                displaySearchResults(results);
                
            } catch (error) {
                displayError('Failed to search: ' + error.message);
            } finally {
                showLoading(false);
            }
        }
        
        function displayAnswer(result) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="answer-box">
                    <h3>🤖 Answer:</h3>
                    <p style="white-space: pre-wrap;">${result.answer || 'No answer available'}</p>
                    
                    <div style="margin-top: 15px; font-size: 14px; color: #666;">
                        ⏱️ Response time: ${result.processing_time?.toFixed(2) || 0}s |
                        📊 Confidence: ${(result.confidence_score * 100)?.toFixed(1) || 0}% |
                        🔢 Tokens used: ${result.tokens_used || 0}
                    </div>
            `;
            
            if (result.sources && result.sources.length > 0) {
                html += `
                    <div class="sources">
                        <h4>📚 Sources:</h4>
                `;
                
                result.sources.forEach((source, index) => {
                    html += `
                        <div class="source-item">
                            <div class="source-title">${index + 1}. ${source.title}</div>
                            <div><a href="${source.url}" target="_blank">${source.url}</a></div>
                            <div class="relevance-score">Relevance: ${(source.relevance_score * 100).toFixed(1)}%</div>
                            <div style="margin-top: 5px; font-size: 14px;">${source.excerpt}</div>
                        </div>
                    `;
                });
                
                html += `</div>`;
            }
            
            html += `</div>`;
            resultsDiv.innerHTML = html;
        }
        
        function displaySearchResults(results) {
            const resultsDiv = document.getElementById('results');
            
            if (!results || results.length === 0) {
                resultsDiv.innerHTML = '<div class="answer-box"><p>No results found.</p></div>';
                return;
            }
            
            let html = `
                <div class="answer-box">
                    <h3>🔍 Search Results:</h3>
            `;
            
            results.forEach((result, index) => {
                html += `
                    <div class="source-item">
                        <div class="source-title">${index + 1}. ${result.title}</div>
                        <div><a href="${result.url}" target="_blank">${result.url}</a></div>
                        <div class="relevance-score">Relevance: ${(result.relevance_score * 100).toFixed(1)}%</div>
                        <div style="margin-top: 5px; font-size: 14px;">${result.excerpt}</div>
                    </div>
                `;
            });
            
            html += `</div>`;
            resultsDiv.innerHTML = html;
        }
        
        function displayError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="answer-box" style="border-left-color: #f44336;">
                    <h3>❌ Error:</h3>
                    <p>${message}</p>
                </div>
            `;
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        // Allow Enter key to submit
        document.getElementById('questionInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                askQuestion();
            }
        });
    </script>
</body>
</html>
```

---

## Real-World Use Cases

### 1. Development Team Onboarding

Create an onboarding script for new team members:

```python
#!/usr/bin/env python3
"""
FastAPI Onboarding Assistant
"""

import requests
import time

def onboarding_session():
    """Interactive onboarding session for FastAPI."""
    
    print("🎉 Welcome to FastAPI Onboarding!")
    print("This assistant will help you learn FastAPI basics.\n")
    
    topics = [
        "What is FastAPI and why should I use it?",
        "How do I install and set up FastAPI?",
        "How do I create my first FastAPI application?",
        "How do I define API endpoints?",
        "How do I handle request parameters?",
        "How do I validate request data?",
        "How do I handle errors in FastAPI?",
        "How do I test FastAPI applications?",
        "How do I document my FastAPI API?"
    ]
    
    for i, topic in enumerate(topics, 1):
        print(f"📚 Lesson {i}: {topic}")
        input("Press Enter to continue...")
        
        # Get answer from the chatbot
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": topic, "max_chunks": 4}
        )
        result = response.json()
        
        print(f"\n💡 Answer:")
        print(result.get('answer', 'No answer available'))
        
        sources = result.get('sources', [])
        if sources:
            print(f"\n📖 Learn more:")
            for source in sources[:2]:
                print(f"• {source['title']}: {source['url']}")
        
        print("\n" + "="*60 + "\n")
        time.sleep(1)
    
    print("🎓 Onboarding complete! You're ready to build with FastAPI!")

if __name__ == "__main__":
    onboarding_session()
```

### 2. Code Review Assistant

Help with code review by checking best practices:

```python
#!/usr/bin/env python3
"""
FastAPI Code Review Assistant
"""

import requests
import ast
import re

def analyze_fastapi_code(file_path):
    """Analyze FastAPI code and provide suggestions."""
    
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Extract potential issues and questions
    issues = []
    
    # Check for common patterns
    if '@app.get' in code or '@app.post' in code:
        issues.append("How do I handle errors properly in FastAPI endpoints?")
    
    if 'Depends(' in code:
        issues.append("What are FastAPI dependency injection best practices?")
    
    if 'BaseModel' in code:
        issues.append("How do I properly validate request models in FastAPI?")
    
    if 'async def' in code:
        issues.append("What are async/await best practices in FastAPI?")
    
    # Get suggestions for each issue
    suggestions = []
    for issue in issues:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": issue, "max_chunks": 3}
        )
        result = response.json()
        suggestions.append({
            'issue': issue,
            'suggestion': result.get('answer', 'No suggestion available')
        })
    
    return suggestions

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python code_review.py <fastapi_file.py>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"🔍 Analyzing {file_path}...")
    
    suggestions = analyze_fastapi_code(file_path)
    
    print(f"\n📋 Code Review Suggestions:")
    print("="*50)
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['issue']}")
        print(f"💡 {suggestion['suggestion'][:300]}...")

if __name__ == "__main__":
    main()
```

### 3. Documentation Generator

Generate documentation based on existing knowledge:

```python
#!/usr/bin/env python3
"""
FastAPI Documentation Generator
"""

import requests
import json

def generate_documentation(topics):
    """Generate documentation for specified topics."""
    
    documentation = {}
    
    for topic in topics:
        print(f"📝 Generating documentation for: {topic}")
        
        # Get detailed information
        response = requests.post(
            "http://localhost:8000/ask",
            json={
                "question": f"Explain {topic} in FastAPI with examples",
                "max_chunks": 8
            }
        )
        result = response.json()
        
        documentation[topic] = {
            'content': result.get('answer', ''),
            'sources': result.get('sources', [])
        }
    
    return documentation

def format_documentation(docs):
    """Format documentation as Markdown."""
    
    markdown = "# FastAPI Documentation\n\n"
    
    for topic, content in docs.items():
        markdown += f"## {topic}\n\n"
        markdown += f"{content['content']}\n\n"
        
        if content['sources']:
            markdown += "### References\n\n"
            for source in content['sources']:
                markdown += f"- [{source['title']}]({source['url']})\n"
            markdown += "\n"
        
        markdown += "---\n\n"
    
    return markdown

def main():
    topics = [
        "Getting Started with FastAPI",
        "Request and Response Handling", 
        "Dependency Injection",
        "Security and Authentication",
        "Database Integration",
        "Testing FastAPI Applications",
        "Deployment Best Practices"
    ]
    
    print("🚀 Generating FastAPI documentation...")
    docs = generate_documentation(topics)
    
    markdown = format_documentation(docs)
    
    with open('fastapi_documentation.md', 'w') as f:
        f.write(markdown)
    
    print("✅ Documentation generated: fastapi_documentation.md")

if __name__ == "__main__":
    main()
```

---

These examples demonstrate the versatility and power of the FastAPI RAG Chatbot system. You can adapt these patterns to create custom integrations that fit your specific workflow and requirements.
