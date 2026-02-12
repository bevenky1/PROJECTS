SEARCH_QUERY_GENERATOR_PROMPT = """
[INSTRUCTION]
Analyze the Chat History and the Follow Up Input. Rephrase the Follow Up Input into a single, standalone question that is search-optimized. 
If the Follow Up Input is about the conversation history itself (e.g., "what did I ask?"), ensure the rephrased question explicitly contains the phrase "previous questions" or "conversation history".

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone search query:"""

RAG_PROMPT_TEMPLATE = """
### ROLE
You are a factual AI assistant for Air India. Your goal is to answer the USER'S QUESTION using ONLY the provided CONTEXT.

### RULES
1. If the answer is not in the CONTEXT, say exactly: "I'm sorry, I don't see that information in the documents I have."
2. Do not use your own knowledge. Use only the provided CONTEXT.
3. If the user asks about the conversation history, use the CHAT HISTORY section below.

### DATA
CHAT HISTORY:
{chat_history}

CONTEXT (PDF DOCUMENTS):
{context}

USER'S QUESTION:
{question}

### ASSISTANT ANSWER:"""

EVALUATION_PROMPT = """
### ROLE
You are a strict Auditor evaluating an AI's response vs a given Context.

### EVALUATION DATA
Context:
{context}

User's Question:
{question}

AI Response:
{response}

### SCORING SYSTEM
- 5 (Perfect): Fully answered using only context.
- 4 (Good): Answered correctly but missing minor detail.
- 3 (Average): Correct but could be more precise.
- 2 (Poor): Missed the main point or contains minor hallucination.
- 1 (Critical Failure): Incorrect, major hallucination, or ignored context.

### OUTPUT FORMAT
You MUST output ONLY a JSON object. Do not include markdown blocks.
Example Output: {{"score": 5, "reasoning": "Explanation here"}}

JSON:"""

BEDROCK_SYSTEM_PROMPT = "You are a helpful assistant for Air India queries."

# UI Prompts and Content
CHAT_INPUT_PLACEHOLDER = "What is the maximum baggage allowance?"

UI_SAMPLE_QUESTIONS = [
    "What is Air India's baggage policy?",
    "Tell me about Air India's history.",
    "What are the major air disasters of AI?",
    "List domestic routes of Feb 2025.",
    "What are all the questions that I have asked you above?"
]

# Test / Evaluation Questions
TEST_QUESTIONS = [
    "What is Air India's baggage policy for international flights?",
    "Tell me about Air India's history and its founders.",
    "What are the major air disasters associated with Air India in its history?",
    "List some of the domestic routes operated by Air India as of February 2025.",
    "What are the key service regulations for AIESL employees mentioned in the documents?",
    "Who are the major interline partners or associated airlines for Air India?",
    "What are all the questions that I have asked you above?"
]
