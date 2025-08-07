# Streamlit Multi-PDF Query Interface Powered by Gemini

This project is a **Streamlit** web application that lets you upload multiple PDF files and ask questions about their contents using the Google Gemini generative AI model with LangChain embeddings
and FAISS vector search.

---

## Features

- Upload multiple PDFs easily through a sidebar.
- Extract and chunk PDF text for better embedding and search.
- Create and persist FAISS vector index locally for fast retrieval.
- Query PDF content in natural language using the Google Gemini LLM.
- Select which PDF to query for focused answers.
- Keep conversational chat history with context for follow-ups.
- Handles errors gracefully and shows debug info.
- Uses safe deserialization for persisted vector stores.

---

## Installation

1. Clone the repository:
  git clone https://github.com/Bhutani63/Streamlit-Multi-PDF-Query-Interface-Powered-by-Gemini.git
  cd Streamlit-Multi-PDF-Query-Interface-Powered-by-Gemini

2. (Optional but recommended) Create and activate a Python virtual environment:
  python -m venv venv

  Windows
  venv\Scripts\activate
  
  macOS/Linux
  source venv/bin/activate

3. Install dependencies:
  pip install -r requirements.txt


4. Add your Google API key:

- Create a `.env` file in the project root.
- Add the line:

  ```
  GOOGLE_API_KEY=your-google-api-key
  ```

---

## Usage

Run the app with:


Then open the URL from Streamlit (usually http://localhost:8501).

### How to Use

- Upload one or more PDF files in the sidebar.
- Click **Submit & Process** to extract text, chunk, and create vector embeddings.
- Once processed, select a PDF from the dropdown to query.
- Enter your question and submit.
- The AI responds based on the selected document's content.
- Reset chat and uploaded PDFs with the **Reset Chat** button.

---

## Notes

- The app uses `gemini-2.5-flash` model from Google Generative AI.  
- Ensure your Google API key has permissions for Gemini API.  
- The vector store is saved locally; deleting `faiss_index` folder resets it.  
- Answers come only from the uploaded PDFs' content. Questions outside their scope result in "I don't know".  
- Filtering by PDF source prevents mixing answers from different documents.

---

## Project Structure

- `app.py`: Streamlit app source code.  
- `.env`: Environment variables (not included).  
- `requirements.txt`: Required libraries.  
- `faiss_index/`: Saved FAISS vector index folder.  
- `chunks.json`: Extracted text chunks for debugging.

---

## Troubleshooting

- Check console logs for detailed error tracebacks if answer generation fails.  
- Verify your Google API key and model access.  
- Adjust `k` value in vector search if answers are lacking detail.  
- Re-upload PDFs after deleting `faiss_index` if stale indices cause issues.

---

## License

MIT License

---

## Acknowledgements

- Built with [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), Google Generative AI, and FAISS.

---

