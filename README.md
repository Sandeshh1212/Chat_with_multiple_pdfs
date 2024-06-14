# Chat_with_multiple_pdfs

This Streamlit application allows users to upload multiple PDF files, process their contents, and interactively ask questions based on the text within these PDFs. Leveraging the capabilities of Google Generative AI and the FAISS vector store, the application provides detailed responses to user queries.

Features
Multiple PDF Uploads: Upload multiple PDF documents simultaneously.
Text Extraction: Extracts text from all uploaded PDF files.
Text Chunking: Splits extracted text into manageable chunks for processing.
Embeddings and Vector Store: Uses Google Generative AI to create embeddings and stores them in a FAISS index for efficient similarity search.
Interactive Q&A: Allows users to ask questions about the content of the uploaded PDFs and receive detailed answers.
Conversational AI: Utilizes Google Generative AI for natural and detailed responses.
How It Works
Upload PDFs: Users can upload multiple PDF files using the sidebar.
Text Processing: The application extracts text from each PDF and splits it into chunks.
Vector Creation: Text chunks are converted into embeddings using Google Generative AI and stored in a FAISS index.
Ask Questions: Users can enter questions in a text input field, and the application searches for relevant text chunks and provides a detailed answer.
