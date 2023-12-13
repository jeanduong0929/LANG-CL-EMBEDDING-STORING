# LangChain PDF Text Retrieval and Conversation Lab Activity

## Objectives
- Develop a comprehensive understanding of text extraction from PDF documents using Python.
- Implement and grasp the concept of text embedding for converting natural language into machine-readable vectors.
- Gain practical experience in integrating text embeddings into a conversational AI interface using LangChain.
- Learn to create a ConversationalRetrievalChain that effectively uses embedded text for responding to user queries.

## Prerequisites
- A working Python environment (Python 3.x recommended).
- Basic to intermediate knowledge of Python programming.
- Familiarity with handling files in Python, especially reading from PDFs.
- Basic understanding of natural language processing (NLP) concepts, particularly text embeddings.
- Prior experience with conversational AI concepts or tools will be beneficial but not necessary.
- Access to an OpenAI API key for using embedding models.

## Lab Description:
This lab involves building a backend application for text retrieval and conversation handling. You will extract text from PDFs, split the text into manageable chunks, embed these chunks for efficient retrieval, and create a conversational interface that responds to user queries by referencing the text data.

## Setup Instructions
1. Ensure Python and necessary tools are installed on your system.
2. Install dependencies using pip or pip3:

```bash
pip install -r requirements.txt
```

or

```bash
pip3 install -r requirements.txt
```

3. Set up your OpenAI API key:
  - Create a file named .env in the root of your project directory (same level as your src folder).
  - Open the .env file and add the following line:

```
OPENAI_API_KEY=your_api_key_here
```

  - Replace your_api_key_here with your actual OpenAI API key.
  - Save the file.

## Lab Overview
In this lab, you will delve into the world of LangChain, focusing on extracting text from PDF documents, processing this text for embedding, and integrating it into a conversational AI interface. You will begin by setting up your environment, including managing API keys securely. Then, you will embark on a journey of embedding text data and utilizing it for conversational purposes. The core of this lab involves implementing functions to extract text from a PDF, split it into manageable chunks, embed these chunks for efficient retrieval, and create a conversational interface that can respond to queries based on the embedded text data.

What is Text Embedding?

Text embedding is a process of converting textual data into numerical form, often as vectors, so that machines can understand and process natural language. It involves using models like BERT or GPT to transform text into a high-dimensional space where semantic relationships and linguistic patterns can be quantitatively analyzed.

How Does It Differ From Traditional Text Processing?

Traditional Text Processing: Traditional methods often involve keyword matching or rule-based systems. They focus on the surface-level features of the text like specific words or phrases.

Text Embedding: Text embedding, by contrast, captures deeper linguistic patterns and semantic meanings. It allows systems to understand context, nuances, and even the sentiment behind text, making it more efficient for tasks like information retrieval, classification, and conversational AI.

Relevance in This Lab

In this lab, you will use LangChain to extract and process text from a PDF document. You will then embed this text using advanced NLP techniques, allowing you to create a conversational AI that can understand and retrieve information based on the content of the PDF. This process exemplifies how embedding transforms raw text into a format suitable for semantic understanding and retrieval, enabling the AI to respond to queries with contextually relevant information.

The lab's highlight is the integration of these components into a ConversationalRetrievalChain, showcasing how text embeddings can be leveraged to enhance the capabilities of conversational interfaces. By the end of this lab, you will have hands-on experience with text extraction, embedding, and utilizing these embeddings in a conversational context, highlighting the practical applications and effectiveness of modern NLP techniques in creating intelligent systems.

## Lab Instructions:
Implement get_pdf_text: This function should take a list of PDF file paths, extract text from each PDF, and return the combined text.

Implement get_text_chunks: Split the raw text into smaller chunks for efficient processing. Use CharacterTextSplitter or a similar mechanism to divide the text.

Implement get_vector_store: Convert the text chunks into embeddings using OpenAIEmbeddings and store these in a FAISS vector store for fast retrieval.

Implement get_conversation_chain: Create a conversation chain using ChatOpenAI for the chat model and the previously created vector store for data retrieval.

### Completion Criteria:
- Complete the 'TODO' methods as outlined in the lab documents.
- To successfully finish the lab, pass all tests in the tests/test_lab.py file.
- Run the application or tests using your editor, or use the provided run scripts.

### Running Scripts
To run a shell script in Unix/Linux/Mac environments, use the command:

```
./run_script.sh
````

Make sure the script has execute permissions. If not, run chmod +x run_script.sh first.

To run a batch script in Windows, simply execute:

```
run_script.bat
```

## Support
- For technical issues or detailed understanding, consult the [LangChain documentation](https://python.langchain.com/docs/get_started/introduction).
- For additional guidance or inquiries, reach out to the lab coordinator or use the dedicated help forum.

## License
This lab activity and its materials are provided under the MIT License. They are intended for educational and training purposes.
