# Import necessary libraries
import os
from typing import List

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

# ------------------------------------------------------------------------------
# TODO Functions - Implement the logic as per instructions
# ------------------------------------------------------------------------------


def get_pdf_text(pdf_path: str) -> str:
    """
    TODO: Implement this method to extract text from the provided PDF file path.
    The text from the PDF file should be extracted and returned as a string.

    Instructions:
    - Raise a FileNotFoundError if the PDF file is not found at the given path.
    - Use PyPDF2 to open and read the PDF file from the given path.
    - Iterate over each page in the PDF.
    - Extract text from each page and concatenate it into a string.
    - Check if the page text is not empty before appending it to the string.
    - Append a newline character after each page to separate them.
    - Return the extracted text.

    Parameters:
    pdf_path (string): Path to the PDF document.

    Returns:
    string: Text extracted from the PDF document.
    """
    # Implement your code here
    raise NotImplementedError("This function is not yet implemented.")


def get_text_chunks(raw_text: str) -> List[str]:
    """
    TODO: Implement this method to split the raw text into smaller chunks for efficient processing.
    Use CharacterTextSplitter or a similar mechanism to divide the text.

    Instructions:
    - Use CharacterTextSplitter to split the raw_text into chunks.
    - Configure with appropriate parameters like separator, chunk_size, chunk_overlap and length_function.
        - separator: Split the text by newline characters
        - chunk_size: Split the text into chunks of [number] characters
        - chunk_overlap: Keep an overlap of [number] characters between chunks
        - length_function: Use the len function to calculate the length of each chunk
    - Split the text into chunks by calling the split_text method of the splitter.
    - Return the list of text chunks.

    Parameters:
    raw_text (string): The raw text to be split into chunks.

    Returns:
    list of strings: The text split into manageable chunks.
    """
    # Implement your code here
    raise NotImplementedError("This function is not yet implemented.")


def get_vector_store(text_chunks: List[str]) -> FAISS:
    """
    TODO: Implement this method to convert the text chunks into embeddings and store these in a FAISS vector store.

    Instructions:
    - Initialize OpenAIEmbeddings to convert text chunks into embeddings.
    - Use FAISS to create a vector store from these embeddings.
    - Configure with appropriate parameters like texts and embedding.
        - texts: The list of text chunks to be converted into embeddings.
        - embedding: The OpenAIEmbeddings instance to be used for converting text into embeddings.
    - Return the FAISS vector store containing the embeddings.

    Parameters:
    text_chunks (list of strings): Text chunks to be converted into vector embeddings.

    Returns:
    FAISS: A FAISS vector store containing the embeddings of the text chunks.
    """
    # Implement your code here
    raise NotImplementedError("This function is not yet implemented.")


def get_conversation_chain(vector_store: FAISS) -> BaseConversationalRetrievalChain:
    """
    Initializes and returns a ConversationalRetrievalChain. This chain integrates a language model
    and a vector store for handling conversational queries and retrieving relevant information.

    - Initializes a ChatOpenAI model with a fixed response pattern (temperature=0).
    - Sets up a ConversationBufferMemory to store and manage conversation history.
    - The ConversationalRetrievalChain combines these components, using the vector store for
      efficient information retrieval based on user queries.

    Parameters:
    vector_store (FAISS): A vector store containing text embeddings for information retrieval.

    Returns:
    BaseConversationalRetrievalChain: A conversational chain ready for handling queries.
    """
    llm = ChatOpenAI(
        temperature=0  # Feel free to change the temperature setting. Closer to 0 is more deterministic while closer to 1 is more random.
    )

    # Set up memory buffer for the conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Create a conversation chain that uses the language model and vector store for retrieving information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # language model for generating responses
        retriever=vector_store.as_retriever(),  # vector store for fetching relevant information
        memory=memory,  # memory buffer for storing conversation history
    )

    # Return the conversation chain
    return conversation_chain


# ------------------------------------------------------------------------------
# Starter Code - TOUCH AT YOUR OWN RISK!
# ------------------------------------------------------------------------------


def clear_screen():
    """
    Clears the terminal screen.

    This function uses a system call to clear the terminal screen. The command
    differs depending on the operating system: 'cls' for Windows ('nt') and 'clear'
    for Unix/Linux.
    """
    os.system("cls" if os.name == "nt" else "clear")


# Main application function
def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate two levels up to the root of the project
    root_dir = os.path.join(script_dir, "..", "..")

    # Construct the path to the resources directory
    folder_path = os.path.join(root_dir, "resources")

    # Construct the full path to the PDF file
    pdf_path = os.path.join(folder_path, "langchain.pdf")

    # Extract text from PDFs
    raw_text = get_pdf_text(pdf_path)

    # Split the text into chunks
    text_chunks = get_text_chunks(raw_text)

    # Create a vector store from the text chunks
    vector_store = get_vector_store(text_chunks)

    # Create a conversation chain
    conversation_chain = get_conversation_chain(vector_store)

    # Example conversation loop
    while True:
        clear_screen()
        user_input = input("Ask a question about the PDF (or type 'exit' to stop): ")
        if user_input.lower() == "exit":
            break
        response = conversation_chain({"question": user_input})
        print(f"\nResponse: {response.get('answer', 'No response generated.')}")
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
