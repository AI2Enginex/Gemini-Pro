import os  # Importing the 'os' module for operating system related functionalities
from PyPDF2 import PdfReader  # Importing PdfReader class from PyPDF2 library for reading PDF files
from langchain.chains.question_answering import load_qa_chain  # Importing the function load_qa_chain from langchain library for question answering
import google.generativeai as genai  # Importing the Google Generative AI module from the google package
from google.generativeai import GenerationConfig  # Importing GenerationConfig class from google.generativeai module
from langchain_google_genai import ChatGoogleGenerativeAI  # Importing ChatGoogleGenerativeAI class from langchain_google_genai module
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter class from langchain module for text splitting
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Importing GoogleGenerativeAIEmbeddings class from langchain_google_genai module
from langchain.vectorstores import FAISS  # Importing FAISS class from langchain module for vector storage
from langchain.prompts import PromptTemplate  # Importing PromptTemplate class from langchain module for prompts

# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key = os.environ['GOOGLE_API_KEY'] = 'AIzaSyDaBLM8TjUihg_SaFO3qc3Lms7ZhY-4dJQ'

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)



class GeminiModel:
    def __init__(self):
        
        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-pro')
        # Creating a GenerationConfig object with specific configuration parameters
        self.generation_config = GenerationConfig(
            temperature=0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

class GeminiChatModel(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)
        # Starting a chat using the model inherited from GeminiModel
        self.chat = self.model.start_chat()

class ChatGoogleGENAI:
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.model = ChatGoogleGenerativeAI(
            model="gemini-pro",  # Using the 'gemini-pro' model
            temperature=0.3,  # Setting temperature for generation
            google_api_key=api_key, # Passing the Google API key
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192, 
        )


class EmbeddingModel:
    def __init__(self, model_name):
        # Initializing GoogleGenerativeAIEmbeddings object with the specified model name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        
class GenerateContext(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)

    def generate_response(self, query):
        try:
            # Generating response content based on the query using the inherited model
            return [response for response in self.model.generate_content(query)]
        except Exception as e:
            return e

class ReadFile:
    @classmethod
    def read_file_text(cls, folder_name=None):
        try:
            text = ""
            with open(folder_name, 'rb') as file:
                reader = PdfReader(file)
                # Extracting text from each page of the PDF file and concatenating
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
        except Exception as e:
            return e
        
    def read_file_and_store_elements(filename):
        try:
            text = ''
            with open(filename, "r") as file:
                for line in file:
                    # Remove leading/trailing whitespace and newline characters
                    line = line.strip()
                    # Add the line to the list of elements
                    text += line
            return text
        except Exception as e:
            return e

class TextChunks:
    @classmethod
    def get_text_chunks(cls, separator=None, chunksize=None, overlap=None, text=None):
        try:
            # Splitting text into chunks based on specified parameters
            text_splitter = RecursiveCharacterTextSplitter(separators=separator, chunk_size=chunksize, chunk_overlap=overlap)
            return text_splitter.split_text(text)
        except Exception as e:
            return e
        
class Vectors:
    @classmethod
    def generate_vectors(cls, chunks, model):
        try:
            # Generating vectors from text chunks using specified model
            embeddings = EmbeddingModel(model_name=model)
            return FAISS.from_texts(chunks, embedding=embeddings.embeddings)
        except Exception as e:
            return e

class ChatModel(GeminiChatModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiChatModel)

    def prompt_to_model(self):
        # Setting up the prompt for the chat bot
        prompt = """
        you are a chat bot.
        users will ask you Questions.
        your job is to provide the answer in the most meaningful manner.
        try to elaborate the answer as much as you can.
        
        
        Answer:
        """ 
        # Creating a PromptTemplate object with the defined template
        return  prompt
    
    def generate_response(self, user_input):
        try:
            # Sending user input along with the prompt to the chat bot and receiving the response
            return self.chat.send_message(self.prompt_to_model() + user_input, stream=True)
        except Exception as e:
            return e
        
    def start_chat(self,user_input):
        try:
            # Starting the chat and yielding responses
            for response in self.generate_response(user_input):
                yield response.text.replace('*', '')  # Removing any asterisks from the response
        except Exception as e:
            return e

class DocumentSummarization(ChatGoogleGENAI):
    def __init__(self, dir_name):
        super().__init__()  # Calling the constructor of the superclass (ChatGoogleGENAI)
        # Reading text from the specified directory and assigning it to self.file
        self.file = ReadFile().read_file_text(dir_name)
    
    def get_chunks(self, separator=None, chunksize=None, overlap=None):
        try:
            # Getting text chunks from the file using TextChunks class
            return TextChunks().get_text_chunks(separator=separator, chunksize=chunksize, overlap=overlap, text=self.file)
        except Exception as e:
            return e
        
    def embeddings(self, separator=None, chunksize=None, overlap=None, model=None):
        try:
            # Generating vectors from text chunks using Vectors class
            return Vectors().generate_vectors(chunks=self.get_chunks(separator, chunksize, overlap), model=model)
        except Exception as e:
            return e
    
    def summarisation_chains(self, chaintype):
        try:
            # Defining a prompt template for summarization
            # the "context" variable provides the background or content from which the summary is derived (ex. the pdf document)
            # the "question" variable prompts the user to focus on specific aspects or details within that context
            prompt_template = """
            You are given a pdf document. 
            Your job is generate a consice summary of the given document in 15 points.
            try to summarize each line in the document.
            Context:\n {context}?\n
            Question: \n{question}\n
        
            Answer:
            """
            # Creating a PromptTemplate object with the defined template
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            # Loading the question-answering chain with the specified chain type and prompt
            return load_qa_chain(self.model, chain_type=chaintype, prompt=prompt)
        except Exception as e:
            return e
    
    def main(self, separator=None, chunksize=None, overlap=None, model=None, type=None, user_question=None):
        try:
            # Retrieving the summarization chain
            chain = self.summarisation_chains(chaintype=type)
            # Generating embeddings for the text
            db = self.embeddings(separator, chunksize, overlap, model)
            # Performing similarity search and obtaining documents
            docs = db.similarity_search(user_question)
            # Generating response using the chain and user question
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response
        except Exception as e:
            return e


class DocumentQuestionAnswering(ChatGoogleGENAI):
    def __init__(self, dir_name):
        super().__init__()  # Calling the constructor of the superclass (ChatGoogleGENAI)
        # Reading text from the specified directory and assigning it to self.file
        self.file = ReadFile().read_file_text(dir_name)
    
    def get_chunks(self, separator=None, chunksize=None, overlap=None):
        try:
            # Getting text chunks from the file using TextChunks class
            return TextChunks().get_text_chunks(separator=separator, chunksize=chunksize, overlap=overlap, text=self.file)
        except Exception as e:
            return e
        
    def embeddings(self, separator=None, chunksize=None, overlap=None, model=None):
        try:
            # Generating vectors from text chunks using Vectors class
            return Vectors().generate_vectors(chunks=self.get_chunks(separator, chunksize, overlap), model=model)
        except Exception as e:
            return e
    
    def conversational_chains(self, chaintype):
        try:
            # Defining a prompt template for conversational question-answering
            # the "context" variable provides the background or content from which the summary is derived (ex. the pdf document)
            # the "question" variable prompts the user to focus on specific aspects or details within that context
            prompt_template = """
            Answer the question as detailed as possible from the provided context.
            make sure to provide all the details.
            do not provide answes in points.
            if the answer is not provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n
        
            Answer:
            """
            # Creating a PromptTemplate object with the defined template
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            # Loading the question-answering chain with the specified chain type and prompt
            return load_qa_chain(self.model, chain_type=chaintype, prompt=prompt)
        except Exception as e:
            return e
    
    def main(self, separator=None, chunksize=None, overlap=None, model=None, type=None, user_question=None):
        try:
            # Retrieving the conversational question-answering chain
            chain = self.conversational_chains(chaintype=type)
            # Generating embeddings for the text
            db = self.embeddings(separator, chunksize, overlap, model)
            # Performing similarity search and obtaining documents
            docs = db.similarity_search(user_question)
            # Generating response using the chain and user question
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response
        except Exception as e:
            return e

if __name__ == '__main__':

   
   pass