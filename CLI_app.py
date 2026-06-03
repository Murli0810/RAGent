import os
from dotenv import load_dotenv
import numexpr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from datetime import datetime
from langchain_tavily import TavilySearch

#loading data from env file
load_dotenv()

#loading pdf file
loader = PyPDFLoader("ragent_test.pdf")
pages = loader.load()

#spliting text from pdf into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

#google's embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

#Initializing the vector store and retriever
vector_store = FAISS.from_documents(chunks, embeddings_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

#Google's chatting model
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

#RAG tool for pdf recognition
@tool
def RAG_agent(query: str) -> str:
    """Return an answer to the user's question based on the retrieved context from the document whenver user says to use the document, otherwise answer based on your general knowledge."""
    
    RAG_chat_template="""You are an elite AI technical mentor.
    Answer the user's question using ONLY the provided background document context.
    If the answer cannot be logically inferred from the context, rely on your general knowledge but state clearly that it isn't in the docs.

    Background Context Documents:
    {context}
    """
    prompt= ChatPromptTemplate.from_messages([
        ("system", RAG_chat_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain= prompt | model | StrOutputParser()

    def get_formatted_context(query_string):
        matched_docs= retriever.invoke(query_string)
        return "\n\n".join(doc.page_content for doc in matched_docs)
    
    try:
        formatted_context= get_formatted_context(query)
        response= chain.invoke({
            "input": query,
            "context": formatted_context,
            "history": []
        })
        return response
    except Exception as e:
        return f"\nAn error occurred inside execution chain: {e}"

#Tool to get current time and date
@tool
def get_current_time() -> str:
    """Returns the current system date and time. 
    Use this ONLY when the user's current message explicitly 
    asks for the time or date. Do NOT use for acknowledgments 
    like 'ok', 'thanks', 'sure', or general replies."""
    now= datetime.now()
    return now.strftime("%Y-%m-%d %H-%M-%S")

#Tool for calculation
@tool
def calculator(expression: str) -> str:
    """Returns the final output of the calculation that the user asked.
    Only use this tool when user have an appropriate expression.
    Else tell the user that the caculation is not possible with the suitable reason."""

    try:
        result= numexpr.evaluate(expression)
        return str(float(result))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

#Tool for WebSearch
tavily_tool = TavilySearch(
    max_results=5,
    description="""Search the internet for current or real-time information.
    Use this ONLY when the user asks about something that cannot be found
    in the document and requires up-to-date web information.
    Do NOT use for document-related questions."""
)

#LLM persona
SYSTEM_MESSAGE = SystemMessage(content="""You are a helpful assistant with access to tools.
Only call a tool when the user's current message explicitly requires it.
Otherwise answer with your own intelligence.
""")

#Binding tools to LLM
tools= [RAG_agent, get_current_time, calculator, tavily_tool]
model_with_tools= model.bind_tools(tools)

chat_history= []

while True:
    user_input= input("\nYou: ")

    if user_input.strip().lower() in ["exit","quit"]:
        print("Goodbye!")
        break

    if not user_input.strip():
        continue

    try:
        messages = [SYSTEM_MESSAGE] + chat_history[-10:] + [HumanMessage(content=user_input)]
        response= model_with_tools.invoke(messages)
        chat_history.append(HumanMessage(content=user_input))

        if response.tool_calls:

            for tool_call in response.tool_calls:
                tool_name= tool_call['name']
                tool_args= tool_call['args']

                print(f"👉 Executing tool: '{tool_name}' with arguments: {tool_args}")

                if tool_name == "RAG_agent":
                    tool_result= RAG_agent.invoke(tool_args)
                elif tool_name == "get_current_time":
                    tool_result= get_current_time.invoke(tool_args)
                elif tool_name == "calculator":
                    tool_result= calculator.invoke(tool_args)
                elif tool_name == "tavily_search":
                    tool_result= tavily_tool.invoke(tool_args)
                else:
                    tool_result= f"Error: Unrecognized tool '{tool_name}'"
                
                final_prompt = f"The user asked: '{user_input}'. The execution of the tool '{tool_name}' returned this data: '{tool_result}'. Formulate a clean, natural response to the user."
                final_response= model.invoke(final_prompt)

                if isinstance(final_response.content, list):
                    clean_text = "".join(
                    block['text'] if isinstance(block, dict) and 'text' in block
                    else block.text if hasattr(block, 'text')
                    else str(block)
                    for block in final_response.content
                )
                else:
                    clean_text = final_response.content

                print(f"\nRAGent: {clean_text}")
                chat_history.append(AIMessage(content=clean_text))

        else:
            if isinstance(response.content, list):
                clean_text = "".join(
                    block['text'] if isinstance(block, dict) and 'text' in block
                    else block.text if hasattr(block, 'text')
                    else str(block)
                    for block in response.content
                )
            else:
                clean_text = response.content

            print(f"\nRAGent: {clean_text}")
            chat_history.append(AIMessage(content=clean_text))

    except Exception as e:
        print(f"\nAn error occurred inside Agent loop: {e}")