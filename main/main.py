import os
from dotenv import load_dotenv
from typing import Optional
from datetime import date

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from operator import itemgetter

# -----------------------------
# 1. Load ENV + LLM
# -----------------------------
load_dotenv()
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2  # ✅ keep deterministic for schema-based parsing
)

# -----------------------------
# 2. Load & Index Data (PDF)
# -----------------------------
pdf_loader = PyPDFLoader("D:\\NameKart\\React + SpringBoot\\ChatBot\\Resource\\B.E.CSE (1).pdf")
docs = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_db = FAISS.from_documents(splits, embedding_model)

retriever = vector_db.as_retriever(search_kwargs={"k":10})  # ✅ reduced from 800 to 10

# -----------------------------
# 3. Subject Extractor (Grades → Numeric)
# -----------------------------
grade_converter = {
    "O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C+": 5, "C": 4
}

extract_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a subject extractor.
        Extract SUBJECT_CODE and SUBJECT_GRADE from input text.
        Convert grades using mapping: {grade_converter}.
        Return ONLY JSON with key 'answer'."""
    ),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

extract_chain = extract_prompt | llm_model | JsonOutputParser()

# -----------------------------
# 4. GPA Calculator Prompt
# -----------------------------
formula = "((((SUBJECT_GRADE)/10)*SUBJECT_CREDIT)+...+n)/TOTAL_CREDIT_USED"

calc_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a GPA calculator.
        Rules:
        1. Perform the calculation exactly using the provided formula.
        2. Return ONLY JSON:
           {{ "answer": <numeric_result>,
              "details": {{"SUBJECT_CODE": "grade*credit"}} }}
        3. On error, return:
           {{ "error": "<missing subject codes or issue>" }}"""
    ),
    HumanMessagePromptTemplate.from_template(
        "Subjects with grades: {question}\n"
        "Reference credits: {context}\n"
        "Formula: {formula}"
    )
])

calc_chain = (
    RunnableParallel({
        "question": itemgetter("question"),
        "context": itemgetter("context"),
        "formula": itemgetter("formula")
    })
    | calc_prompt
    | llm_model
    | JsonOutputParser()
)

# -----------------------------
# 5. Pipeline Execution
# -----------------------------
def run_pipeline(user_input: str):
    # Step 1: Extract subject codes + grades
    extracted = extract_chain.invoke({"user_input": user_input, "grade_converter": grade_converter})

    # Step 2: Retrieve credits info in parallel
    retrieved = retriever.invoke(extracted)

    # Step 3: GPA Calculation
    result = calc_chain.invoke({
        "question": extracted,
        "context": retrieved,
        "formula": formula
    })
    return result


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    user_input = """5SEM CB3491 Cryptography And Cyber Security B+
                    5SEM CCS346 Exploratory Data Analysis (T & P) A
                    5SEM CS3501 Compiler Design (T&P) A+"""
    output = run_pipeline(user_input)
    print(output)
