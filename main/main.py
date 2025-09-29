import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sqlalchemy import Null

load_dotenv()
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2  # ✅ keep deterministic for schema-based parsing
)



import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

index_path = "faiss_index"

if os.path.exists(index_path):
    vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader(file_path="D:\\NameKart\\React + SpringBoot\\ChatBot\\Resource\\B.E.CSE (1).pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vector_db = FAISS.from_documents(splits, embedding_model)
    vector_db.save_local(index_path)

retriever = vector_db.as_retriever(search_kwargs={"k":800})



from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser

extract_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a subject extractor.
        Extract SUBJECT_CODE and SUBJECT_GRADE from input text.
        Convert grades using mapping: {grade_converter}.
        Return ONLY JSON with key 'answer' with its grade and its subject code."""
    ),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

extract_chain = extract_prompt | llm_model | JsonOutputParser()


from langchain_core.runnables import RunnableParallel
from operator import itemgetter


calc_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a GPA calculator.
        Rules:
        1. Perform the calculation exactly using the provided formula.
        2. answer should be 0 =< answer >= 10, If not multiple the answer be 10.
        3. give the answer in x.xx two decimal values. 
        4. Return ONLY JSON:
           {{ "answer": <numeric_result>,
              "details": {{"SUBJECT_CODE": "grade*credit"}} }}
        5. On error, return:
           {{ "error": "<missing subject codes or issue>" }}"""
    ),
    HumanMessagePromptTemplate.from_template(
        "Subjects with grades: {question}\n"
        "Reference credits: {context}\n"
        "Formula: {formula}"
    )
])

formula = "[((SUBJECT_GRADE/10)*SUBJECT_CREDIT)+...+n]/TOTAL_CREDIT_USED"

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


import json

def run_pipeline(user_input: str):
    grade_converter = {
    "O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C+": 5, "C": 4
    }
    extracted = extract_chain.invoke({"user_input": user_input, "grade_converter": grade_converter})
    retrieved = retriever.invoke(json.dumps(extracted))
    result = calc_chain.invoke({
        "question": extracted,
        "context": retrieved,
        "formula": formula
    })
    return result


if __name__ == "__main__":
    user_input = """Tamil - II	GE3252	B+	70	61
English - II	HS3252 	A	80	71
Maths - II	MA3251	B+	70	61
Physics - II	PH3256	A	80	71
B.E(EEE)	BE3251	A	80	71
Engineering Graphics 	GE3251	A	80	71
Program in C	CS3251	A	80	71
Lab-English	GE3272	O	100	91
Lab-Epl	GE3271	O	100	91
Lab-Programming in C	CS3271	O	100	91"""
    output = run_pipeline(user_input)
    print(output)





