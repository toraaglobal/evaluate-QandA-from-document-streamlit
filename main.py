import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import retrieval_qa
from langchain.evaluation.qa import QAEvalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS



def generate_response(
        uploaded_file,
        openai_api_key,
        query_text,
        response_test
):
    # format uploaded file 
    documents = [uploaded_file.read().decode("utf-8")]

    # break it in small chunk 
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    texts = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )

    ## create a vector database 
    db = FAISS.from_documents(
        texts,
        embeddings
    )

    ## create retrival interface 
    retrival = db.as_retrieval_interface()

    # create a real QA dictionary 
    real_qa =[
        {
            "question": query_text,
            "answer": response_test
        }
    ]

    ## reqular QA chain 
    qachain = retrieval_qa.RetrievalQAChain(
        llm = OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retrival=retrival,
        input_key="question",
    )


    ## prediction 
    prediction = qachain.predict(real_qa)

    ## create an evaluation chain 
    eval_chain = QAEvalChain.from_llm(
        llm = OpenAI(openai_api_key=openai_api_key),
    )

    ## have it grade itself 
    graded_output = eval_chain.evaluate(
        real_qa,
        prediction,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )

    response = {
        "prediction": prediction,
        "graded_output": graded_output
    }

    return response

## page configuration
st.set_page_config(
    page_title="Evaluate a RAG App",
    page_icon="🧠",
    layout="centered",
)

st.title("Evaluate a RAG App")

with st.expander(
    "Evaluate the quality of a RAG APP",
    expanded=True
):
    st.write(
        """
        To evaluate the quality of a RAG app, we will
        ask it questions for which we already know the
        real answers.
        
        That way we can see if the app is producing
        the right answers or if it is hallucinating.
        """
    )

uploaded_file = st.file_uploader(
    "Upload the text file",
    type=["txt"]
)

query_text = st.text_input(
    "Enter the question to ask the RAG app",
    placeholder="Write your question here",
    disabled= not uploaded_file
)

response_text = st.text_input(
    "Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

result = []

with st.form(
    "myform",
    clear_on_submit=True
):
    openai_api_key = st.text_input(
        "Enter your OpenAI API key",type="password",disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button("Evaluate",disabled= not (uploaded_file and query_text))

    if submitted and openai_api_key.startwith("sk"):
        with st.spinner("Evaluating the RAG app"):
            response = generate_response(
                uploaded_file,
                openai_api_key,
                query_text,
                response_text
            )
            result.append(response)
            del openai_api_key

if len(result):
    st.write("Question")
    st.info(response["predictions"][0]["question"])
    st.write("Real answer")
    st.info(response["predictions"][0]["answer"])
    st.write("Answer provided by the AI App")
    st.info(response["predictions"][0]["result"])
    st.write("Therefore, the AI App answer was")
    st.info(response["graded_outputs"][0]["results"])

        


