from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    '''
    formats the list of documents into a string

    Args:
      docs: list of documents

    Returns:
      string of documents
    '''
    return "\n\n".join(doc.page_content for doc in docs)


def traditional_rag(question, retriever, llm, verbose=True):
    '''
    uses traditional RAG to answer a question

    Args:
      question: question to answer
      retriever: langchain retriever to use
      llm: langchain language model to use
      verbose: whether to print the prompt
    Returns:
      a string that's the answer to the question
    '''
    if verbose:
        print('Traditional RAG')

    prompt = prompt = ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[
                                         'context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
    context = retriever.invoke(
        'Who are the characters mentioned in the chapter titled "The Gold Lender of Babylon"')

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return result
