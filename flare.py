from openai import OpenAI
import ast
from .flare_helper import *
from langchain_core.documents import Document

class Fake_Retriever:
  '''
  Instead of a vector database. FLARE can be used for regular zero-shot
  prompting. If the context is relatively not too long, it may be better
  to just give the whole context to the LLM at all cases.

  The class Fake_Retriever returns the whole context everytime a retriever
  is called.
  '''
  def __init__(self, data):
    self.data = Document(page_content=data)
  def get_relevant_documents(self, query):
    return [self.data]
  

def flare(question, retriever, openai_api_key, openai_model='gpt-4o-mini', tolerance=-0.4, verbose=True):
    '''
    Uses a advanced RAG technique called FLARE to answer the question. It's an implementation
    of the paper: "Active Retrieval Augmented Generation" Jiang ZB and fellow scientists in October
    2023.

    Args:
      question: The question to be answered.
      retriever: langchain retriever to be used.
      openai_api_key: The OPENAI API key to be used.
      openai_model: OpenAI Model to use
      tolerance: The tolerance of logprobs to be marked as uncertain
    Returns:
      A 3-item tuple, which are all OpenAI objects. The items are the original answer,
      annotated answer, and finally the improved answer.

    '''
    client = OpenAI(api_key=openai_api_key)

    # getting the first output, normal rag
    # get context
    context = retriever.get_relevant_documents(question)

    # constructing message
    message = [
        {"role": "system", "content": """You are a book expert that answers questions about books. Use the following pieces of retrieved context to answer the question."""},
        {"role": "user", "content": f"""
        Context: {format_docs(context)}
        Question: {question}
        Answer:
        """}
    ]

    if verbose: print(f'Acquiring answer with traditional RAG...')
    # answer the question
    answer = client.chat.completions.create(
        model=openai_model,
        messages=message,
        logprobs=True,
    )

    if verbose: print(f'Finding uncertain tokens, and annotating the answer...')
    # annotate the question
    annotated_answer = uncertain_marker(annotated_combiner(annotater(sequential_combine(
        combine_token_to_word(answer), 5, np.mean), tolerance=tolerance), np.mean))

    # constructing the questions for the uncertained answers
    message += [
        {"role": "assistant", "content": answer.choices[0].message.content},
        {"role": "system", "content": f"""Now, I have marked the answer to where you are uncertain with the phrases. For every, phrases in between
        [uncertain] [/uncertain], please construct a question that will answer each uncertain phrase and mark it as [Search(question)].

        This question is going to be used independently to get get relevant texts from a vector database
        It's critical that the question includes the object and subject of the phrase
        It's critical that the question has context about the annswer


        First example:
        user: What is meaning of the colors in the flag of Ghana?
        assistant: Red is for the blood of martyrs, green for forests, and gold for mineral wealth.
        user: Here's the annotated version: ([uncertain] Red [/uncertain]) is for the blood of martyrs, ([uncertain] green for forests [/uncertain]), and gold for mineral wealth.
        assistant: [Search(is red a color in the flag of Ghana?)] is for the blood of martyrs, [Search(is green a color in the flag of Ghana? If so, what does it symbolize?)], and gold for mineral wealth.

        Second example:
        user: Give me a very short summary of Joe Biden's journey becoming the president!
        assistant: Joe Biden announced his candidacy for the 2020 presidential election on August 18, 2019. His campaign focused on issues such as restoring the 'soul of America', expanding healthcare access, and addressing climate change.
        user: Here's an annotated version: Joe Biden announced his candidacy for the 2020 presidential election on ([uncertain] August 18, 2019 [/uncertain]). His campaign focused on issues such as restoring the 'soul of America', expanding healthcare access, and addressing climate change.
        assistant: Joe Biden announced his candidacy for the 2020 presidential election on [Search(When did Joe Biden announce his candidancy for the 2020 presidential election?)].  His campaign focused on issues such as restoring the 'soul of America', expanding healthcare access, and addressing climate change.
        """},
        {"role": "user",
            "content": f"Here's the annotated version: {annotated_answer.choices[0].message.content}"},
    ]

    if verbose: print(f'Constructing questions for the annotated tokens...')
    questions_construction = client.chat.completions.create(
        model=openai_model,
        messages=message,
    )

    # extracting the questions
    message += [
        {"role": "assistant",
            "content": questions_construction.choices[0].message.content},
        {'role': "user", "content": """Now for all the questions marked as [Search(question)], please extract them in a python dictionary format:
        {
        "1": "question 1",
        "2": "question 2",
        ...
        "n": "question n"
        }

        It is critical to only output the dictionary and nothing else.
        It is critical to not output it in a markdown format.
        It is critical that the first character of the output starts with an open curly bracket '{'
      """}
    ]

    if verbose: print(f'Extracting constructed questions...')
    questions = client.chat.completions.create(
        model=openai_model,
        messages=message,
    )
    retry_count= 1
    try:
        questions_dict = ast.literal_eval(questions.choices[0].message.content)
    except:
        while retry_count <= 3:
            if verbose: print(f"Couldn't convert to dictionary, attempting to fix the dictionary. Retry count: {retry_count}")
            questions = client.chat.completions.create(
                model=openai_model,
                messages=message,
            )
            try:
                questions_dict = ast.literal_eval(questions.choices[0].message.content)
                break
            except:
                retry_count += 1
                continue
        else:
            print(f'FLARE Failed, try to call the function again.')
            return

    # message to answer the questions one by one
    new_message = [
        {"role": "system", "content": f"""You are a book expert that answers questions about books.
      Question: {question}
      Context: called the RAG
      Original Answer: {answer.choices[0].message.content}

      Now, I have marked the answer to where you are uncertain with the phrases. For every, phrases in between
      [uncertain] [/uncertain], please construct a question that will answer each uncertain phrase and mark it as [Search(question)]

      Annotated Answer: {annotated_answer.choices[0].message.content}
      Constructed Questions Answer: {questions_construction.choices[0].message.content}
      """},
    ]

    if verbose: print(f'Answering each questions...')
    # answering each of the questions one by one
    constructed_question_answer = {}
    for i in range(1, len(questions_dict) + 1):
        question_temp = questions_dict[str(i)]

        # getting the context for the question
        context = retriever.get_relevant_documents(question_temp)

        # constructing the question and context message
        question_string = f"""Use the following pieces of retrieved context to answer the question.
        Question: {question}
        Context: {format_docs(context)}
        Answer:
        """
        question_message = new_message + \
            [{"role": "user", "content": question_string}]

        # answering the question
        question_answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=question_message,
        )
        constructed_question_answer[str(i)] = [
            question_temp, question_answer.choices[0].message.content]

    # reconstructing to get the final answer
    question_answer = ""
    for i in range(1, len(questions_dict) + 1):
        question_answer += f"""Question: {questions_dict[str(i)]}\nAnswer: {constructed_question_answer[str(i)][1]}\n"""

    reconstructing = f"""
    Here are the questions and their answers:
    {question_answer}
    Now with answers to those questions and the original question: {question}, improve the original answer without changing the format of the answer.
    
    Notes:
    It's critical to just output the final answer.
    It's critical to not output the annotated answer.
    It's critical to not output constructed questions answer.

    Final Answer:
    """
    reconstructing_message = new_message + \
        [{"role": "user", "content": reconstructing}]
    if verbose: print(f'Reconstructing the final answer...')
    reconstructed_answer = client.chat.completions.create(
        model=openai_model,
        messages=reconstructing_message,
    )

    # printing some information
    # It prints the question construction for the annotated answer because it's clearer to make the annotations as the questions itself
    return answer, questions_construction, reconstructed_answer
