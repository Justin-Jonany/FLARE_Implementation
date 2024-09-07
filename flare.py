# format_docs
# def flare(question, retriever, openai_api_key, openai_model='gpt-4o-mini'):
#   '''
#   Uses a advanced RAG technique called FLARE to answer the question. It's an implementation 
#   of the paper: "Active Retrieval Augmented Generation" Jiang ZB and fellow scientists in Octover
#   2023.

#   Args:
#     question: The question to be answered.
#     retriever: langchain retriever to be used.
#     openai_api_key: The OPENAI API key to be used.
#   Returns:
#     The answer to the question.

#   '''
#   client = OpenAI(api_key=openai_api_key)

#   # getting the first output, normal rag
#   ## get context
#   context = retriever.get_relevant_documents(question)

#   ## constructing message
#   message = [
#         {"role": "system", "content": """You are a book expert that answers questions about books. Use the following pieces of retrieved context to answer the question."""},
#         {"role": "user", "content": f"""
#         Context: {format_docs(context)}
#         Question: {question}
#         Answer:
#         """}
#   ]
#   ## answer the question
#   answer = client.chat.completions.create(
#       model=openai_model,
#       messages=message,
#       logprobs=True,
#   )

#   # annotate the question
#   annotated_answer = uncertain_marker(annotated_combiner(annotater(sequential_combine(combine_token_to_word(answer), 5, np.mean), tolerance= -0.4), np.mean))

#   # constructing the questions for the uncertained answers
#   message += [
#       {"role": "assistant", "content": answer.choices[0].message.content},
#       {"role": "system", "content": f"""Now, I have marked the answer to where you are uncertain with the phrases. For every, phrases in between
#         [uncertain] [/uncertain], please construct a question that will answer each uncertain phrase and mark it as [Search(question)].

#         This question is going to be used independently to get get relevant texts from a vector database
#         It's critical that the question includes the object and subject of the phrase
#         It's critical that the question has context about the annswer


#         First example:
#         user: What is meaning of the colors in the flag of Ghana?
#         assistant: Red is for the blood of martyrs, green for forests, and gold for mineral wealth.
#         user: Here's the annotated version: ([uncertain] Red [/uncertain]) is for the blood of martyrs, ([uncertain] green for forests [/uncertain]), and gold for mineral wealth.
#         assistant: [Search(is red a color in the flag of Ghana?)] is for the blood of martyrs, [Search(is green a color in the flag of Ghana? If so, what does it symbolize?)], and gold for mineral wealth.

#         Second example:
#         user: Give me a very short summary of Joe Biden's journey becoming the president!
#         assistant: Joe Biden announced his candidacy for the 2020 presidential election on August 18, 2019. His campaign focused on issues such as restoring the 'soul of America', expanding healthcare access, and addressing climate change.
#         user: Here's an annotated version: Joe Biden announced his candidacy for the 2020 presidential election on ([uncertain] August 18, 2019 [/uncertain]). His campaign focused on issues such as restoring the 'soul of America', expanding healthcare access, and addressing climate change.
#         assistant: Joe Biden announced his candidacy for the 2020 presidential election on [Search(When did Joe Biden announce his candidancy for the 2020 presidential election?)].  His campaign focused on issues such as restoring the 'soul of America', expanding healthcare access, and addressing climate change.
#         """},
#       {"role": "user", "content": f"Here's the annotated version: {annotated_answer.choices[0].message.content}"},
#   ]
#   questions_construction = client.chat.completions.create(
#       model=openai_model,
#       messages=message,
#   )

#   # extracting the questions
#   message += [
#       {"role": "assistant", "content": questions_construction.choices[0].message.content},
#       {'role': "user", "content": """Now for all the questions marked as [Search(question)], please extract them in a python dictionary format:
#       {
#       "1": "question 1",
#       "2": "question 2",
#       ...
#       "n": "question n"
#       }

#       it is critical to only output the dictionary and nothing else.
#       """}
#   ]
#   questions = client.chat.completions.create(
#       model=openai_model,
#       messages=message,
#   )
#   questions_dict = ast.literal_eval(questions.choices[0].message.content)

#   # message to answer the questions one by one
#   new_message = [
#       {"role": "system", "content": f"""You are a book expert that answers questions about books.
#       Question: {question}
#       Context: called the RAG
#       Answer: {answer.choices[0].message.content}

#       Now, I have marked the answer to where you are uncertain with the phrases. For every, phrases in between
#       [uncertain] [/uncertain], please construct a question that will answer each uncertain phrase and mark it as [Search(question)]

#       Annotated Version: {annotated_answer.choices[0].message.content}
#       Constructed Questions: {questions_construction.choices[0].message.content}
#       """},
#   ]

#   # answering each of the questions one by one
#   constructed_question_answer = {}
#   for i in range(1, len(questions_dict) + 1):
#     question_temp = questions_dict[str(i)]

#     ## getting the context for the question
#     context = retriever.get_relevant_documents(question_temp)

#     ## constructing the question and context message
#     question_string = f"""Use the following pieces of retrieved context to answer the question.
#     Question: {question_temp}
#     Context: {format_docs(context)}
#     Answer:
#     """.format(question=question_temp, context=context)
#     question_message = new_message + [{"role": "user", "content": question_string}]

#     ## answering the question
#     question_answer = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=question_message,
#     )
#     constructed_question_answer[str(i)] = [question_temp, question_answer.choices[0].message.content]

#   # reconstructing to get the final answer
#   question_answer = ""
#   for i in range(1, len(questions_dict) + 1):
#     question_answer += f"""Question: {questions_dict[str(i)]}\nAnswer: {constructed_question_answer[str(i)][1]}\n"""

#   reconstructing = f"""
#   Here are the questions and their answers:
#   {question_answer}
#   Now with answers to those questions, fix the original answer into a paragraph:
#   """
#   reconstructing_message = new_message + [{"role": "user", "content": reconstructing}]
#   reconstructed_answer = client.chat.completions.create(
#       model=openai_model,
#       messages=reconstructing_message,
#   )

#   # printing some information
#   print(f"""Question: {question}\nAnswer: {answer.choices[0].message.content}\nAnnotated Answer: {annotated_answer.choices[0].message.content}\nReconstructed Answer: {reconstructed_answer.choices[0].message.content}""")
#   return answer, annotated_answer, reconstructed_answer


