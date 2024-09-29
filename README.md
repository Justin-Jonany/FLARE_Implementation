# **FLARE**
We know that when LLM's output tokens, they also output the probabilities associated with each token, as a result of the transformer model's outputting layer. However, what does these number mean? There have been several researchs arround this, some takes these numbers as how confident the LLMs are with its answer. This means that what if for every token that an LLM outputs with low probability, we assist the LLM to make it sure.

Introducing, the paper "[Active Retrieval Augmented Generation](https://arxiv.org/pdf/2305.06983)" by Jiang ZB and fellow researchers written on October 22, 2023, and the goal of this project will be to implement the paper and test it for various tasks. Although, at the the time of this project's starting date (August 6, 2024), it may seem relatively out-of-date, this technique is still relevant and efficient, and can be implemented for any future model that allows users to access the probabilities of each tokens.

All functions can be accessed in my github repository: [github.com/Justin-Jonany/FLARE_Implementation](https://github.com/Justin-Jonany/FLARE_Implementation)

## **Summary of the Paper (outputted by my implementation of FLARE)**
The paper explores the limitations of existing retrieval-augmented language models, which typically execute a single retrieval based solely on user input, leading to inaccurate and contextually irrelevant outputs. Recognizing the need for continual information gathering to enhance long-form text generation, the authors propose Forward-Looking Active Retrieval Augmented Generation (FLARE).

This method employs an iterative approach, where the model generates a provisional next sentence to identify low-confidence tokens and selectively retrieves relevant information to refine its output. FLARE operates through two main strategies: FLAREinstruct, which prompts the model to create retrieval queries as needed, and FLAREdirect, where the model’s generative output directly informs search queries. The authors empirically validate FLARE across multiple knowledge-intensive tasks and demonstrate its superior performance compared to traditional retrieval methods, emphasizing its effectiveness in enriching the generation process with accurate, contextually rich information that adapts to the evolving needs of text generation.

## **My Implementation**
I'm going to implement the FLARE Instruct version of the paper based on the paper's algorithm structure and will not be based-on the paper's source code. I will additionally use GPT-4o instead of GPT 3.5. The retriever can be any object that has a method called "get_relevant_documents". For this notebook, the retriever is Chroma with a langchain OpenAI Embedding.

The steps of the RAG are the following:
### 1. Use the LLM for traditional RAG or regular querying to answer the question
As the first output of the recursive steps, the question will be invoked on the LLM given a context retrieved from a basic RAG call to the retriever.

### 2. Check logits for each token from the LLM and annotate with a symbol any where the llm is not confident.
The next step is to check each token's logarithmic probabilities, and for a given tolerance, a function will automatically  modify the OpenAI Response object's logprob field with annotations. For every phrase where the it's unsure, it will be marked as the following: "[uncertain]...[/uncertain]". This is done with the assumption, that if the LLM outputs token with low probabilities, it implies that it's unsure about it, meaning it has a higher chance to be wrong.

### 3. Constructing questions to get a more confident answer for that token.
Now for all phrases that's marked as uncertain, the LLM will be reinvoked to turn the "[uncertain]...[/uncertain]" into "[search(question)]", where "question" would be the prompt to answer the uncertain phrase. After, these questions will be extracted into a dictionary, to ease the question-answering process.

### 4. Answer these questions
Now, having the questions, the LLM will be used to answer each question with a prompt of the following format: a shortened version of the steps so far, the annotated answer, the question, and finally the context retrieved. All answers will be saved a in dictionary as the value, and the question as the key.

### 5. Reconstruct the answer
Now having, all the answers, the LLM will be used the reconstruct the final answer.

## **Example Steps**
For the question: Why did Arkad believe that good luck follows opportunity?. My implementaton of FLARE would look like this:

### Step 1. Regular RAG Answer
In "The Richest Man in Babylon," Arkad, who is the richest man in Babylon, believes that good luck follows opportunity because he sees luck as a byproduct of one’s readiness to seize opportunities when they arise. He explains that many people often miss chances to become wealthy because they fail to recognize or act upon the opportunities presented to them.

Arkad suggests that those who are diligent, prepared, and willing to work toward their goals are more likely to encounter opportunities that lead to success. In essence, he emphasizes that luck is not merely random chance; rather, it is created through effort, willingness to take risks, and the ability to recognize and act on chances that life presents. This perspective encourages readers to be proactive and to seek out opportunities, rather than relying solely on chance for financial success.

### Step 2. Annotating uncertain tokens
In "The Richest Man in ([uncertain] Babylon," Arkad, who is the [/uncertain])  richest man in Babylon, believes that good luck follows opportunity because he sees luck as ([uncertain] a byproduct of one’s readiness [/uncertain])  to seize opportunities when they arise. ([uncertain] He explains that many people often miss chances to become wealthy because they fail to recognize or act upon the [/uncertain])  opportunities presented to them.

 ([uncertain]Arkad suggests that those who are diligent, prepared, and willing to work toward their goals [/uncertain])  are more likely to encounter ([uncertain] opportunities that lead to success. [/uncertain])  ([uncertain] In essence, he emphasizes that luck is not merely random chance; rather, it is created through effort, willingness to take [/uncertain])  risks, and the ability to ([uncertain] recognize and act on chances [/uncertain])  ([uncertain] that life presents. [/uncertain])  ([uncertain] This perspective encourages readers to be proactive and to seek out opportunities, rather than relying solely on chance for financial [/uncertain])  success.

 ### Step 3. Constructing questions to fix the low-confidence tokens
 In "The Richest Man in [Search(What is the setting of the book "The Richest Man in Babylon"?)], Arkad, who is the [Search(Who is Arkad in the context of the book?)] richest man in Babylon, believes that good luck follows opportunity because he sees luck as [Search(What does Arkad mean by "a byproduct of one’s readiness" in terms of seizing opportunities?)] to seize opportunities when they arise. [Search(Why do people miss chances to become wealthy according to Arkad?)] He explains that many people often miss chances to become wealthy because they fail to recognize or act upon the [Search(What type of opportunities does Arkad refer to in the book?)] opportunities presented to them.

[Search(What characteristics do diligent, prepared, and hardworking people exhibit according to Arkad?)] Arkad suggests that those who are diligent, prepared, and willing to work toward their goals are more likely to encounter [Search(What are examples of opportunities that lead to success in the book?)] opportunities that lead to success. [Search(What does Arkad say about the nature of luck?)] In essence, he emphasizes that luck is not merely random chance; rather, it is created through effort, willingness to take [Search(What type of risks does Arkad encourage people to take?)] risks, and the ability to [Search(How does one recognize and act on chances?)][Search(What does Arkad mean by "chances" in the context of life?)] recognize and act on chances [Search(What does Arkad mean by "that life presents"?)] that life presents. [Search(Why does Arkad advocate for a proactive approach to achieving financial success?)] This perspective encourages readers to be proactive and to seek out opportunities, rather than relying solely on chance for financial [Search(What does financial success mean in the context of the book?)] success.

### Step 4. Extract as dictionary
{'1': "What is the setting of the book 'The Richest Man in Babylon'?",
 '2': 'Who is Arkad in the context of the book?',
 '3': "What does Arkad mean by 'a byproduct of one’s readiness' in terms of seizing opportunities?",
 '4': 'Why do people miss chances to become wealthy according to Arkad?',
 '5': 'What type of opportunities does Arkad refer to in the book?',
 '6': 'What characteristics do diligent, prepared, and hardworking people exhibit according to Arkad?',
 '7': 'What are examples of opportunities that lead to success in the book?',
 '8': 'What does Arkad say about the nature of luck?',
 '9': 'What type of risks does Arkad encourage people to take?',
 '10': 'How does one recognize and act on chances?',
 '11': "What does Arkad mean by 'chances' in the context of life?",
 '12': "What does Arkad mean by 'that life presents'?",
 '13': 'Why does Arkad advocate for a proactive approach to achieving financial success?',
 '14': 'What does financial success mean in the context of the book?'}

 ### Step 5. Answering each question
 1. **Question**: What is the setting of the book 'The Richest Man in Babylon'?\
**Answer**: Arkad believed that good luck follows opportunity because he understood that wealth is not simply a matter of fate or luck; it is a result of taking action when opportunities present themselves. During his journey to accumulate wealth, he realized that many people fail to recognize the opportunities available to them or choose not to act on them. Instead of waiting for fortune to smile upon them, Arkad emphasized the importance of being proactive, diligent, and prepared to seize chances when they arise.\
According to Arkad, those who actively seek opportunities, educate themselves about how to manage and grow their wealth, and are willing to work and invest wisely are more likely to create their own "luck." This philosophy is encapsulated in his advice that a part of all one earns should be kept for oneself, which reflects the notion that individuals have the power to influence their financial destiny by being open to opportunity and ready to take positive steps toward wealth accumulation.

2. Question: Who is Arkad in the context of the book?\
Answer: Arkad believed that good luck follows opportunity because he understood that those who are prepared and willing to act upon the opportunities that life presents are more likely to find success. Essentially, luck, in Arkad's view, is not a mere coincidence; instead, it is a result of one’s readiness to seize chances and make the most of them. He recognized that many people fail to achieve wealth because they do not identify or take action on the opportunities available to them.\
Moreover, Arkad’s own experiences shaped this philosophy. He started his own fortune in humble beginnings and learned valuable lessons that he later shared with others in Babylon. His teachings emphasized that by being proactive, recognizing potential opportunities, and being willing to take calculated risks, individuals could attract what they perceive as "good luck." Thus, in Arkad's perspective, the consistent effort towards preparation and action creates the conditions for good luck to manifest.

...other question-answer pairs...

### Step 6. Reconstructing the final answer:
In "The Richest Man in Babylon," Arkad, who is the richest man in Babylon, believes that good luck follows opportunity because he understands that wealth is not simply a matter of fate or luck; it is a result of taking action when opportunities present themselves. Throughout his life, he observed that many people fail to recognize the opportunities available to them or choose not to act on them. Instead of waiting for fortune to smile upon them, Arkad emphasized the importance of being proactive, diligent, and prepared to seize chances when they arise.

According to Arkad, those who actively seek opportunities, educate themselves about how to manage and grow their wealth, and are willing to work and invest wisely are more likely to create their own "luck." He noted that while opportunities are available to everyone, only a few grasp them and achieve their desires, while the majority hesitate or falter and consequently fall behind. This perspective underscores that luck is not merely random chance; rather, it is created through effort, willingness to take risks, and the ability to recognize and act on possibilities that life presents.

In essence, Arkad's philosophy encourages readers to be proactive and to seek out opportunities, rather than relying solely on chance for financial success. He conveys that good luck is closely linked to the willingness to engage with life's opportunities, reinforcing the idea that preparation and action can transform luck from a mere chance event into a consistent part of one's journey towards success.



## **FLARE Use-Cases**
FLARE will be demonstrated for various tasks: Fund Statement PDF Extraction, Novel Question-Answering, and Scientific Journal Summarization.

# **FLARE for Fund Statement PDF Extraction**
Although a zero-shot PDF Extraction with GPT 4o is typically decent, it gets harder as the PDF gets more complicated. For the case of Fund Statements Extraction, the performance of a zero-shot prompting may not be the most optimal because of the following issues:
1. 25 Fields to extract, meaning the prompt will be long to explain each field, decreasing the LLM's performance.
2. Different fund statements have different structure, so fields can be found different pages of the statement.
3. Statements can have structured data like tables or charts, so the result of the extract may not be ideal, increasing risk of hallucination.
4. A PDF can contain multiple statements, and each of the statement can have different number of pages.

With all these problems to tackle, there's so much risk of hallucination that a zero-shot prompting can incur. Now, it may seem silly to use RAG for PDF Extraction, because we would need the whole PDF to extract the fields, so here, I create a class called `Fakre_Retriever` where it will always return the whole page when called for context.

## Steps
The goal of the notebook is of course to extract 25 fields from any fund statement from any financial institutions. There will be three main steps to the process:\
&nbsp;&nbsp;&nbsp;&nbsp;Step 1. Extracting information from the PDF into text per page\
&nbsp;&nbsp;&nbsp;&nbsp;Step 2. Extracting 25 fields from the text per page with LLM\
&nbsp;&nbsp;&nbsp;&nbsp;Step 3. Combining the output of the LLM to fund statements\
### Step 1. Extracting information from the PDF into text per page
This step will be done by first turning the PDF into an image because a PDF can contain text, charts, and tables. Then, two types of OCR, PyTesseract and EasyOCR, will be used to extra the texts, and the performance will be compared in the end. All of this will be done page-per-page.

Note: After finishing this project, I found a library called [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/), where it extracts a PDF, turning text it into markdown, perfect for LLM,  and getting the images, where I can use OCR to extrac the text. I used this library for zero-shot prompting PDF extraction with Llama3 and managed to increase the accuracy from 85%% when using OCR for pdf-to-text, to 94%.

### Step 2. Extracting 25 fields from the text per page with LLM
There will be 2 methods of extraction: Regular and FLARE. The only difference is that in the first things in Step 2 is zero-shot prompting and FLARE. After getting the initial extract, a call to the LLM to clean the output into a python dictionary will be done. On the rare occassions that the cleaning fails, the whole extraction repeats up to a specified time.

### Step 3. Combining the output of the LLM to fund statements
Now that we have dictionaries of the fields found in each page in a PDF (where each PDF can have multiple funds of different number of pages), we need to combine them. This will be done recursively through quite a number of steps. However, the algorithm revolves around the assumption that the initial start of a fund will always have the number of fields, and each fund will have 3-5 pages. Although not perfect and haven't been tested on a large dataset, this algorithm hasn't faced any issue so far. The implementation of this can be found [here](https://github.com/Justin-Jonany/FLARE_Implementation/blob/main/fund_statement_extractor/extractor.py)

