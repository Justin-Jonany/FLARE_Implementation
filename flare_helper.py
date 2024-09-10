from openai import OpenAI
import numpy as np
import copy

def format_docs(docs):
    '''
    formats the list of documents into a string

    Args:
      docs: list of documents

    Returns:
      string of documents
    '''
    return "\n\n".join(doc.page_content for doc in docs)


def logprobs_simple_print(response, n=None):
    """
    Prints an OpenAI response with logprobs for each tokens

    Args:
      response: OpenAI Response object with logprobs
      n: First n tokens to print
    """
    for i, token_data in enumerate(response.choices[0].logprobs.content[:n]):
        print(f'{i}: {token_data.token}')
        print(token_data.bytes)
        if hasattr(token_data, 'uncertain'):
            print(f'{token_data.logprob}\t{str(token_data.uncertain)}')
        else:
            print(f'{token_data.logprob}')
        print('\n')


def logprobs_pretty_print(response, prompt):
    """
    Pretty prints an OpenAI response with logprobs for each tokens

    Args:
      response: OpenAI Response object with logprobs
      prompt: Original Prompt
    """
    logprobs = [token.logprob for token in response.choices[0].logprobs.content]
    response_text = response.choices[0].message.content
    response_text_tokens = [
        token.token for token in response.choices[0].logprobs.content]
    if hasattr(response.choices[0].logprobs.content[0], 'uncertain'):
        uncertainty = [
            token.uncertain for token in response.choices[0].logprobs.content]

    max_starter_length = max(
        len(s) for s in ["Prompt:", "Response:", "Tokens:", "Logprobs:", "Perplexity:"])
    max_token_length = max(len(s) for s in response_text_tokens)

    new_lines_index = []
    for i, token in enumerate(response_text_tokens):
        if '\n' in token:
            new_lines_index += [i]
    formatted_response_tokens = [
        s.rjust(max_token_length) for s in response_text_tokens]
    formatted_lps = [f"{lp:.2f}".rjust(max_token_length) for lp in logprobs]
    formatted_linear_probs = [
        f"{np.round(np.exp(lp)*100,2):.2f}%".rjust(max_token_length) for lp in logprobs]
    if hasattr(response.choices[0].logprobs.content[0], 'uncertain'):
        formatted_uncertain = [str(uncertain).rjust(
            max_token_length) for uncertain in uncertainty]
    perplexity_score = np.exp(-np.mean(logprobs))
    print("Prompt:".ljust(max_starter_length), prompt)
    print("Response:".ljust(max_starter_length), response_text, "\n")
    print("==========================================================================================================================================")
    cut_off_start = 0
    cut_off_end = 0
    for i, new_line_index in enumerate(new_lines_index):
        cut_off_start = 0 if i == 0 else new_lines_index[i - 1] + 1
        cut_off_end = new_lines_index[i] + 1
        print("Tokens:".ljust(max_starter_length), " ".join(
            formatted_response_tokens[cut_off_start:cut_off_end]))
        print("Logprobs:".ljust(max_starter_length), " ".join(
            formatted_lps[cut_off_start:cut_off_end]))
        print("Linprob:".ljust(max_starter_length), " ".join(
            formatted_linear_probs[cut_off_start:cut_off_end]))
        if hasattr(response.choices[0].logprobs.content[0], 'uncertain'):
            print("Uncertainty:".ljust(max_starter_length), " ".join(
                formatted_uncertain[cut_off_start:cut_off_end]))
        print("==========================================================================================================================================")
    print("Tokens:".ljust(max_starter_length), " ".join(
        formatted_response_tokens[cut_off_end:]))
    print("Logprobs:".ljust(max_starter_length),
          " ".join(formatted_lps[cut_off_end:]))
    print("Linprob:".ljust(max_starter_length),
          " ".join(formatted_linear_probs[cut_off_end:]))
    if hasattr(response.choices[0].logprobs.content[0], 'uncertain'):
        print("Uncertainty:".ljust(max_starter_length),
              " ".join(formatted_uncertain[cut_off_end:]))

    print("==========================================================================================================================================")
    print("Perplexity:".ljust(max_starter_length), perplexity_score, "\n")


def combine_token_to_word(response):
    """
    Combines the tokens in an OpenAI response that are parts of words, into a word.

    Args:
        response: OpenAI response object

    Returns:
        Open AI Response object
    """
    temp_response = copy.deepcopy(response)
    new_logprobs_list = []
    new_logprob = temp_response.choices[0].logprobs.content[0]
    skip_next = False
    for i, token_data in enumerate(temp_response.choices[0].logprobs.content):
        if (i == 0) or (skip_next):
            if skip_next:
                skip_next = False
            continue
        if '\n' in token_data.token:
            new_logprob.token += token_data.token
            new_logprob.bytes += token_data.bytes
            new_logprob.logprob = min(new_logprob.logprob, token_data.logprob)
            new_logprob.top_logprobs += token_data.top_logprobs
            new_logprobs_list.append(new_logprob)
            new_logprob = temp_response.choices[0].logprobs.content[i + 1]
            skip_next = True
            continue
        if token_data.bytes[0] == 32:
            new_logprobs_list.append(new_logprob)
            new_logprob = token_data
        else:
            new_logprob.token += token_data.token
            new_logprob.bytes += token_data.bytes
            new_logprob.logprob = min(new_logprob.logprob, token_data.logprob)
            new_logprob.top_logprobs += token_data.top_logprobs
        if i == (len(temp_response.choices[0].logprobs.content) - 1):
            new_logprobs_list.append(new_logprob)
    temp_response.choices[0].logprobs.content = new_logprobs_list
    return temp_response


def split(list, n):
    """
    Given a list, it returns a new list of n-sized lists. The items in each
    n-sized list is determined by the order of the original list.

    Args:
        list: a Python list
        n: int

    Return:
        A list of n-sized lists
    """
    return [list[i:i+n] for i in range(0, len(list), n)]


def token_data_group_sequeeze(token_data_list, aggregate_func):
    """
    Given a list of Open AI token data, it squeezes it into one token. All the token will
    be concatenated, the bytes will be concatenated, the logprob will be determined by
    the aggregate_func, and the logprobs will be concatenated.

    Args:
        token_data_list: a list of Open AI token data objects
        aggregate_func: a function that accepts n-numbers of int as it's argument and returns an int
    
    Returns:
        An Open AI token data object
    """
    new_logprob = copy.deepcopy(token_data_list[0])
    new_logprob.token = ''.join(
        [token_data.token for token_data in token_data_list])
    new_logprob.bytes = [
        byte for token_data in token_data_list for byte in token_data.bytes]
    new_logprob.logprob = aggregate_func(
        [token_data.logprob for token_data in token_data_list])
    new_logprob.top_logprobs = [
        top_logprob for token_data in token_data_list for top_logprob in token_data.top_logprobs]
    if hasattr(token_data_list[0], 'uncertain'):
        new_logprob.uncertain = token_data_list[0].uncertain
    return new_logprob


def sequential_combine(response, mode, aggregate_func=min):
    """
    Given an OpenAI response object with logprobs, combines the token_data list
    into groups of size mode. The method to combine the logprobs value is determined
    by the aggregate_func.

    Args:
        response: OpenAI response object
        mode: int
        aggregate_func: a function that accepts n-numbers of int as it's argument and returns an int

    Returns:
        An OpenAI Response object    
    """
    temp_response = copy.deepcopy(response)
    skip_next = False
    logprobs = temp_response.choices[0].logprobs.content

    # split by sentences
    sentences = []
    start = 0
    total_token = 0
    for i, token_data in enumerate(logprobs):
        if ('\n' in token_data.token) or ('.' in token_data.token) or ('?' in token_data.token) or ('!' in token_data.token):
            sentences += [logprobs[start: i + 1]]
            total_token += len(logprobs[start: i + 1])
            start = i + 1
    if total_token != len(logprobs):
        sentences += [logprobs[start:]]
        total_token += len(logprobs[start:])

    log_probs_list = []
    # splits the sentences by words, and group them based on mode
    if isinstance(mode, int):
        for sentence in sentences:
            grouped_sentence = split(sentence, mode)
            for group in grouped_sentence:
                log_probs_list += [
                    token_data_group_sequeeze(group, aggregate_func)]
        temp_response.choices[0].logprobs.content = log_probs_list
    return temp_response


def annotater(response, tolerance=-0.4):
    """
    Given an OpenAI response object with logprobs, marks all the tokens
    where the logprob is below the tolerance as uncertain by adding a field
    called uncertain and marking it as True.

    Args:
        response: OpenAI response object
        tolerance: int

    Returns:
        An OpenAI response object
    """
    temp_response = copy.deepcopy(response)
    for token_data in temp_response.choices[0].logprobs.content:
        if token_data.logprob < tolerance:
            token_data.uncertain = True
        else:
            token_data.uncertain = False
    return temp_response


def annotated_combiner(response, aggregate_func=np.mean):   
    """
    Given a OpenAI response object with logprobs, combines all adjacent 
    token data objects in the list of response into one token data object,
    aggregated with aggregate_func.

    Args:
        response: OpenAI Response object
        aggregate_func: a function that accepts n-numbers of int as it's argument and returns an int
    
    Returns:
        An OpenAI response object
    """
    temp_response = copy.deepcopy(response)
    index_groups = []
    current_group = []
    for token_data in temp_response.choices[0].logprobs.content:
        if token_data.uncertain:
            if ('\n' not in token_data.token) and ('.' not in token_data.token) and ('?' not in token_data.token) and ('!' not in token_data.token):
                current_group += [token_data]
            else:
                if len(current_group) > 0:
                    index_groups += [current_group]
                    current_group = []
                index_groups += [[token_data]]
        else:
            if len(current_group) > 0:
                index_groups += [current_group]
                current_group = []
                index_groups += [[token_data]]
                continue
            index_groups += [[token_data]]
    log_probs_list = []
    for group in index_groups:
        log_probs_list += [token_data_group_sequeeze(group, aggregate_func)]
    temp_response.choices[0].logprobs.content = log_probs_list
    return temp_response


def uncertain_marker(response):
    """
    Given an OpenAI response object with logprobs, modifies all the logprobs token data list 
    strings where uncertain is set to True with '[uncertain]' + token + '[/uncertain]'.

    Args:
        response: OpenAI Object
    
    Returns:
        An OpenAI response object
    """
    temp_response = copy.deepcopy(response)
    if not hasattr(response.choices[0].logprobs.content[0], 'uncertain'):
        temp_response = annotated_combiner(annotater(temp_response))
    for token_data in temp_response.choices[0].logprobs.content:
        if token_data.uncertain:
            token_data.token = ' ([uncertain]' + \
                token_data.token + ' [/uncertain]) '
    temp_response.choices[0].message.content = ''.join(
        [i.token for i in temp_response.choices[0].logprobs.content])
    return temp_response
