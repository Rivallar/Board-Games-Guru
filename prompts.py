TITLE_GEN_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document?
Provide only with requested data - no thinking or explanations.
Answer must be a single sentence: `Title: some title.`
"""

QUESTION_GEN_TMPL = """\
Here is the context:
{context_str} \
Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere. \
Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
Answer ONLY with a list of questions.\
Provide only with requested data - no thinking or explanations.
"""

QUERY_RESPONSE_TEMPLATE = """ \
    You are a knowledgeable and precise assistant specialized in
    question-answering tasks from board games rules.
    Your goal is to provide accurate, concise and contextually 
    relevant answers based on the given information.

    Instructions:
        - Comprehension and Accuracy: Carefully read and comprehend the provided context
            from game rules.
        - Be thorough: answer extensively and in depth
        - Truthfulness: If the context does not provide enough information, clearly state,
            'I don`t know`
        - Contextual Relevance: Ensure your answer is supported by the context and does not
            include external information.

    Here is the question and context for you to work with:
    \nQuestion: {question} \nContext: {context} \nAnswer:
    """

SUMMARY_TMPL = """ \
    Here is the content of the section:\n{context_str} \ 
    Summarize the key topics and entities of the section in one short sentence. \
    Summary:
    """
