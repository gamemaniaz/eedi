# >>> LLAMA >>> #

llama_base_task_prompt = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
Before answering the question think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."""

llama_genk_knowledge_prompt = """Task Construct Name : {ConstructName}
Subject Name : {SubjectName}

Given the mathematical subject and task construct, explain how a student can perform such a task step by step, in at most 5 steps."""

llama_knowledge_task_prompt = """<knowledge>
{Knowledge}
</knowledge>

<task-context>
Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}
</task-context>

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>. Before answering the math question, think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."""

llama_tot_knowledge_prompt = """
"""

# >>> QWEN >>> #

qwen_base_task_prompt = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response></response>.
Before answering the question think step by step concisely in 1-2 sentence inside <thinking></thinking> tag and respond your final misconception inside <response></response> tag."""

qwen_genk_knowledge_prompt = """Task Construct Name : {ConstructName}
Subject Name : {SubjectName}

Given the mathematical subject and task construct, explain how a student can perform such a task step by step, in at most 5 steps."""

qwen_knowledge_task_prompt = """<knowledge>
{Knowledge}
</knowledge>

<task-context>
Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}
</task-context>

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response></response>. Before answering the math question, think step by step concisely in 1-2 sentence inside <thinking></thinking> tag and respond your final misconception inside <response></response> tag."""

qwen_tot_knowledge_prompt = """
"""
