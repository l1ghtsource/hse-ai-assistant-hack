import pickle
from unsloth import FastLanguageModel
import pandas as pd
import token
import tokenize
from io import StringIO
import re
from tqdm import tqdm

# если какие-то ошибки версий, выполните пожалуйста следуюшие команды (версия для A100):
# pip install "torch==2.4.0"
# pip install "peft==0.13.2"
# pip install "trl==0.11.1"
# pip install "transformers==4.44.2"
# pip install "bitsandbytes==0.44.1"
# pip install "accelerate==0.34.2"
# pip install "huggingface_hub==0.25.1"
# pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='outputs/checkpoint-140',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

prompt = """
### Instruction:
Вы — опытный ментор по программированию, помогающий студентам учиться на своих ошибках. Ваша задача — давать студенту полезные советы и направлять его, не предоставляя готового решения задачи. Когда студент показывает код, вы должны:
Никогда не давать готовое решение задачи.
Указать на ошибку или недочет, если они есть.
Объяснить, в чем состоит проблема, и предложить направление для исправления.
При необходимости объяснять концепции, которые могут помочь студенту найти решение самостоятельно.
Объясните, что может быть не так в его решении, используя за основу одну или несколько из следующих фраз:
- Обратите внимание на неверный...
- ...необходимо проверить, что...
- Вы забыли поставить...
- Необходимо использовать...
- Вы некорректно...
- Проверьте написание...
- В данном случае не нужно...
- ...неверный синтаксис...
- Ваш код охватывает не все возможные случаи...
- Попробуйте дополнить...
- Вы использовали неверную...
- Ошибка при обращении к...
- Проверьте, что...
- Ваш код использует неверный...

### Input:
{}

### Response:
{}
"""


def get_hint(input_,):
    inputs = tokenizer(
        [
            prompt.format(
                input_,
                '',
            )
        ], return_tensors='pt').to('cuda')

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    return tokenizer.batch_decode(outputs)


test_solutions = pd.read_excel('../data/test/solutions.xlsx')
test_tasks = pd.read_excel('../data/test/tasks.xlsx')
test_tests = pd.read_excel('../data/test/tests.xlsx')


def combine_data(solutions_, tasks_, tests_):
    solutions, tasks, tests = solutions_.copy(), tasks_.copy(), tests_.copy()

    grouped_tests = tests.groupby('task_id').apply(
        lambda x: "; ".join([
            f"number: {row['number']}, type: {row['type']}, input: {row['input']}, output: {row['output']}"
            for _, row in x.iterrows()
        ])
    ).reset_index()
    grouped_tests = grouped_tests.rename(columns={0: 'tests'})

    merged_df = pd.merge(solutions, tasks, left_on='task_id', right_on='id', suffixes=('_solution', '_task'))
    final_df = pd.merge(merged_df, grouped_tests, how='left', on='task_id')

    return final_df


test = combine_data(test_solutions, test_tasks, test_tests)

test_text = test[['description', 'author_solution', 'student_solution', 'tests']]


def strip_comments_and_docstrings(code_str):
    """
    Strip comments, docstrings, and multiline comments from a given code string.
    """
    # First, remove multiline comments using regex
    multiline_comment_pattern = r'/\*.*?\*/'
    code_str = re.sub(multiline_comment_pattern, '', code_str, flags=re.DOTALL)

    result = []
    prev_toktype = token.INDENT
    last_lineno = -1
    last_col = 0

    # Tokenize the cleaned string
    try:
        tokgen = tokenize.generate_tokens(StringIO(code_str).readline)
        for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
            if slineno > last_lineno:
                last_col = 0
            if scol > last_col:
                result.append(" " * (scol - last_col))

            if toktype == token.STRING and prev_toktype == token.INDENT:
                # Skip docstring
                pass
            elif toktype == tokenize.COMMENT:
                # Skip comment
                pass
            else:
                result.append(ttext)

            prev_toktype = toktype
            last_col = ecol
            last_lineno = elineno
    except tokenize.TokenError:
        # Handle TokenError gracefully
        pass

    return ''.join(result)


test_text['student_solution'] = test_text['student_solution'].apply(strip_comments_and_docstrings)

input_template = '''
Условие задачи: {description}

Эталонное решение: {author_solution}

Решение студента: {student_solution}
'''

test_df = pd.DataFrame({
    'input': test_text.apply(lambda row: input_template.format(
        description=row['description'],
        author_solution=row['author_solution'],
        student_solution=row['student_solution']
    ), axis=1)
})

hints = []
for i in tqdm(range(len(test_df))):
    hint = get_hint(test_df.iloc[i].input)
    hints.append(hint[0].split('Подсказка:')[-1].split('\n\n<|im_end|>')[0].strip())


with open('hints_synthlora_qwen32b_140steps.pkl', 'wb') as f:
    pickle.dump(hints, f)
