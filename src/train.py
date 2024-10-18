from unsloth import FastLanguageModel
import torch
import pandas as pd
import pickle
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

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
    model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=52,
    use_rslora=False,
    loftq_config=None
)

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

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_, output in zip(inputs, outputs):
        text = prompt.format(input_, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


train_solutions = pd.read_excel('../data/train/solutions.xlsx')
train_tasks = pd.read_excel('../data/train/tasks.xlsx')
train_tests = pd.read_excel('../data/train/tests.xlsx')

synth = pd.read_csv('../data/train_generated.csv')

with open('../data/NEW_hints_synth_total.pkl', 'rb') as f:
    synth_hints = pickle.load(f)


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


train = combine_data(train_solutions, train_tasks, train_tests)

train_text = train[['description', 'author_solution', 'student_solution', 'author_comment', 'tests']]

synth['author_comment'] = synth_hints
synth_text = synth[['description', 'author_solution', 'student_solution', 'author_comment', 'tests']]

train_text = pd.concat([train_text, synth_text], axis=0).sample(frac=1, random_state=52).reset_index(drop=True)

input_template = '''
Условие задачи: {description}

Эталонное решение: {author_solution}

Решение студента: {student_solution}
'''

output_template = '''
Подсказка: {author_comment}
'''

train_df = pd.DataFrame({
    'input': train_text.apply(lambda row: input_template.format(
        description=row['description'],
        author_solution=row['author_solution'],
        student_solution=row['student_solution']
    ), axis=1),
    'output': train_text.apply(lambda row: output_template.format(
        author_comment=row['author_comment']
    ), axis=1)
})

train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

args = TrainingArguments(
    report_to='none',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    learning_rate=2e-4,
    fp16=(not is_bfloat16_supported()),
    bf16=(is_bfloat16_supported()),
    save_strategy="steps",
    save_steps=20,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=52,
    output_dir="outputs"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

trainer_stats = trainer.train()
