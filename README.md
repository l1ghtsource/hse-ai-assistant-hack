# HSE AI Assistant Hack: Python

_MISIS Kurenkov AI team_

Team Members:

1. **Даниил Душенев** - ML
2. **Кирилл Рыжичкин** - ML
3. **Егор Гречин** - ML
4. **Артём Плужников** - ML

## Кратко

> Необходимо разработать алгоритм, который будет подсказывать студентам, как поправить их Python код, чтобы он работал корректно.

## Зачем и кому это?

> Студенты, изучающие программирование на Python, периодически сталкиваются с трудностями при выполнении практических заданий. Особенно это проявляется при посылке задачи в тестирующую систему - открытые тесты пройдены, а вот закрытые - нет, и узнать их никак нельзя. Мы хотим, чтобы алгоритм играл роль личного ментора, который сможет проанализировать код на предмет не только синтаксических ошибок (которые сможет выявить и среда разработки, и линтер, и др.), но и логические ошибки. Он будет направлять мысль в нужное русло, указывать на необходимость перепроверки какой-либо логики в коде, но не будет писать код за него.

## Предложенное решение

1. Были собраны синтетические данные с помощью `gpt-4o` и `Qwen/Qwen2.5-72B`, расширен размер обучающей выборки до 1400 примеров: [условия и авторские решения](data/train_generated.csv), [неверные решения студентов](data/NEW_hints_synth_total.pkl)
2. В качестве основного решения предлагается LoRA-адаптер для `unsloth/Qwen2.5-32B-bnb-4bit` (скачать можно по [ссылке](https://huggingface.co/lightsource/final-lora-qwen32b))
3. Помимо этого был обучен второй LoRA-адаптер для `unsloth/Qwen2.5-32B-bnb-4bit` с другими гиперпараметрами, однако он имеет несколько меньший скор (скачать можно по [ссылке](https://huggingface.co/lightsource/qwen32b-4bit-lora-newsynth-newparams-81steps))
4. Также есть легковесное решение: LoRA-адаптер для `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, при обучении потребляется всего 9GB VRAM (скачать можно по [ссылке](https://huggingface.co/lightsource/lora-synth-8b-llama))
5. Дополнительно есть решение с Few-Shot + Answer Reconstruction для `Qwen/Qwen2.5-72B` по Api HuggingFace, здесь модель не дообучалась под эту задачу.

## Для воспроизведения решения:

Cпособ 1:

1. Зайдите в папку `/src`, запустите скрипт `src/main.py`.
2. В корне этой папке после окончания работы скрипта вы получите готовый сабмит.

Cпособ 2 (предпочтителен):

1. Запустить [ноутбук](<notebooks/6.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D64)%20Training.ipynb>) для обучения модели.
2. Запустить [ноутбук](<notebooks/7.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D64)%20Inference.ipynb>) для инференса, предварительно проверив, что выбран именно `checkpoint-140`.
3. Запустить [ноутбук](notebooks/10.%20Generate%20Submit.ipynb) для генерации сабмита, а качестве .pkl файла указать output предыдущего инференс-ноутбука.

Уточнение:

> Вся работа производилась на `NVIDIA A100`, `CUDA 12.4`, `torch 2.4.0`, версии библиотек подобраны именно под эту конфигурацию. Docker образ: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

<div style="border: 2px solid red; padding: 10px; background-color: #ffe6e6; color: red; border-radius: 5px;">
  <strong>❗ Важно:</strong> есть шанс того, что лучшим на привате окажется не тот сабмит, для которого сформирован main.py, в таком случае Вам доступен только второй вариант запуска - через jupiter ноутбуки (свяжитесь с нами, и мы сообщим какие ноутбуки запускать для воспроизведения решения).
</div>

## Ноутбуки

- [Few-shot Qwen72b Inference](notebooks/0.%20Few-shot%20Qwen72b%20Inference.ipynb) - 5-shot + answer reconstruction w/ HF Api с `Qwen/Qwen2.5-72B`, решение с готовой моделью, дающее скор 0.657 на Public LB
- [Generate Synthetic Tasks](notebooks/1.%20Generate%20Synthetic%20Tasks.ipynb) - генерация задач с `gpt-4o`
- [Generate Synthetic Wrong Solutions](notebooks/2.%20Generate%20Synthetic%20Wrong%20Solutions.ipynb) - генерация неверных решений для новых задач с `Qwen/Qwen2.5-72B`
- [Generate Answers for Synthetic Train](notebooks/3.%20Generate%20Answers%20for%20Synthetic%20Train.ipynb) - генерация авторских комментариев для новых задач с `gpt-4o`
- [Llama3.1 8b 4bit LoRA Training](notebooks/4.%20Llama3.1%208b%204bit%20LoRA%20Training.ipynb) - обучение LoRA адаптера для `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, **легковесное решение** (only 9GB VRAM for training)
- [Llama3.1 8b 4bit LoRA Inference](notebooks/5.%20Llama3.1%208b%204bit%20LoRA%20Inference.ipynb) - инференс вышеуказанной модели, скор 0.637 на Public LB
- [Qwen2.5 32b 4bit LoRA (r=64) Training](<notebooks/6.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D64)%20Training.ipynb>) - обучение LoRA адаптера для `unsloth/Qwen2.5-32B-bnb-4bit`, **основное решение** (30GB VRAM for training)
- [Qwen2.5 32b 4bit LoRA (r=64) Inference](<notebooks/7.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D64)%20Inference.ipynb>) - инференс вышеуказанной модели, скор **0.688** на Public LB
- [Qwen2.5 32b 4bit LoRA (r=256) Training](<notebooks/8.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D256)%20Training.ipynb>) - обучение LoRA адаптера для `unsloth/Qwen2.5-32B-bnb-4bit` с другими гиперпараметрами (70GB VRAM for training)
- [Qwen2.5 32b 4bit LoRA (r=256) Inference](<notebooks/9.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D256)%20Inference.ipynb>) - инференс вышеуказанной модели, скор 0.676 на Public LB
- [Generate Submit](notebooks/10.%20Generate%20Submit.ipynb) - генерация сабмита из .pkl файлов инференса
