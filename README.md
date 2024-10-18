# HSE AI Assistant Hack: Python

*MISIS Kurenkov AI team*

Team Members:
1) **Даниил Душенев** - ML
2) **Кирилл Рыжичкин** - ML
3) **Егор Гречин** - ML
5) **Артём Плужников** - ML
 
## Кратко

> Необходимо разработать алгоритм, который будет подсказывать студентам, как поправить их Python код, чтобы он работал корректно.

## Зачем и кому это?

> Студенты, изучающие программирование на Python, периодически сталкиваются с трудностями при выполнении практических заданий. Особенно это проявляется при посылке задачи в тестирующую систему - открытые тесты пройдены, а вот закрытые - нет, и узнать их никак нельзя. Мы хотим, чтобы алгоритм играл роль личного ментора, который сможет проанализировать код на предмет не только синтаксических ошибок (которые сможет выявить и среда разработки, и линтер, и др.), но и логические ошибки. Он будет направлять мысль в нужное русло, указывать на необходимость перепроверки какой-либо логики в коде, но не будет писать код за него.

## Предложенное решение

тут описать решение

## Ноутбуки

- [Few-shot Qwen72b Inference](notebooks/0.%20Few-shot%20Qwen72b%20Inference.ipynb) - 5-shot + answer reconstruction w/ HF Api с `Qwen/Qwen2.5-72B`, решение с готовой моделью, дающее скор 0.657 на Public LB
- [Generate Synthetic Tasks](notebooks/1.%20Generate%20Synthetic%20Tasks.ipynb) - генерация задач с `gpt-4o`
- [Generate Synthetic Wrong Solutions](notebooks/2.%20Generate%20Synthetic%20Wrong%20Solutions.ipynb) - генерация неверных решений для новых задач с `Qwen/Qwen2.5-72B`
- [Generate Answers for Synthetic Train](notebooks/3.%20Generate%20Answers%20for%20Synthetic%20Train.ipynb) - генерация авторских комментариев для новых задач с `gpt-4o`
- [Llama3.1 8b 4bit LoRA Training](notebooks/4.%20Llama3.1%208b%204bit%20LoRA%20Training.ipynb) - обучение LoRA адаптера для `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, легковесное решение (only 9GB VRAM for training)
- [Llama3.1 8b 4bit LoRA Inference](notebooks/5.%20Llama3.1%208b%204bit%20LoRA%20Inference.ipynb) - инференс вышеуказанной модели, скор 0.637 на Public LB
- [Qwen2.5 32b 4bit LoRA (r=64) Training](notebooks/6.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D64)%20Training.ipynb) - обучение LoRA адаптера для `unsloth/Qwen2.5-32B-bnb-4bit`, __основное решение__ (30GB VRAM for training)
- [Qwen2.5 32b 4bit LoRA (r=64) Inference](notebooks/7.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D64)%20Inference.ipynb) - инференс вышеуказанной модели, скор **0.688** на Public LB
- [Qwen2.5 32b 4bit LoRA (r=256) Training](notebooks/8.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D256)%20Training.ipynb) - обучение LoRA адаптера для `unsloth/Qwen2.5-32B-bnb-4bit` с другими гиперпараметры (70GB VRAM for training)
- [Qwen2.5 32b 4bit LoRA (r=256) Inference](notebooks/9.%20Qwen2.5%2032b%204bit%20LoRA%20(r%3D256)%20Inference.ipynb) - инференс вышеуказанной модели, скор 0.676 на Public LB
- [Generate Sumbit](notebooks/10.%20Generate%20Sumbit.ipynb) - генерация сабмита из .pkl файлов инференса
