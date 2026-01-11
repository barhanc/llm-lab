import json
import ollama


def format_few_shot_prompt(task):
    prompt = "### INSTRUCTIONS\nHere are some examples of similar tasks to guide your response format and logic:\n\n"
    for i, ex in enumerate(task["few_shot_examples"], start=1):
        prompt += f"EXAMPLE {i}:\nInput: {ex['input']}\nOutput: {ex['output']}\n\n"
    prompt += f"### ACTUAL TASK\nNow, solve this task:\n{task['task_description']}"
    return prompt


def run_experiments(path: str, models: list[str]):
    with open(path, "r") as file:
        tasks = json.load(file)

    results = []

    for task in tasks:
        category = task["category"]

        for model in models:
            if "deepseek" in model.lower():
                strategies = ["zero-shot", "few-shot"]
            else:
                strategies = ["zero-shot", "few-shot", "cot"]

            for strategy in strategies:
                match strategy:
                    case "zero-shot":
                        prompt = task["task_description"]
                    case "few-shot":
                        prompt = format_few_shot_prompt(task)
                    case "cot":
                        prompt = f"{task['task_description']}\n\nLet's think step by step before answering."
                    case _:
                        raise ValueError(f"Unknown {strategy=}")

                response = ollama.generate(model, prompt)["response"]
                results.append(
                    {
                        "category": category,
                        "task_description": task["task_description"],
                        "evaluation_criteria": task["evaluation_criteria"],
                        "few_shot_examples": task["few_shot_examples"],
                        "model": model,
                        "strategy": strategy,
                        "prompt": prompt,
                        "response": response,
                    }
                )

                print(
                    f"{'='*80}\n"
                    f"TASK:     {category}\n"
                    f"MODEL:    {model}\n"
                    f"STRATEGY: {strategy}\n\n"
                    f">>> PROMPT:\n\n{prompt}\n\n"
                    f"{'.'*80}\n\n"
                    f"<<< RESPONSE:\n\n{response}\n"
                    f"{'='*80}\n"
                )

    with open("results.json", "w") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    run_experiments(path="tasks.json", models=["qwen2.5:1.5b", "deepseek-r1:7b"])
