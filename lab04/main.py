import argparse
import torch

from datasets import load_dataset
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)


class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1 or state.global_step % 10 == 0:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f" [Step {state.global_step}] Peak VRAM: {peak:.2f} GB")


def run_experiment(args):
    model_config = MistralConfig(
        vocab_size=32_000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        sliding_window=args.window_size if args.use_window else None,  # type: ignore
        attn_implementation="flash_attention_2" if args.use_flash else "eager",
    )

    model = MistralForCausalLM(model_config)
    if args.use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=f"./results_{args.experiment_name}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        learning_rate=5e-4,
        logging_steps=5,
        # Optimization Toggles
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
        gradient_checkpointing=args.use_grad_checkpoint,
        # Housekeeping
        eval_steps=50,
        report_to="none",
        disable_tqdm=False,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # type:ignore
        eval_dataset=dataset["validation"],  # type:ignore
        callbacks=[MemoryCallback],  # type:ignore
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Reset memory stats before starting
    torch.cuda.reset_peak_memory_stats()

    st = torch.cuda.Event(enable_timing=True)
    et = torch.cuda.Event(enable_timing=True)

    st.record()
    trainer.train()
    et.record()
    torch.cuda.synchronize()

    metrics = trainer.evaluate()

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    total_time = st.elapsed_time(et) / 1000  # Convert ms to seconds

    print(
        "-" * 40 + "\n"
        f"Experiment: {args.experiment_name}\n"
        f"Batch Size: {args.batch_size}\n"
        f"Peak Memory: {peak_memory:.2f} GB\n"
        f"Total Time:  {total_time:.2f} s\n"
        f"Perplexity:  {torch.exp(torch.tensor(metrics['eval_loss'])):.2f}\n"
        "-" * 40
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_flash", action="store_true")
    parser.add_argument("--use_window", action="store_true")
    parser.add_argument("--use_grad_checkpoint", action="store_true")

    args = parser.parse_args()
    run_experiment(args)
