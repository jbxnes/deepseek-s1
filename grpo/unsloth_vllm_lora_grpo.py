"""
Unsloth memory-efficient TRL-vLLM GRPO
Exported from
https://docs.unsloth.ai/basics/reasoning-grpo-and-rl
"""
# Use `PatchFastRL` before all functions to patch GRPO and other RL algorithms!

import re
import sys

sys.path = [
    "",
    ".venv/lib/python3.12/site-packages",
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/lib/python3/dist-packages",
]

# Must import unsloth before trl.
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import torch


PatchFastRL("GRPO", FastLanguageModel)

# Load up `Qwen 2.5 3B Instruct`, and set parameters


max_seq_length = 4096  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/model-weights/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

# ### Data Prep
# <a name="Data"></a>
#
# We directly leverage
# [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
# for data prep and all reward functions. You are free to create your own!


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train", num_examples: int = None) -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    
    if num_examples is not None:
        data = data.select(range(min(num_examples, len(data))))
        
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore

dataset = get_gsm8k_questions()


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# ### Train the model

# Patch for budget forcing
def vLLMSamplingParams(**kwargs):
    attach_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = attach_kwargs
    return sampling_params

device = "cuda" if torch.cuda.is_available() else "cpu"
# logits_processor = WaitLogitsProcessor(tokenizer, device, next_token_id=14190, min_num_tokens=512)

# vllm_sampling_params=vLLMSamplingParams(max_tokens=4096, logits_processors=[logits_processor])
vllm_sampling_params = None 

# Now set up GRPO Trainer and all configurations!
training_args = GRPOConfig(
    use_vllm=True,  # use vLLM for fast inference!
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=8,  # Decrease if out of memory
    max_prompt_length=256,
    max_completion_length=4096,
    num_train_epochs=1,  # Set to 1 for a full training run
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="wandb",  # Can use Weights & Biases
    output_dir="outputs",
    vllm_sampling_params=vllm_sampling_params,
)

# And let's run the trainer! If you scroll up, you'll see a table of rewards.
# The goal is to see the `reward` column increase!
#
# You might have to wait 150 to 200 steps for any action.
# You'll probably get 0 reward for the first 100 steps. Please be patient!
#
# | Step | Training Loss | reward    | reward_std | completion_length | kl       |
# |------|---------------|-----------|------------|-------------------|----------|
# | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
# | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
# | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
#

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# ### Inference
# Now let's try the model we just trained!
# First, let's first try the model without any GRPO trained:

text = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "How many r's are in strawberry?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)


sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0]
    .outputs[0]
    .text
)

# And now with the LoRA we just trained with GRPO - we first save the LoRA first!

model.save_lora("grpo_saved_lora")

# Now we load the LoRA and test:

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "How many r's are in strawberry?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)


sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0]
    .outputs[0]
    .text
)

# Our reasoning model is much better - it's not always correct,
# since we only trained it for an hour or so
# - it'll be better if we extend the sequence length and train for longer!

# Saving to float16 for VLLM

# Merge to 16bit
if False:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("example", tokenizer, save_method="merged_16bit", token="")
