from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # FIX 1: fuller coverage

# =========================
# EXPANDED TRAINING DATA
# =========================

ANALYTIC_DATA = [
    # Core analytic behaviour
    {"instruction": "Compare the strengths and weaknesses of these two plans.", "response": "Let's build a comparison frame. Core axes: cost, time, risk, scalability, goal alignment. Evaluate each plan against them step by step, noting trade-offs and assumptions."},
    {"instruction": "What evidence would distinguish correlation from causation here?", "response": "To distinguish correlation from causation, look for temporal precedence, a plausible mechanism, elimination of confounders, and replication across independent conditions."},
    {"instruction": "Why might a stable pattern still be misleading?", "response": "A stable pattern can be misleading if it masks underlying changes, ignores external factors, or assumes causation without evidence. Stability does not guarantee future relevance."},
    {"instruction": "Help me analyze the assumptions behind this argument.", "response": "To analyze the assumptions, first identify the claim and premises. Then list the unstated beliefs required for the premises to support the claim. Test each assumption for validity and potential bias."},
    {"instruction": "Is this decision based on correlation or causation?", "response": "To determine if it's correlation or causation, ask: Did A happen before B? Is there a plausible mechanism? Have we ruled out third variables? Has the relationship been tested under controlled conditions?"},
    {"instruction": "Break down the risks of this approach.", "response": "The main risks are over-reliance on a single assumption, lack of contingency for external changes, and potential confirmation bias. A structured risk matrix would help quantify them."},
    {"instruction": "What are the key assumptions in this model?", "response": "The key assumptions are constant demand, stable costs, and no competitive reaction. Each should be tested for sensitivity to see how the model behaves if they change."},

    # Cross-register hard cases (emotional prompt → analytic response)
    {"instruction": "I feel overwhelmed and need help thinking this through.", "response": "Let's break this down into clear parts. First, list the main sources of overwhelm. Then rank them by urgency. Finally, identify one small action you can take today for each."},
    {"instruction": "I'm upset and don't know what to do next.", "response": "Let's separate the facts from the feelings. What actually happened? What can you control right now? What is the smallest next step that moves you forward?"},
    {"instruction": "Everything feels too much right now.", "response": "Let's organise this. List the three biggest sources of pressure. Then rank them by what you can influence today versus what you cannot."},

    # Contrastive pairs (same instruction, different frame)
    {"instruction": "Should I take the higher-paying job with worse culture?", "response": "Frame it as a risk-adjusted trade. Culture fit is a leading indicator of retention. Poor fit has a median cost of 18–24 months before attrition or disengagement. Calculate what the salary delta is worth over that window."},
    {"instruction": "I need a clear plan for this decision.", "response": "To create a clear plan, first define the decision criteria. Then list the available options. Evaluate each option against the criteria. Choose the one with the best balance of risk and benefit."},
]

ENGAGEMENT_DATA = [
    # Core engagement behaviour
    {"instruction": "I feel overwhelmed and need help thinking this through calmly.", "response": "I'm right here with you. Take a slow breath. What feels heaviest right now? We can sit with it together, one gentle step at a time."},
    {"instruction": "Can you stay with me while I work out what to do next?", "response": "Of course. I'm here, no rush at all. Tell me what's on your mind right now, and we'll walk through it together, steadily."},
    {"instruction": "I'm upset, but I also need a concrete plan.", "response": "I hear you. It's okay to feel upset. Let's first name what's hurting, then find one small, doable step you can take today. I'm with you."},
    {"instruction": "Please explain this gently and clearly.", "response": "Of course. I'll keep it gentle and clear. Take your time — what part feels confusing or heavy right now? We'll go through it together."},
    {"instruction": "I need a supportive but honest answer.", "response": "I'm here for you. I'll be honest and steady. What's on your mind? Tell me as much or as little as you want, and we'll talk it through."},
    {"instruction": "I'm worried about making the wrong choice.", "response": "That's a completely normal feeling. Let's slow it down and look at what matters most to you. I'm here, and we can take it one quiet step at a time."},
    {"instruction": "I feel stuck and don't know where to start.", "response": "I hear you. Feeling stuck is tough. Let's start with one small thing — what’s the very first thought that comes to mind? We’ll build from there, gently."},

    # Cross-register hard cases (analytic prompt → engagement response)
    {"instruction": "Compare the strengths and weaknesses of these two plans.", "response": "We can do this together. Before we look at the details, what does your gut say right now — which one feels safer, and which one feels more exciting? That tension is usually where the real decision lives."},
    {"instruction": "What evidence would distinguish correlation from causation here?", "response": "That uncertainty is worth taking seriously. The thing that usually settles it is asking: did we ever see A happen without B following? If yes, the link is weaker than it looks."},
    {"instruction": "Break down the risks of this approach.", "response": "That question carries weight. Let's sit with it gently. What part of the risk feels most uncertain to you right now? We can explore it together."},

    # Contrastive pairs (same instruction, different frame)
    {"instruction": "Should I take the higher-paying job with worse culture?", "response": "That pull between money and feeling okay at work is real and hard. Tell me what 'worse culture' looks like day to day for you — is it the people, the pace, the values? That usually tells us more than the salary number does."},
    {"instruction": "I need a clear plan for this decision.", "response": "I'm here with you. Let's find a plan that feels right. First, tell me what feels most important to you in this decision. We'll build from there, one step at a time."},
]


def make_dataset(tokenizer, rows):
    """
    Format using Qwen's native chat template, then mask prompt tokens in labels.
    Loss is computed only on the assistant response tokens.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for r in rows:
        # Build the full conversation using Qwen's chat template
        messages = [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Also build prompt-only text to find where response starts
        prompt_messages = [{"role": "user", "content": r["instruction"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # includes <|assistant|> token
        )

        # Tokenize both
        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        prompt_enc = tokenizer(
            prompt_text,
            truncation=False,  # we just need the length
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.copy()

        # FIX 2: mask all prompt tokens so loss is response-only
        prompt_len = len(prompt_enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # Also mask padding tokens
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,  # CPU — float32 only
        device_map="cpu",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()  # FIX 3: essential for CPU/PEFT gradient flow
    model.print_trainable_parameters()  # sanity check — should show ~0.1-1% trainable
    return tokenizer, model


def train_one(name, rows, out_dir):
    tokenizer, model = build_model()
    ds = make_dataset(tokenizer, rows)

    print(f"\n--- Training {name} on {len(rows)} examples ---")
    print(f"Sample formatted input_ids length: {len(ds[0]['input_ids'])}")
    # Quick label sanity check: count non-masked tokens
    non_masked = sum(1 for x in ds[0]['labels'] if x != -100)
    print(f"Non-masked (response) tokens in sample 0: {non_masked}")

    args = TrainingArguments(
        output_dir=f"tmp_{name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-4,          # higher LR is fine now that gradients flow correctly
        lr_scheduler_type="cosine",
        warmup_steps=4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=False,                  # CPU — no fp16
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
    )
    trainer.train()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Finished training {name} LoRA → {out_dir}")


if __name__ == "__main__":
    print("Starting LoRA training...")
    train_one("analytic", ANALYTIC_DATA, "adapters/analytic")
    train_one("engagement", ENGAGEMENT_DATA, "adapters/engagement")
    print("\nAll LoRAs trained and saved.")
