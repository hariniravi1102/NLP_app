from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch, json, os
from reward_model_train import RewardModel  # Same as trained above

prompts_path = "feed_data.jsonl"
reward_model_path = "reward_model/checkpoint-final"
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, device_map="auto")
reward_model = RewardModel("distilbert-base-uncased").to("cuda" if torch.cuda.is_available() else "cpu")
reward_model.load_state_dict(torch.load(os.path.join(reward_model_path, "pytorch_model.bin")))
reward_model.eval()

def get_prompts():
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts

def compute_reward(prompt, response):
    text = f"Prompt: {prompt}\nOutput: {response}"
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(reward_model.base.device) for k, v in tokens.items()}
    with torch.no_grad():
        result = reward_model(**tokens)
    return result["logits"].item()

prompts = get_prompts()
ppo_config = PPOConfig(model_name=model_name, batch_size=1)
trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

for prompt in prompts:
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    response_ids = model.generate(**tokens, max_new_tokens=200)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    reward = compute_reward(prompt, response)
    trainer.step([prompt], [response], [reward])

model.save_pretrained("ppo_model")