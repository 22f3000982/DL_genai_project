import gradio as gr
import torch
import torch.nn as nn
from transformers import RobertaModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# ----------------------------
# CONFIG
# ----------------------------
MODEL_ID = "Ashish4129/roberta-emotion-detection"   # HF repo
BASE_MODEL = "roberta-base"
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
THRESHOLDS = [0.5, 0.5, 0.5, 0.5, 0.5]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# CUSTOM MODEL (same as training)
# ----------------------------
class CustomRobertaEmotion(nn.Module):
    def __init__(self, model_name, num_labels=5):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

# ----------------------------
# LOAD TOKENIZER
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ----------------------------
# DOWNLOAD + LOAD MODEL WEIGHTS
# ----------------------------
ckpt_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="pytorch_model.bin"
)

state_dict = torch.load(ckpt_path, map_location=device)

model = CustomRobertaEmotion(BASE_MODEL, num_labels=len(LABELS))
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_emotions(text):
    text = text.strip()
    if not text:
        return {label: 0.0 for label in LABELS}, "⚠️ Please enter text."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    score_dict = {label: float(p) for label, p in zip(LABELS, probs)}
    active = [label for label, p, th in zip(LABELS, probs, THRESHOLDS) if p >= th]
    active_text = ", ".join(active) if active else "No strong emotion detected."

    return score_dict, active_text

# ----------------------------
# GRADIO UI
# ----------------------------
description = """
### 🧠 Emotion Detection using your Fine-tuned RoBERTa  
Enter text and predict emotions: **anger, fear, joy, sadness, surprise**.
"""

# ----------------------------
# SAMPLE SENTENCES (10 samples)
# ----------------------------
examples = [
    ["I am extremely happy today!"],                # joy
    ["Why did you betray me like this?"],           # anger
    ["I feel scared walking alone at night."],      # fear
    ["I miss my old friends so much, it hurts."],   # sadness
    ["Oh wow, I was not expecting that!"],          # surprise
    ["I feel so loved and appreciated by my friends today."],  # joy
    ["Stop yelling at me, you're making me angry."],            # anger
    ["This news made me cry..."],                                # sadness
    ["Something is not right, I feel really anxious."],           # fear
    ["OMG! That twist was totally unexpected!"]                  # surprise
]

with gr.Blocks(title="Emotion Detection") as demo:

    gr.Markdown("# Emotion Detection from Text")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(label="Enter text", lines=3)
            btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            label_scores = gr.Label(label="Emotion Probabilities")
            active_labels = gr.Textbox(
                label="Detected Emotion(s)",
                interactive=False
            )

    btn.click(
        fn=predict_emotions,
        inputs=txt,
        outputs=[label_scores, active_labels]
    )

    # Add sample examples here
    gr.Examples(
        examples=examples,
        inputs=txt,
        outputs=[label_scores, active_labels],
        fn=predict_emotions
    )

if __name__ == "__main__":
    demo.launch()
