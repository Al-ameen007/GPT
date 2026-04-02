import torch


def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze()
    decoded = tokenizer.decode(flat.tolist())
    return decoded


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_simple(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, 50, context_size)

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def generate(
    model, idx, max_new_tokens, context_size, temp=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temp > 0.0:
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def classify_review(
    text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    actual_length = min(len(input_ids), max_length, supported_context_length)

    input_ids = input_ids[:actual_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, actual_length - 1, :]
    output = torch.argmax(logits, dim=-1).item()

    return "spam" if output == 1 else "not spam"


def generate_text(
    prompt, model, tokenizer, device, max_new_tokens=100, temperature=0.7, top_k=50
):
    model.eval()

    encoded = text_to_token(prompt, tokenizer).to(device)
    context_size = model.pos_emb.weight.shape[0]

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temp=temperature,
            top_k=top_k,
            eos_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[
                0
            ],
        )

    generated = token_ids_to_text(token_ids, tokenizer)
    return generated


def classify_text(
    text, model, tokenizer, device, max_length, pad_token_id=50256, return_probs=False
):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    actual_length = min(len(input_ids), max_length, supported_context_length)

    input_ids = input_ids[:actual_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, actual_length - 1, :]

    if return_probs:
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()
        prediction = "SPAM" if predicted_class == 1 else "NOT SPAM"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "prob_not_spam": probabilities[0, 0].item(),
            "prob_spam": probabilities[0, 1].item(),
        }
    else:
        output = torch.argmax(logits, dim=-1).item()
        return "spam" if output == 1 else "not spam"
