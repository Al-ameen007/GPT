import torch


def calc_loss_batch(inputs, targets, model, device):
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)

    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())

    return loss


def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0

    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

    model.train()
    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    from gpt2_framework.generation import generate_and_print_simple

    model.train()
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                track_tokens_seen.append(tokens_seen)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Ep {epoch + 1} (Step {global_step:.0f}): ")
                print(f"Train Loss: {train_loss:.3f}")
                print(f"Validation Loss: {val_loss:.3f}")

        generate_and_print_simple(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def calc_loss_batch_classifier(input_batch, target_batch, model, device, num_classes):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader_classifier(
    dataloader, model, device, num_batches=None, num_classes=2
):
    total_loss = 0

    if len(dataloader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch_classifier(
                input_batch, target_batch, model, device, num_classes
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def calc_accuracy_loader(dataloader, model, device, num_batches=None):
    model.eval()
    correct_predictions = 0
    num_examples = 0

    if len(dataloader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    if num_examples == 0:
        return float("nan")

    return correct_predictions / num_examples


def train_classifier_simple(
    model, train_loader, val_loader, device, optimizer, eval_freq, eval_iter, num_epochs
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch_classifier(
                input_batch, target_batch, model, device, num_classes=2
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_classifier(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print(
            f"Training accuracy: {train_accuracy * 100:.2f}% | Validation accuracy: {val_accuracy * 100:.2f}%"
        )

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model_classifier(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_classifier(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader_classifier(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
