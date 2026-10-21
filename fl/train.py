"""Entrenamiento local de un cliente (lo que pasa dentro de cada computadora cliente en cada ronda)."""
import torch
import torch.nn.functional as F


def train_local(model, loader, epochs, lr, momentum):
    """Entrena `epochs` pasadas sobre los datos locales con SGD. Devuelve loss y accuracy medios de la última época.

    El optimizador se crea de nuevo cada vez: en FL, cada ronda arranca desde el modelo global recibido.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for _ in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)
    return {"train_loss": total_loss / seen, "train_accuracy": correct / seen}
