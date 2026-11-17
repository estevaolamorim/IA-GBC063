"""DQN (PyTorch): rede Q, replay buffer, rede alvo e epsilon-greedy."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RedeQ(nn.Module):
    """Estado -> valor Q de cada ação."""

    def __init__(self, dim_estado, n_acoes, ocultas=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_estado, ocultas),
            nn.ReLU(),
            nn.Linear(ocultas, ocultas),
            nn.ReLU(),
            nn.Linear(ocultas, n_acoes),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacidade):
        self.dados = deque(maxlen=capacidade)

    def add(self, s, a, r, s2):
        self.dados.append((s, a, r, s2))

    def sample(self, n):
        lote = random.sample(self.dados, n)
        s, a, r, s2 = zip(*lote)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.dados)


class AgenteDQN:
    def __init__(
        self,
        dim_estado,
        n_acoes,
        gamma=0.99,
        lr=1e-3,
        batch=64,
        capacidade_buffer=20_000,
        passos_atualiza_alvo=250,
        escala_recompensa=0.1,  # recompensas ~±100 -> ~±10 (treino mais estável)
    ):
        self.n_acoes = n_acoes
        self.gamma = gamma
        self.batch = batch
        self.passos_atualiza_alvo = passos_atualiza_alvo
        self.escala = escala_recompensa

        self.q = RedeQ(dim_estado, n_acoes)
        self.alvo = RedeQ(dim_estado, n_acoes)
        self.alvo.load_state_dict(self.q.state_dict())
        self.alvo.eval()

        self.otimizador = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacidade_buffer)
        self.passos = 0

    def agir(self, estado, epsilon):
        """Epsilon-greedy: com prob. epsilon explora, senão escolhe o maior Q."""
        if random.random() < epsilon:
            return random.randrange(self.n_acoes)
        with torch.no_grad():
            q = self.q(torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def aprender(self):
        if len(self.buffer) < self.batch:
            return None
        s, a, r, s2 = self.buffer.sample(self.batch)

        q_atual = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_prox = self.alvo(s2).max(dim=1).values
            meta = r * self.escala + self.gamma * q_prox

        perda = F.smooth_l1_loss(q_atual, meta)
        self.otimizador.zero_grad()
        perda.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.otimizador.step()

        self.passos += 1
        if self.passos % self.passos_atualiza_alvo == 0:
            self.alvo.load_state_dict(self.q.state_dict())
        return float(perda.item())
