"""Ambiente de gerenciamento de estoque (MDP) – Desafio 1.

Uma loja vende um único produto. No início de cada dia o agente decide
quantas unidades pedir ao fornecedor. O pedido chega no dia seguinte.

Estado   s = (estoque atual, pedido a caminho, dia da semana)
Ação     a ∈ {0, 5, 10, 15, 20} unidades
Recompensa do dia:
    +10 por unidade vendida
    -6  por unidade pedida
    -0,5 por unidade parada no estoque no fim do dia
    -4  por unidade que o cliente queria e não havia (venda perdida)
Episódio: 30 dias.
"""
import numpy as np


class EstoqueEnv:
    ACOES = (0, 5, 10, 15, 20)
    CAPACIDADE = 50
    HORIZONTE = 30
    PRECO_VENDA = 10.0
    CUSTO_COMPRA = 6.0
    CUSTO_ESTOQUE = 0.5
    PENALIDADE_FALTA = 4.0
    # demanda média (Poisson) de segunda (0) a domingo (6)
    DEMANDA_MEDIA = (5, 5, 6, 6, 8, 12, 10)

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n_acoes = len(self.ACOES)
        self.dim_estado = 2 + 7  # estoque, a caminho, dia da semana (one-hot)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """Começa um novo episódio e devolve o estado inicial."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.estoque = int(self.rng.integers(10, 31))
        self.transito = 0
        self.dia = int(self.rng.integers(0, 7))
        self.t = 0
        return self.observacao()

    def observacao(self):
        """Estado normalizado que entra na rede neural."""
        dia_onehot = np.zeros(7, dtype=np.float32)
        dia_onehot[self.dia] = 1.0
        base = np.array(
            [self.estoque / self.CAPACIDADE, self.transito / max(self.ACOES)],
            dtype=np.float32,
        )
        return np.concatenate([base, dia_onehot])

    # ------------------------------------------------------------------
    def step(self, idx_acao):
        """Executa um dia. Devolve (estado, recompensa, fim, info)."""
        pedido = self.ACOES[idx_acao]

        # 1) chega o pedido feito no dia anterior (o que passar da capacidade é perdido)
        self.estoque = min(self.estoque + self.transito, self.CAPACIDADE)

        # 2) demanda aleatória do dia
        demanda = int(self.rng.poisson(self.DEMANDA_MEDIA[self.dia]))
        vendas = min(demanda, self.estoque)
        faltas = demanda - vendas
        self.estoque -= vendas

        # 3) recompensa do dia
        recompensa = (
            self.PRECO_VENDA * vendas
            - self.CUSTO_COMPRA * pedido
            - self.CUSTO_ESTOQUE * self.estoque
            - self.PENALIDADE_FALTA * faltas
        )

        # 4) atualiza o estado
        self.transito = pedido
        self.dia = (self.dia + 1) % 7
        self.t += 1
        fim = self.t >= self.HORIZONTE

        info = {
            "pedido": pedido,
            "demanda": demanda,
            "vendas": vendas,
            "faltas": faltas,
            "estoque_fim": self.estoque,
        }
        return self.observacao(), float(recompensa), fim, info


# ----------------------------------------------------------------------
# Políticas de referência (para comparar com o DQN)
# ----------------------------------------------------------------------
def criar_politica_aleatoria(seed=0):
    rng = np.random.default_rng(seed)  # gerador próprio: não mexe na demanda do ambiente
    return lambda env, obs: int(rng.integers(0, env.n_acoes))


def criar_politica_fixa(unidades=10):
    idx = EstoqueEnv.ACOES.index(unidades)
    return lambda env, obs: idx


def criar_politica_reposicao(nivel=25):
    """Pede o suficiente para chegar perto de `nivel` (estoque + a caminho)."""
    acoes = np.array(EstoqueEnv.ACOES)

    def politica(env, obs):
        falta = max(0, nivel - (env.estoque + env.transito))
        return int(np.argmin(np.abs(acoes - falta)))

    return politica


# ----------------------------------------------------------------------
def avaliar_politica(escolher, n_episodios=300, semente_base=10_000):
    """Retorno total (lucro de 30 dias) de cada episódio.

    Todas as políticas usam as mesmas sementes, então enfrentam a mesma
    sequência de demanda: a comparação é justa.
    """
    env = EstoqueEnv()
    retornos = []
    for i in range(n_episodios):
        obs = env.reset(seed=semente_base + i)
        fim, total = False, 0.0
        while not fim:
            obs, r, fim, _ = env.step(escolher(env, obs))
            total += r
        retornos.append(total)
    return np.array(retornos)


def rodar_episodio(escolher, seed=123):
    """Registra um episódio dia a dia (para o gráfico de exemplo)."""
    env = EstoqueEnv()
    obs = env.reset(seed=seed)
    registro, fim = [], False
    while not fim:
        estoque_antes = env.estoque
        obs, r, fim, info = env.step(escolher(env, obs))
        info["recompensa"] = r
        info["estoque_inicio"] = estoque_antes
        registro.append(info)
    return registro
