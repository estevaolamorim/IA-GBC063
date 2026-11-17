"""Treina o DQN no ambiente de estoque, compara com políticas simples e gera os gráficos.

Uso:
    python treinar.py                       # 1500 episódios
    python treinar.py --episodios 2000 --semente 1
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from dqn import AgenteDQN
from estoque_env import (
    EstoqueEnv,
    avaliar_politica,
    criar_politica_aleatoria,
    criar_politica_fixa,
    criar_politica_reposicao,
    rodar_episodio,
)
from graficos import comparacao_politicas, curva_aprendizado, episodio_exemplo


def epsilon_do_episodio(ep, total, eps_ini=1.0, eps_fim=0.05, fracao=0.6):
    """Decai linearmente durante os primeiros 60% dos episódios."""
    meta = max(1, int(total * fracao))
    return max(eps_fim, eps_ini - (eps_ini - eps_fim) * ep / meta)


def treinar(env, agente, episodios, warmup=500):
    retornos = []
    for ep in range(episodios):
        eps = epsilon_do_episodio(ep, episodios)
        obs = env.reset()
        fim, total = False, 0.0
        while not fim:
            acao = agente.agir(obs, eps)
            obs2, r, fim, _ = env.step(acao)
            # O episódio acaba por limite de tempo (30 dias), não por um estado
            # final do problema: por isso o alvo continua usando Q(s') no último passo.
            agente.buffer.add(obs, acao, r, obs2)
            if len(agente.buffer) >= warmup:
                agente.aprender()
            obs = obs2
            total += r
        retornos.append(total)
        if (ep + 1) % 100 == 0:
            print(
                f"episódio {ep + 1:5d} | lucro médio (últimos 100): "
                f"{np.mean(retornos[-100:]):8.1f} | epsilon: {eps:.2f}"
            )
    return retornos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodios", type=int, default=1500)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--avaliacao", type=int, default=300, help="episódios na avaliação final")
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()

    random.seed(args.semente)
    np.random.seed(args.semente)
    torch.manual_seed(args.semente)
    os.makedirs(args.saida, exist_ok=True)

    env = EstoqueEnv(seed=args.semente)
    agente = AgenteDQN(env.dim_estado, env.n_acoes)

    print("Treinando...")
    retornos = treinar(env, agente, args.episodios)
    torch.save(agente.q.state_dict(), os.path.join(args.saida, "modelo_dqn.pt"))

    # ---- avaliação final (política gulosa, mesmas sementes para todos) ----
    politicas = {
        "Aleatória": criar_politica_aleatoria(),
        "Pedir 10 sempre": criar_politica_fixa(10),
        "Reposição até 25": criar_politica_reposicao(25),
        "DQN": lambda e, o: agente.agir(o, 0.0),
    }
    resultados = {}
    print("\nAvaliação final:")
    for nome, escolher in politicas.items():
        r = avaliar_politica(escolher, n_episodios=args.avaliacao)
        resultados[nome] = {
            "media": float(r.mean()),
            "desvio": float(r.std()),
            "erro_padrao": float(r.std() / np.sqrt(len(r))),
        }
        print(f"  {nome:18s} lucro médio = {r.mean():8.1f}  (± {r.std():.1f})")

    with open(os.path.join(args.saida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ---- gráficos ----
    curva_aprendizado(
        retornos,
        resultados["Reposição até 25"]["media"],
        os.path.join(args.saida, "curva_aprendizado.png"),
    )
    comparacao_politicas(resultados, os.path.join(args.saida, "comparacao_politicas.png"))
    registro = rodar_episodio(lambda e, o: agente.agir(o, 0.0))
    episodio_exemplo(registro, os.path.join(args.saida, "episodio_exemplo.png"))
    print(f"\nPronto! Arquivos salvos em '{args.saida}/'.")


if __name__ == "__main__":
    main()
