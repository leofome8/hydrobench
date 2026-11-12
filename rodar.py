"""Execução pelo terminal, sem notebook (bom para deixar rodando a noite inteira no laptop).

Exemplos:
  python rodar.py --dados "dados/Vazões_Diárias_1931_2024.xlsx" --indices dados --saida resultados_v2
  python rodar.py --dados ... --experimentos E2,E3 --relatorio
  python rodar.py --saida resultados_v2 --so-relatorio

Cada execução fica salva na pasta de saída; se o computador desligar, rode o mesmo comando e ele continua.
O progresso vai para a tela e para <saida>/execucao.log.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydrobench as HB  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Protocolo v2: pré-treino hidrológico em reservatórios")
    ap.add_argument("--dados", default="dados/Vazões_Diárias_1931_2024.xlsx")
    ap.add_argument("--indices", default="dados")
    ap.add_argument("--saida", default="resultados_v2")
    ap.add_argument("--grupos", default=None, help="CSV opcional basin,group (cascatas)")
    ap.add_argument("--experimentos", default="E1,E3,E2,E3FT,E4,E5,E6,E7,E8",
                    help="ordem de execução; E3 = zero-shot dos Chronos, E3FT = pré-treino contínuo")
    ap.add_argument("--sementes", type=int, default=5)
    ap.add_argument("--passos", type=int, default=3000)
    ap.add_argument("--passos-chronos", type=int, default=2000)
    ap.add_argument("--sementes-chronos", type=int, default=5)
    ap.add_argument("--sem-t5-large", action="store_true")
    ap.add_argument("--relatorio", action="store_true", help="gera o relatório ao final")
    ap.add_argument("--so-relatorio", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.saida, exist_ok=True)
    logf = open(os.path.join(a.saida, "execucao.log"), "a", encoding="utf8")

    def log(*m):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + " ".join(str(x) for x in m)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"GPU: {p.name} | {p.total_memory / 1024**3:.1f} GB | CUDA {torch.version.cuda}")
    else:
        log("ATENÇÃO: GPU não encontrada; vai rodar na CPU (bem mais lento). Veja INSTALAR_LOCAL.md, passo 2.")

    cfg = HB.Config(data_path=a.dados, indices_dir=a.indices, out_dir=a.saida, groups_csv=a.grupos,
                    seeds=list(range(a.sementes)), steps=a.passos, chronos_ft_steps=a.passos_chronos,
                    chronos_ft_seeds=list(range(a.sementes_chronos)))
    if a.sem_t5_large:
        cfg.chronos_t5 = [m for m in cfg.chronos_t5 if not m.endswith("large")]
    st = HB.Study(cfg, log=log)
    log("Plano:\n" + st.plan().to_string(index=False))
    if not a.so_relatorio:
        steps = {"E1": st.run_e1, "E3": st.run_e3_zeroshot, "E2": st.run_e2, "E3FT": st.run_e3_finetune,
                 "E4": st.run_e4, "E5": st.run_e5, "E6": st.run_e6, "E7": st.run_e7, "E8": st.run_e8}
        for e in [x.strip().upper() for x in a.experimentos.split(",") if x.strip()]:
            if e not in steps:
                log(f"experimento desconhecido: {e}")
                continue
            t0 = time.time()
            log(f"==== {e} ====")
            steps[e]()
            log(f"==== {e} concluído em {(time.time() - t0) / 60:.1f} min ====")
    if a.relatorio or a.so_relatorio:
        _, _, _, path = HB.make_report(st, st.plan())
        log(f"Relatório: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
