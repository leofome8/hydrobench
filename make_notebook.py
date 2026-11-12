"""Gera o notebook Colab com o pacote embutido (código oculto em formulários)."""
import base64, io, json, os, zipfile, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(os.path.join(ROOT, "hydrobench"))):
        if f.endswith(".py"):
            z.write(os.path.join(ROOT, "hydrobench", f), "hydrobench/" + f)
PKG = base64.b64encode(buf.getvalue()).decode()


def code(src, title=None):
    lines = ([f"#@title {title}\n"] if title else []) + [l + "\n" for l in src.strip("\n").split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [],
            "source": lines}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in src.strip().split("\n")]}


UI = r'''
import html as _html, time as _time, pandas as _pd
from IPython.display import display, HTML
_CSS = """<style>
.hb{background:#121212;color:#EDEDED;font-family:Inter,Segoe UI,Arial,sans-serif;padding:16px 18px;border-radius:12px;border:1px solid #2A2A2A}
.hb h3{color:#FF7A00;margin:0 0 8px;font-size:17px}.hb .s{color:#9A9A9A;font-size:12.5px}
.hb table{border-collapse:collapse;font-size:12.5px;margin-top:8px;width:100%}.hb th{background:#262626;color:#FFB070;text-align:left;padding:5px 8px}
.hb td{padding:4px 8px;border-top:1px solid #2A2A2A;color:#EDEDED}.hb .k{display:inline-block;background:#1C1C1C;border-left:4px solid #FF7A00;
border-radius:6px;padding:8px 12px;margin:4px 8px 4px 0}.hb .k b{display:block;color:#FF7A00;font-size:20px}
.hb .log{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#CFCFCF;white-space:pre-wrap;max-height:220px;overflow:auto}
.hb img{background:#fff;border-radius:6px;padding:4px;max-width:100%}</style>"""

def painel(titulo, corpo="", sub=""):
    display(HTML(_CSS + f"<div class='hb'><h3>{titulo}</h3><div class='s'>{sub}</div>{corpo}</div>"))

def tabela(df, titulo, sub="", n=60, digits=3):
    if df is None or len(df) == 0:
        return painel(titulo, "<p class='s'>sem resultados ainda</p>", sub)
    t = df.head(n).to_html(index=False, border=0, na_rep="–", float_format=lambda v: f"{v:.{digits}f}")
    painel(titulo, t, sub)

def kpis(pares, titulo):
    painel(titulo, "".join(f"<span class='k'>{a}<b>{b}</b></span>" for a, b in pares))

class Monitor:
    """Substitui o print: mostra as últimas linhas de progresso num painel que se atualiza."""
    def __init__(self, titulo):
        self.titulo, self.linhas, self.t0 = titulo, [], _time.time()
        self.h = display(HTML(""), display_id=True)
        self(f"iniciando")
    def __call__(self, *a):
        self.linhas.append(_time.strftime("%H:%M:%S") + "  " + " ".join(str(x) for x in a))
        corpo = "<div class='log'>" + _html.escape("\n".join(self.linhas[-14:])) + "</div>"
        sub = f"{len(self.linhas)} eventos · {(_time.time()-self.t0)/60:.1f} min"
        self.h.update(HTML(_CSS + f"<div class='hb'><h3>{self.titulo}</h3><div class='s'>{sub}</div>{corpo}</div>"))

def contar(st, prefixo):
    import glob, os
    return len(glob.glob(os.path.join(st.cfg.out_dir, "preds", prefixo + "*.npz")))
'''

cells = [
    md("""# Pré-treino hidrológico em 161 reservatórios — protocolo v2
Responde ao parecer: normalização só com o passado, regimes de informação separados, split por grupos de dependência (cascatas) e
teste espaço-temporal, ablação com corpus real e sintético caracterizado, Chronos/Chronos-Bolt/Chronos-2 com e sem pré-treino
contínuo, várias sementes. **Use GPU** (Ambiente de execução → Alterar tipo → T4/L4/A100).

Tudo é salvo no Drive: se a sessão cair, rode de novo as mesmas células e ele continua de onde parou."""),
    code("""
!pip -q install "chronos-forecasting>=2.2" lightgbm peft pyarrow openpyxl
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA (ative a GPU)")""", "1 · Instalar dependências"),
    code("""
from google.colab import drive
drive.mount('/content/drive')
ARQUIVO_VAZOES = "/content/drive/MyDrive/reservatorios/Vazões_Diárias_1931_2024.xlsx"  #@param {type:"string"}
PASTA_INDICES = "/content/drive/MyDrive/reservatorios"  #@param {type:"string"}
PASTA_RESULTADOS = "/content/drive/MyDrive/reservatorios/resultados_v2"  #@param {type:"string"}
CSV_GRUPOS_CASCATA = ""  #@param {type:"string"}""", "2 · Arquivos (Drive)"),
    code(f"""
import base64, io, zipfile, sys, os
_PKG = "{PKG}"
os.makedirs("/content/hb", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(_PKG))).extractall("/content/hb")
sys.path.insert(0, "/content/hb")
for m in [k for k in sys.modules if k.startswith("hydrobench")]:
    del sys.modules[m]
import hydrobench as HB
from hydrobench import analysis as A, report as R
{UI}
painel("Pacote hydrobench carregado", "<p class='s'>dados · splits · janelas · corpora · modelos · Chronos · métricas · estatística · relatório</p>")""",
         "3 · Pacote do estudo (oculto)"),
    code("""
L_CONTEXTO = 24  #@param {type:"integer"}
H_HORIZONTE = 12  #@param {type:"integer"}
ANO_INICIAL = 1931  #@param {type:"integer"}
FIM_TREINO = "2004-12"  #@param {type:"string"}
INICIO_TESTE = "2005-01"  #@param {type:"string"}
N_FOLDS = 5  #@param {type:"integer"}
N_REGIMES = 5  #@param {type:"integer"}
LIMIAR_CORRELACAO_CASCATA = 0.85  #@param {type:"number"}
N_SEMENTES = 5  #@param {type:"integer"}
PASSOS_TREINO = 3000  #@param {type:"integer"}
PASSOS_PRETREINO_CHRONOS = 2000  #@param {type:"integer"}
SEMENTES_PRETREINO_CHRONOS = 5  #@param {type:"integer"}
INCLUIR_CHRONOS_T5_LARGE = True  #@param {type:"boolean"}

cfg = HB.Config(data_path=ARQUIVO_VAZOES, indices_dir=PASTA_INDICES or None, out_dir=PASTA_RESULTADOS,
                groups_csv=CSV_GRUPOS_CASCATA or None, start_year=ANO_INICIAL, L=L_CONTEXTO, H=H_HORIZONTE,
                train_end=FIM_TREINO, test_start=INICIO_TESTE, n_folds=N_FOLDS, n_regimes=N_REGIMES,
                group_corr_threshold=LIMIAR_CORRELACAO_CASCATA, seeds=list(range(N_SEMENTES)),
                steps=PASSOS_TREINO, chronos_ft_steps=PASSOS_PRETREINO_CHRONOS,
                chronos_ft_seeds=list(range(SEMENTES_PRETREINO_CHRONOS)))
if not INCLUIR_CHRONOS_T5_LARGE:
    cfg.chronos_t5 = [m for m in cfg.chronos_t5 if not m.endswith("large")]
mon = Monitor("Preparando dados")
st = HB.Study(cfg, log=mon)
qc = st.qc_report
kpis([("postos na planilha", len(qc)), ("mantidos no QC", int(qc.kept.sum())), ("no estudo", len(st.basins)),
      ("grupos de dependência", st.groups.nunique()), ("folds", cfg.n_folds)], "Corpus")
tabela(_pd.DataFrame(st.meta.groupby("regime_name").agg(
       bacias=("fold","size"), grupos=("group","nunique"), vazao_mediana=("mean_flow","median"), cv=("cv","median"))).reset_index(),
       "Regimes (assinatura do período de treino)")
g = st.groups.value_counts()
tabela(_pd.DataFrame({"grupo": g.index[:15], "bacias": g.values[:15],
                      "membros": [", ".join(st.groups[st.groups == k].index[:6]) for k in g.index[:15]]}),
       "Maiores grupos de dependência", "Bacias do mesmo grupo nunca ficam em lados diferentes da divisão. Confira se correspondem às cascatas reais.")
tabela(qc[~qc.kept], "Postos descartados no controle de qualidade", f"registro contíguo < {cfg.min_years} anos")
tabela(st.plan(), "Plano de execução", "Treinos/inferências por bloco. Rode as células na ordem; cada uma pode ser interrompida e retomada.")
img = R.fig_regimes(st)
painel("Assinatura sazonal por regime", f"<img src='data:image/png;base64,{img}'/>")""", "4 · Configurar e preparar dados"),
    code("""
mon = Monitor("E1 · Benchmark (todos os modelos, espacial e espaço-temporal)")
st.log = mon
st.run_e1()
kpis([("execuções E1 salvas", contar(st, "E1_"))], "E1 concluído")""", "5 · E1 Benchmark"),
    code("""
mon = Monitor("E3a · Chronos, Chronos-Bolt e Chronos-2 zero-shot (contexto L e longo)")
st.log = mon
st.run_e3_zeroshot()
kpis([("inferências salvas", contar(st, "E1_zs"))], "Zero-shot concluído")""", "6 · Modelos fundacionais zero-shot"),
    code("""
mon = Monitor("E2 · Ablação de corpus (hydro, M4, sintético genérico, sintético com estatística hidrológica)")
st.log = mon
st.run_e2()
tabela(_pd.read_csv(os.path.join(cfg.out_dir, "tables", "corpus_stats.csv")) if os.path.exists(os.path.join(cfg.out_dir, "tables", "corpus_stats.csv")) else None,
       "Caracterização dos corpora")""", "7 · E2 Ablação de corpus"),
    code("""
mon = Monitor("E3b · Pré-treino contínuo de Chronos-Bolt e Chronos-2 no pool hidrológico")
st.log = mon
st.run_e3_finetune()
kpis([("execuções salvas", contar(st, "E3_"))], "Pré-treino contínuo concluído")""", "8 · Domínio x escala nos modelos fundacionais"),
    code("""
mon = Monitor("E4 · Adaptação local (full, head, L2-SP, do zero)")
st.log = mon
st.run_e4()""", "9 · E4 Adaptação local"),
    code("""
mon = Monitor("E5 · Tamanho e diversidade do pool")
st.log = mon
st.run_e5()""", "10 · E5 Tamanho do pool"),
    code("""
mon = Monitor("E6 · Índices climáticos")
st.log = mon
st.run_e6()""", "11 · E6 Índices climáticos"),
    code("""
mon = Monitor("E7 · Contexto longo com a mesma informação para todos (L = 60 e 120 meses)")
st.log = mon
st.run_e7()
kpis([("execuções E7 salvas", contar(st, "E7_"))], "E7 concluído")""", "12 · E7 Contexto longo (controle decisivo)"),
    code("""
mon = Monitor("E8 · Modelos treinados com 240 e 360 meses de contexto (resposta ao parecer)")
st.log = mon
st.run_e8()
kpis([("execuções E8 salvas", contar(st, "E8_"))], "E8 concluído")""", "13 · E8 Contexto de 240 e 360 meses"),
    code("""
mon = Monitor("Métricas, testes estatísticos e figuras")
st.log = mon
st, df, T, caminho = HB.make_report(st, st.plan())
for nome in ["T2_benchmark_h1", "T4_ablacao_NSE_mediano_h1", "T4b_ablacao_wilcoxon", "T5_dominio_x_escala",
             "T10_contexto_NSE_mediano_h1", "T10b_melhor_dominio_vs_melhor_chronos",
             "T6_adaptacao_NSE_mediano_h1", "T7_tamanho_pool_NSE_mediano_h1", "T8_cheias_h1", "T9b_ganho_preditivo_indices",
             "S3_wilcoxon_vs_PAR-local", "S2g_ranks_medios_grupos_spatiotemporal", "T5c_E3_dispersao_sementes",
             "T10c_contexto_alvos_comuns_h1", "T11b_subconjuntos_sem_vizinho_correlacionado", "T11c_pareados_por_subconjunto",
             "T11d_ranks_com_agrupamentos_alternativos", "T4d_corpus_vs_hydro_IC_por_horizonte",
             "T10d_pares_todos_contextos_IC_equivalencia"]:
    if nome in T:
        tabela(T[nome], nome)
_dm = os.path.join(cfg.out_dir, "tables", "S4_diebold_mariano_vs_ref.csv")
if os.path.exists(_dm):
    tabela(_pd.read_csv(_dm), "S4_diebold_mariano_vs_ref")
for f in ["F2_CD_spatiotemporal", "F3_dominio_x_escala", "F4_horizonte", "F5_tamanho_pool", "F6_por_regime"]:
    p = os.path.join(cfg.out_dir, "figs", f + ".png")
    if os.path.exists(p):
        painel(f, f"<img src='data:image/png;base64,{base64.b64encode(open(p,'rb').read()).decode()}'/>")
painel("Relatório completo", f"<p class='s'>{caminho}<br>Tabelas em CSV: {cfg.out_dir}/tables · Figuras PNG/PDF: {cfg.out_dir}/figs</p>")""",
         "14 · Resultados e relatório (v3, com T10d)"),
    code("""
mon = Monitor("Grupos de dependência só com dados anteriores a 2005 (parecer 3)")
st.log = mon
if "df" not in globals():
    df = A.compute_metrics(st)
T12 = A.pre2005_sensitivity(st, df)
for nome, tab in T12.items():
    tabela(tab, nome)
# versões das bibliotecas e commit de cada checkpoint (para o artigo e o repositório)
import importlib.metadata as _im
_v = []
for _p in ["torch", "chronos-forecasting", "transformers", "peft", "lightgbm", "scikit-learn", "scipy", "pandas", "numpy"]:
    try:
        _v.append(dict(tipo="biblioteca", nome=_p, versao=_im.version(_p)))
    except Exception:
        _v.append(dict(tipo="biblioteca", nome=_p, versao="não instalada"))
import sys as _sys
_v.append(dict(tipo="biblioteca", nome="python", versao=_sys.version.split()[0]))
try:
    from huggingface_hub import model_info as _mi
    for _m in sorted(set(cfg.chronos_t5 + cfg.chronos_bolt + cfg.chronos2)):
        try:
            _v.append(dict(tipo="checkpoint", nome=_m, versao=_mi(_m).sha))
        except Exception as _e:
            _v.append(dict(tipo="checkpoint", nome=_m, versao=f"erro: {_e}"))
except Exception:
    pass
_vers = _pd.DataFrame(_v)
_vers.to_csv(os.path.join(cfg.out_dir, "tables", "environment_versions.csv"), index=False)
tabela(_vers, "environment_versions", n=40)
# métricas por bacia (média das sementes) para publicar no repositório
_pb = A.seed_mean(df, ["exp", "setting", "info_regime", "label", "L"] if "L" in df.columns else ["exp", "setting", "info_regime", "label"])
_cols = [c for c in ["exp", "setting", "info_regime", "label", "L", "basin", "h", "NSE", "NSE_vs_clim", "KGE", "PBIAS",
                     "logNSE", "nRMSE", "NSE_seed_sd"] if c in _pb.columns]
_arq = os.path.join(cfg.out_dir, "tables", "per_basin_metrics_seedmean.csv.gz")
_pb[_cols].to_csv(_arq, index=False, float_format="%.5f", compression="gzip")
kpis([("linhas por bacia", len(_pb)), ("tamanho (MB)", round(os.path.getsize(_arq) / 1e6, 1))], "Métricas por bacia exportadas")
painel("T12 salvo", f"<p class='s'>CSVs em {cfg.out_dir}/tables: T12a a T12d, environment_versions.csv e "
       f"per_basin_metrics_seedmean.csv.gz. Nada foi re-treinado: só métricas já salvas.</p>")""",
         "15 · Sensibilidade: grupos construídos só até 2004"),
]

LOCAL_HEAD = [
    md("""# Pré-treino hidrológico em 161 reservatórios — protocolo v2 (GPU local)
Versão para rodar no Jupyter ou no VS Code, na GPU do próprio computador. Veja INSTALAR_LOCAL.md para preparar o ambiente.
Tudo é salvo na pasta de resultados: se o computador desligar ou hibernar, rode de novo as mesmas células e ele continua.
Para aproveitar o que já rodou no Colab, copie a pasta resultados_v2 do Drive para cá e aponte PASTA_RESULTADOS para ela."""),
    code(f"""
import base64, io, zipfile, sys, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_PKG = "{PKG}"
_DEST = os.path.abspath("_hydrobench_pkg")
os.makedirs(_DEST, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(_PKG))).extractall(_DEST)
sys.path.insert(0, _DEST)
for m in [k for k in sys.modules if k.startswith("hydrobench")]:
    del sys.modules[m]
import torch, pandas, numpy
import hydrobench as HB
from hydrobench import analysis as A, report as R
{UI}
_lin = [("Python", sys.version.split()[0]), ("PyTorch", torch.__version__), ("pandas", pandas.__version__)]
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    _cap = torch.cuda.get_device_capability(0)
    _lin += [("GPU", _p.name), ("memória da GPU", f"{{_p.total_memory/1024**3:.1f}} GB"),
             ("CUDA", torch.version.cuda), ("precisão dos Chronos", "bfloat16" if _cap[0] >= 8 else "float32")]
    kpis(_lin, "Ambiente local")
    if _p.total_memory / 1024**3 < 7.5:
        painel("Aviso de memória", "<p class='s'>GPU com menos de 8 GB: os Chronos maiores reduzem o lote sozinhos se faltar memória "
               "(cada redução aparece no painel e fica registrada nos metadados). O pré-treino contínuo do Chronos-Bolt-base é o bloco mais exigente.</p>")
else:
    kpis(_lin, "Ambiente local")
    painel("GPU não encontrada", "<p class='s'>O PyTorch instalado não enxerga a placa. Siga o passo 2 do INSTALAR_LOCAL.md "
           "(PyTorch com CUDA) e reinicie o kernel. Sem GPU tudo roda, mas muito mais devagar.</p>")""", "1 · Ambiente e pacote"),
    code("""
ARQUIVO_VAZOES = r"dados/Vazões_Diárias_1931_2024.xlsx"  #@param {type:"string"}
PASTA_INDICES = r"dados"  #@param {type:"string"}
PASTA_RESULTADOS = r"resultados_v2"  #@param {type:"string"}
CSV_GRUPOS_CASCATA = ""  #@param {type:"string"}
for _p in (ARQUIVO_VAZOES, PASTA_INDICES):
    if not os.path.exists(_p):
        painel("Arquivo não encontrado", f"<p class='s'>{os.path.abspath(_p)}</p>")
painel("Arquivos", f"<p class='s'>vazões: {os.path.abspath(ARQUIVO_VAZOES)}<br>índices: {os.path.abspath(PASTA_INDICES)}<br>"
       f"resultados: {os.path.abspath(PASTA_RESULTADOS)}</p>")""", "2 · Arquivos"),
]
# células 4 em diante iguais às do Colab
import copy as _copy, re as _re
LOCAL_CELLS = LOCAL_HEAD
for _c in cells[4:]:
    _c = _copy.deepcopy(_c)
    if _c["source"] and _c["source"][0].startswith("#@title"):
        _c["source"][0] = _re.sub(r"#@title (\d+)", lambda m: f"#@title {int(m.group(1)) - 1}", _c["source"][0])
    LOCAL_CELLS.append(_c)

nb_local = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "cells": LOCAL_CELLS}
json.dump(nb_local, open("protocolo_v2_local.ipynb", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("protocolo_v2_local.ipynb", os.path.getsize("protocolo_v2_local.ipynb") // 1024, "KB")

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": cells}
out = sys.argv[1] if len(sys.argv) > 1 else "protocolo_v2_reservatorios.ipynb"
json.dump(nb, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(out, os.path.getsize(out) // 1024, "KB")
