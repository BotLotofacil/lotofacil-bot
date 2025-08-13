# =========================
# Bloco 1 — IMPORTS & SETUP
# =========================

# Stdlib
import os
import sys
import signal 
import traceback
import shutil
import pickle
import random
from io import BytesIO
from datetime import datetime
from collections import Counter, defaultdict
import logging
import warnings
from typing import Optional, Dict, List, Tuple, Set
import psutil
from threading import Lock

# Configuração segura do psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil não instalado - monitoramento desativado")

# ---- Logging precisa estar definido antes de qualquer uso de logger ----
warnings.filterwarnings("ignore", message="oneDNN custom operations are on")
logging.basicConfig(
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Terceiros
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
try:
    import tensorflow as tf
except ImportError:
    logger.critical("TensorFlow não instalado!")
    raise

# Matplotlib: backend seguro para servidor/headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Telegram
import telegram
print("PTB version:", telegram.__version__)
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext

# Núcleo preciso (score + GRASP + diversidade)
from apostas_engine import gerar_apostas as gerar_apostas_precisas
from apostas_engine import Config as ApostaConfig

# ==== Barra de carregamento para /aposta (JobQueue) ====
from time import time as _now
import time

# Configuração do timeout/pool para o bot (compatível com PTB v13.x)
REQUEST_KWARGS = {
    "connect_timeout": int(os.getenv("TELEGRAM_CONNECT_TIMEOUT", 10)),
    "read_timeout": int(os.getenv("TELEGRAM_READ_TIMEOUT", 20)),
    "con_pool_size": int(os.getenv("TELEGRAM_CON_POOL_SIZE", 8)),
    # REMOVIDO: "pool_timeout" (não existe no v13)
}

_PROGRESS_FRAMES = ["▁","▃","▄","▅","▆","▇","█","▇","▆","▅","▄","▃"]

def _progress_tick(context):
    data = context.job.context
    msg = data.get("msg")
    i = data.get("i", 0)
    t0 = data.get("t0", _now())
    frame = _PROGRESS_FRAMES[i % len(_PROGRESS_FRAMES)]
    elapsed = int(_now() - t0)
    try:
        bar = frame * 20
        msg.edit_text(
            f"⏳ <b>Gerando apostas...</b>\n{bar}\n<i>{elapsed}s</i>",
            parse_mode='HTML'
        )
    except Exception:
        pass
    data["i"] = i + 1

def _start_progress(context, chat_id):
    """Cria a mensagem e agenda atualizações periódicas."""
    msg = context.bot.send_message(chat_id, "⏳ Iniciando geração...", parse_mode='HTML')
    job = context.job_queue.run_repeating(
        _progress_tick,
        interval=0.7,
        first=0.0,
        context={"msg": msg, "i": 0, "t0": _now()}
    )
    return msg, job

def _stop_progress(job, msg, final_text):
    """Para as atualizações e finaliza a mensagem de progresso."""
    try:
        job.schedule_removal()
    except Exception:
        pass
    try:
        msg.edit_text(final_text, parse_mode='HTML')
    except Exception:
        pass

def handler(signum, frame):
    raise TimeoutError("Operação excedeu o tempo limite")
signal.signal(signal.SIGALRM, handler)  # Configura o handler

# ==== fim helpers ====

# ================================
# Bloco 2 — Constantes & Bootstrap
# ================================

AVISO_LEGAL = (
    "<b>AVISO LEGAL E DE PRIVACIDADE</b>\n"
    "• Este bot da Lotofácil IA é uma ferramenta analítica para entretenimento. Nenhum prêmio é garantido.\n"
    "• O único dado coletado é o ID numérico do Telegram, utilizado exclusivamente para autenticação e controle de acesso.\n"
    "• Não são coletados ou compartilhados nomes, fotos, telefone, conversas ou qualquer outro dado pessoal.\n"
    "• Nenhuma informação fornecida pelos usuários é compartilhada, vendida ou transmitida a terceiros.\n"
    "• Ao utilizar este bot, você concorda com estas condições.\n"
)

MANUAL_USUARIO = (
    "<b>Bem-vindo ao Bot Lotofácil IA!</b>\n\n"
    "<b>Como funciona:</b>\n"
    "Você poderá gerar apostas inteligentes, acompanhar tendências e consultar análises estatísticas da Lotofácil direto pelo Telegram.\n\n"
    "<b>Comandos disponíveis:</b>\n"
    "/aposta     - Recebe apostas inteligentes sugeridas pelo sistema\n"
    "/tendencia  - Recebe aposta baseada nas tendências recentes\n"
    "/analise    - Consulta estatísticas e gráficos\n"
    "/status     - Consulta status geral do sistema e últimos resultados\n"
    "/meuid      - Mostra seu ID do Telegram\n\n"
    "<b>Privacidade:</b>\n"
    "Seu ID do Telegram é usado somente para controle de acesso.\n"
    "Nenhum dado pessoal é coletado, compartilhado ou vendido.\n\n"
    "<b>Aviso legal:</b>\n"
    "O bot é uma ferramenta de análise e entretenimento. Não garante prêmios ou lucro em apostas.\n"
)

def backup_csv() -> None:
    """Cria um backup timestampado do CSV principal, se existir."""
    origem = "lotofacil_historico.csv"
    if os.path.exists(origem):
        destino = f"lotofacil_historico_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            shutil.copy2(origem, destino)
            logger.info(f"Backup automático criado: {destino}")
        except Exception as e:
            logger.warning(f"Falha ao criar backup do CSV: {e}")

# Configuração inicial de warnings e logging
warnings.filterwarnings("ignore", message="oneDNN custom operations are on")
logging.basicConfig(
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Token do Telegram via variável de ambiente (Railway/Render/etc.)
TOKEN = os.getenv("TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("TOKEN não definido no ambiente.")

# Diretório persistente do container (ex.: /data). Se existir, usamos como CWD.
DATA_DIR = os.getenv("DATA_DIR")
if DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)

    # Copia arquivos iniciais para o volume, se ausentes
    for _fn in ("lotofacil_historico.csv", "whitelist.txt", "modelo_lotofacil_avancado.keras"):
        src = os.path.join(os.getcwd(), _fn)
        dst = os.path.join(DATA_DIR, _fn)
        if (not os.path.exists(dst)) and os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.warning(f"Não foi possível copiar {src} -> {dst}: {e}")

    # Remove modelo legado .h5 se existir no volume
    try:
        old_h5 = os.path.join(DATA_DIR, "modelo_lotofacil_avancado.h5")
        if os.path.exists(old_h5):
            os.remove(old_h5)
    except Exception:
        pass

    # Passa a trabalhar dentro do volume
    try:
        os.chdir(DATA_DIR)
    except Exception as e:
        logger.warning(f"Falha ao trocar para DATA_DIR ({DATA_DIR}): {e}")

# Admins e rate-limit
ADMIN_USER_IDS: List[int] = [5344714174]  # ajuste conforme necessário

_rate_limit_map: Dict[int, Dict[str, float]] = {}

def rate_limit(update: Update, comando: str, segundos: int = 8) -> bool:
    """
    Anti-spam por usuário/comando.
    Retorna False se deve bloquear a execução (muito cedo); True caso possa prosseguir.
    """
    user_id = update.effective_user.id
    agora = _now()
    user_map = _rate_limit_map.setdefault(user_id, {})
    ultimo = user_map.get(comando, 0.0)
    if agora - ultimo < segundos:
        try:
            update.message.reply_text("⏳ Aguarde alguns segundos antes de usar novamente.")
        except Exception:
            pass
        return False
    user_map[comando] = agora
    return True

# Adicione após as constantes existentes:
_RESOURCE_LOCK = Lock()
_MAX_MEMORY_ALERT = 85  # % de uso de memória para alerta
_MIN_MEMORY_ALERT = 1024  # MB livres mínimos para alerta

# ======================================
# Bloco 3 — SecurityManager & DataFetcher
# ======================================

class SecurityManager:
    """Gerencia whitelist e permissões de administrador."""
    def __init__(self) -> None:
        self.whitelist: Set[int] = set()
        self.admins: Set[int] = set(ADMIN_USER_IDS)
        self.load_whitelist()

    def load_whitelist(self, file: str = "whitelist.txt") -> None:
        """Carrega/recupera whitelist do disco (id por linha)."""
        try:
            if os.path.exists(file):
                with open(file, "r", encoding="utf-8") as f:
                    self.whitelist = {
                        int(line.strip())
                        for line in f
                        if line.strip().isdigit()
                    }
            else:
                open(file, "a", encoding="utf-8").close()
        except Exception as e:
            logger.error(f"Erro ao carregar whitelist: {e}")

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.admins

    def is_authorized(self, user_id: int) -> bool:
        """Autorizado se estiver na whitelist ou for admin."""
        return user_id in self.whitelist or self.is_admin(user_id)

class ResourceMonitor:
    """Monitoramento avançado de recursos do sistema."""
    @staticmethod
    def get_system_stats() -> Dict[str, float]:
        if not PSUTIL_AVAILABE:
            return {}
            
        with _RESOURCE_LOCK:
            try:
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                cpu = psutil.cpu_percent(interval=1)
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                
                return {
                    'cpu_percent': cpu,
                    'mem_total': mem.total / (1024 ** 3),  # GB
                    'mem_used': mem.used / (1024 ** 3),
                    'mem_percent': mem.percent,
                    'swap_used': swap.used / (1024 ** 3),
                    'disk_used': disk.used / (1024 ** 3),
                    'disk_percent': disk.percent,
                    'bytes_sent': net_io.bytes_sent / (1024 ** 2),  # MB
                    'bytes_recv': net_io.bytes_recv / (1024 ** 2),
                    'process_mem': psutil.Process().memory_info().rss / (1024 ** 2)  # MB
                }
            except Exception as e:
                logger.error(f"Erro ao coletar métricas: {str(e)}")
                return {}

    @staticmethod
    def log_resource_usage(context: CallbackContext) -> None:
        """Log detalhado com alertas condicionais."""
        stats = ResourceMonitor.get_system_stats()
        if not stats:
            return

        alert = ""
        if stats['mem_percent'] > _MAX_MEMORY_ALERT:
            alert = f" ⚠️ ALERTA: Memória acima de {_MAX_MEMORY_ALERT}%"
        elif (stats['mem_total'] - stats['mem_used']) < (_MIN_MEMORY_ALERT / 1024):
            alert = f" ⚠️ ALERTA: Menos de {_MIN_MEMORY_ALERT}MB livres"

        message = (
            f"📊 Resource Monitor | "
            f"CPU: {stats['cpu_percent']:.1f}% | "
            f"Mem: {stats['mem_percent']:.1f}% ({stats['mem_used']:.2f}GB/{stats['mem_total']:.2f}GB) | "
            f"Process: {stats['process_mem']:.2f}MB | "
            f"Disk: {stats['disk_percent']:.1f}%{alert}"
        )
        
        logger.info(message)
        
        if alert and hasattr(context, 'bot') and ADMIN_USER_IDS:
            for admin_id in ADMIN_USER_IDS:
                try:
                    context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🚨 ALERTA DE RECURSOS\n{message}",
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"Falha ao enviar alerta para admin {admin_id}: {str(e)}")

class DataFetcher:
    """Obtém último resultado da Lotofácil com fallback entre fontes."""
    API_URLS = [
        "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil",
        "https://api-loterias.herokuapp.com/api/v1/lotofacil",
    ]

    @staticmethod
    def fetch_data(url: str, timeout: int = 10) -> Optional[Dict]:
        """Baixa JSON de uma URL com timeout e tratamento de erros."""
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"HTTP {resp.status_code} ao acessar {url}")
        except Exception as e:
            logger.warning(f"Falha ao acessar {url}: {e}")
        return None

    @classmethod
    def get_latest_data(cls) -> Optional[Dict]:
        """Percorre as URLs até conseguir um payload válido."""
        for url in cls.API_URLS:
            data = cls.fetch_data(url)
            if data and cls.validate_data(data):
                return data
        return None

    @staticmethod
    def validate_data(data: Dict) -> bool:
        """
        Valida estrutura básica do payload:
        - possui chaves numero, data, dezenas
        - dezenas é lista de 15 inteiros entre 1 e 25
        """
        try:
            if not all(k in data for k in ("numero", "data", "dezenas")):
                return False
            dezenas = data["dezenas"]
            if not isinstance(dezenas, list) or len(dezenas) != 15:
                return False
            return all(1 <= int(n) <= 25 for n in dezenas)
        except Exception:
            return False

class BotLotofacil:
    def __init__(self):
        self.security = SecurityManager()
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.modelo_path = 'modelo_lotofacil_avancado.keras'
        self.dados = self.carregar_dados()
        if self.dados is not None:
            self.analisar_dados()
            self.modelo = self.construir_modelo()
        else:
            logger.error("Falha ao carregar dados. Verifique sua conexão com a internet.")

        # Núcleo preciso
        self.cfg_precisa = ApostaConfig()
        self.ultima_geracao_precisa: List[List[int]] = []

        # Saúde/observabilidade do engine preciso
        self.precise_enabled: bool = True
        self.precise_fail_count: int = 0
        self.precise_last_error: Optional[str] = None

        # Autoteste simples no startup
        try:
            _ = self._teste_engine_precisa_startup()
            self.precise_enabled = True
            self.precise_fail_count = 0
            self.precise_last_error = None
            logger.info("Engine precisa OK no startup.")
        except Exception as e:
            self.precise_enabled = False
            self.precise_last_error = str(e)
            logger.warning(f"Engine precisa desativado no startup: {e}")

    # -------------------------
    # Dados / preparação
    # -------------------------
    def carregar_dados(self, atualizar: bool = False) -> Optional[pd.DataFrame]:
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        if not atualizar and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache corrompido. Recriando... Erro: {str(e)}")

        if not os.path.exists('lotofacil_historico.csv'):
            logger.error("Arquivo lotofacil_historico.csv não encontrado.")
            return None

        df = pd.read_csv('lotofacil_historico.csv')
        processed = self.preprocessar_dados(df) if df is not None else None

        if processed is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed, f)
                processed.to_csv('lotofacil_historico.csv', index=False)
            except Exception as e:
                logger.error(f"Falha ao salvar cache ou CSV: {str(e)}")

        return processed

    def preprocessar_dados(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            required_cols = ['data'] + [f'B{i}' for i in range(1,16)]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Colunas obrigatórias faltantes. Esperado: {required_cols}")
                return None

            try:
                df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='raise')
            except Exception:
                try:
                    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='raise')
                except Exception:
                    df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')

            if df['data'].isnull().any():
                logger.warning(f"Linhas descartadas por data inválida: {df['data'].isnull().sum()}")
                df = df.dropna(subset=['data'])

            if 'numero' in df.columns:
                df['numero'] = df['numero'].astype(int)
            elif 'concurso' in df.columns:
                df['numero'] = df['concurso'].astype(int)
            else:
                df['numero'] = range(1, len(df)+1)

            for i in range(1, 16):
                df[f'B{i}'] = df[f'B{i}'].astype(int)

            if 'numero' in df.columns:
                df = df.sort_values('numero').reset_index(drop=True)
            else:
                df = df.sort_values('data').reset_index(drop=True)

            for rep in range(1, 6):
                repetidos = []
                for idx, row in df.iterrows():
                    if idx < rep:
                        repetidos.append(0)
                    else:
                        nums_atual = {row[f'B{i}'] for i in range(1, 16)}
                        nums_anterior = {df.iloc[idx - rep][f'B{i}'] for i in range(1, 16)}
                        repetidos.append(len(nums_atual & nums_anterior))
                df[f'repetidos_{rep}'] = repetidos

            cols_retorno = ['numero', 'data'] + [f'B{i}' for i in range(1,16)] + [f'repetidos_{j}' for j in range(1,6)]
            return df[cols_retorno]

        except Exception as e:
            logger.error(f"Falha crítica no pré-processamento: {str(e)}\nDados recebidos:\n{df.head()}")
            return None

    def analisar_dados(self) -> None:
        contagem = Counter(self.dados.filter(like='B').values.flatten())
        self.frequencias = Counter({n: contagem.get(n, 0) for n in range(1, 26)})
        self.coocorrencias = self.calcular_coocorrencia()
        self.sequencias_iniciais = self.analisar_sequencias_iniciais()
        self.clusters = self.identificar_clusters()

    def calcular_coocorrencia(self) -> np.ndarray:
        cooc = np.zeros((25, 25))
        N = len(self.dados)
        for i in range(1, N):
            nums_atual = set(self.dados.iloc[i][[f'B{j}' for j in range(1,16)]].values)
            nums_anterior = set(self.dados.iloc[i-1][[f'B{k}' for k in range(1,16)]].values)
            dist = max(1, N - i)
            w = 1.0 / (dist ** 0.5)
            for num1 in nums_atual:
                for num2 in nums_anterior:
                    cooc[num1-1, num2-1] += w
        return cooc

    def analisar_sequencias_iniciais(self) -> Dict[Tuple[int, int, int], int]:
        sequencias = defaultdict(int)
        for _, row in self.dados.iterrows():
            nums_ordenados = sorted(row[[f'B{i}' for i in range(1,16)]].values)
            chave = tuple(nums_ordenados[:3])
            sequencias[chave] += 1
        return sequencias

    def identificar_clusters(self) -> Dict[int, List[int]]:
        cache_file = os.path.join(self.cache_dir, "clusters_cache.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache de clusters corrompido. Recriando... Erro: {str(e)}")

        dados_cluster = self.dados[[f'B{i}' for i in range(1, 16)]]
        kmeans = KMeans(n_clusters=4, random_state=42).fit(dados_cluster)

        clusters: Dict[int, List[int]] = {i: [] for i in range(4)}
        for num in range(1, 26):
            sample = pd.DataFrame([[num] * 15], columns=dados_cluster.columns)
            cluster = kmeans.predict(sample)[0]
            clusters[cluster].append(num)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(clusters, f)
        except Exception as e:
            logger.error(f"Falha ao salvar cache de clusters: {str(e)}")

        return clusters

    def construir_modelo(self) -> Optional[tf.keras.Model]:
        if os.path.exists(self.modelo_path):
            try:
                return tf.keras.models.load_model(self.modelo_path)
            except Exception as e:
                logger.warning(f"Modelo corrompido/incompatível. Recriando... Erro: {e}")
                try:
                    os.remove(self.modelo_path)
                except Exception:
                    pass

        X, y = self.preparar_dados_treinamento()

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(25, activation='sigmoid'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.modelo_path, save_best_only=True)

        n = len(X)
        if n < 10:
            model.fit(X, y, epochs=50, batch_size=32, callbacks=[early, checkpoint], shuffle=False, verbose=0)
        else:
            cut = int(n * 0.8)
            X_train, y_train = X[:cut], y[:cut]
            X_val, y_val = X[cut:], y[cut:]
            model.fit(X_train, y_train, epochs=50, batch_size=32,
                      validation_data=(X_val, y_val), callbacks=[early, checkpoint],
                      shuffle=False, verbose=0)

        try:
            model.save(self.modelo_path)
        except Exception:
            pass
        return model

    def preparar_dados_treinamento(self) -> Tuple[np.ndarray, np.ndarray]:
        dados_numeros = self.dados[[f'B{i}' for i in range(1,16)]].values
        X, y = [], []
        janela = 10
        for i in range(janela, len(dados_numeros)):
            seq_numeros = dados_numeros[i-janela:i]
            seq_bin = np.zeros((janela, 25))
            for j in range(janela):
                for num in seq_numeros[j]:
                    seq_bin[j, num-1] = 1

            features_extras = []
            for k in range(1,6):
                col_name = f'repetidos_{k}'
                features_extras.append(self.dados.iloc[i-1][col_name] if col_name in self.dados.columns else 0)
            features_extras = np.array(features_extras).reshape(1, -1)

            X_seq = np.concatenate([seq_bin, np.tile(features_extras, (janela, 1))], axis=1)

            target = np.zeros(25)
            for num in dados_numeros[i]:
                target[num-1] = 1

            X.append(X_seq)
            y.append(target)
        return np.array(X), np.array(y)

    # -------------------------
    # Helpers de regras/viés
    # -------------------------
    def _maior_sequencia_consecutivos(self, aposta: List[int]) -> int:
        if not aposta:
            return 0
        nums = sorted(aposta)
        best = cur = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    def _tem_prefixo_123(self, aposta: List[int]) -> bool:
        s = set(aposta)
        return {1, 2, 3}.issubset(s)

    def _diferenca_minima(self, ap: List[int], existentes: List[List[int]], min_diff: int = 4) -> bool:
        s = set(ap)
        for e in existentes:
            comum = len(s & set(e))
            if 15 - comum < min_diff:
                return False
        return True

    def _valida_regras_basicas(self, ap: List[int]) -> bool:
        """Checagem rápida de sanidade das apostas."""
        if len(ap) != 15 or len(set(ap)) != 15:
            return False
        if any(n < 1 or n > 25 for n in ap):
            return False
        pares = sum(1 for n in ap if n % 2 == 0)
        soma = sum(ap)
        if not (5 <= pares <= 10):
            return False
        if not (160 <= soma <= 220):
            return False
        if self._maior_sequencia_consecutivos(ap) >= 5:
            return False
        return True

    def _repara(self, ap: List[int], rng: random.Random) -> List[int]:
        """Ajusta aposta para 15 únicos e dentro das faixas (pares/soma), com limite de tentativas."""
        ap = sorted(set(int(x) for x in ap if 1 <= int(x) <= 25))
        # completa até 15
        while len(ap) < 15:
            n = rng.randrange(1, 26)
            if n not in ap:
                ap.append(n)
        ap = sorted(ap[:15])

        tent = 0
        while not self._valida_regras_basicas(ap) and tent < 30:
            pares = sum(1 for n in ap if n % 2 == 0)
            soma = sum(ap)
            # decide quem tirar
            if pares > 10:
                cand = [n for n in ap if n % 2 == 0]
            elif pares < 5:
                cand = [n for n in ap if n % 2 == 1]
            elif soma > 220:
                cand = [n for n in ap if n > 13]
            elif soma < 160:
                cand = [n for n in ap if n < 13]
            else:
                cand = ap[:]
            sai = rng.choice(cand)
            ap.remove(sai)

            # escolhe entrada que melhora cobertura e regras
            fora = [n for n in range(1, 26) if n not in ap]
            rng.shuffle(fora)
            entra = fora[0]
            ap.append(entra)
            ap = sorted(ap)
            tent += 1
        return ap

    def _count_low_mid_high(self, ap: List[int]) -> Tuple[int, int, int]:
        """Conta quantos números caem nas faixas: low(1-8), mid(9-17), high(18-25)."""
        low = sum(1 for n in ap if 1 <= n <= 8)
        mid = sum(1 for n in ap if 9 <= n <= 17)
        high = sum(1 for n in ap if 18 <= n <= 25)
        return low, mid, high

    def _score_balance(self, ap: List[int]) -> float:
        """
        Score de balanceamento por faixas:
        - bonifica distribuição próxima de (5,5,5)
        - penaliza excesso de low (viés típico)
        - pequeno bônus por cobrir extremos (ex.: ter >=1 em 1–3 e >=1 em 23–25)
        """
        low, mid, high = self._count_low_mid_high(ap)
        target = (5, 5, 5)
        # distância L1 ao alvo (quanto menor, melhor)
        dist = abs(low - target[0]) + abs(mid - target[1]) + abs(high - target[2])
        base = 15.0 - dist  # 0..15 (quanto mais perto do alvo, maior)
        # penaliza excesso de low
        penalty_low = max(0, low - 8) * 2.5  # >8 baixos cai forte
        # bônus se cobre extremidades
        bonus_extremos = 2.0 if any(n <= 3 for n in ap) and any(n >= 23 for n in ap) else 0.0
        return base - penalty_low + bonus_extremos

    # -------------------------
    # Motores de geração
    # -------------------------
    def _mutacao_suave(
        self,
        aposta: List[int],
        rng: random.Random,
        cobertura_execucao: Counter,
        max_trocas: int = 3,
        tol_score: float = 0.5,
        p_aplicar: float = 0.8,
    ) -> List[int]:
        """
        Mutações pequenas guiadas pelo score (inclui _score_balance).
        Aceita troca se melhora o score ou piora pouco mas melhora cobertura.
        Limites soft por faixa para evitar viés.
        """
        if rng.random() > p_aplicar:
            return sorted(aposta[:])

        base = sorted(aposta[:])
        score_orig = float(self.avaliar_aposta_ga(base)[0])

        # ordem de remoção: mais pressionados (freq histórica + já cobertos)
        pressao_remover = {n: self.frequencias.get(n, 0) + cobertura_execucao[n] for n in base}
        cand_remover = sorted(base, key=lambda n: (-pressao_remover[n], n))

        # ordem de inclusão: preferência por baixa frequência histórica + leve cooc
        fora = [n for n in range(1, 26) if n not in base]
        vant_incluir = {
            n: -self.frequencias.get(n, 0) + float(np.sum(self.coocorrencias[n-1, [x-1 for x in base]])) * 0.03
            for n in fora
        }
        cand_incluir = sorted(fora, key=lambda n: (vant_incluir[n], -n), reverse=True)

        tentativa = base[:]
        trocas = 0
        irem = 0
        iinc = 0

        def _ok_faixas(ap):
            low, mid, high = self._count_low_mid_high(ap)
            if low > 8:        # não deixa estourar baixo
                return False
            if mid == 0 or high == 0:  # evita colapsar uma faixa
                return False
            return True

        while trocas < max_trocas and irem < len(cand_remover) and iinc < len(cand_incluir):
            sai = cand_remover[irem]; irem += 1
            entra = cand_incluir[iinc]; iinc += 1
            if entra in tentativa:
                continue

            nova = [x for x in tentativa if x != sai] + [entra]
            nova.sort()

            # regras básicas e faixas
            if not self._valida_regras_basicas(nova):
                continue
            if not _ok_faixas(nova):
                continue
            # anti {1,2,3}
            if {1, 2, 3}.issubset(set(nova)):
                continue

            score_novo = float(self.avaliar_aposta_ga(nova)[0])

            # bônus de cobertura local
            cover_bonus = sum(1.0 / (1.0 + cobertura_execucao[n]) for n in nova) - \
                          sum(1.0 / (1.0 + cobertura_execucao[n]) for n in tentativa)

            # aceita se melhora score, ou piora pouco e melhora cobertura
            if (score_novo + tol_score >= score_orig) or (score_novo >= score_orig - tol_score and cover_bonus > 0.5):
                tentativa = nova
                score_orig = score_novo
                trocas += 1

        return sorted(tentativa)

    def gerar_por_modelo(self) -> List[int]:
        if not hasattr(self, "modelo") or self.modelo is None:
            raise RuntimeError("Modelo LSTM indisponível.")
        ult = self.dados[[f'B{i}' for i in range(1,16)]].values[-10:]
        t = len(ult)
        X = np.zeros((1, 10, 25 + 5))
        for i in range(t):
            row = ult[-t + i]
            for num in row:
                X[0, i, num - 1] = 1
            for j in range(1, 6):
                col_name = f'repetidos_{j}'
                if col_name in self.dados.columns:
                    X[0, i, 25 + j - 1] = self.dados.iloc[-t + i][col_name]
        pred = self.modelo.predict(X, verbose=0)[0]
        return sorted([i + 1 for i in np.argsort(pred)[-15:]])

    def gerar_por_algoritmo_genetico(self) -> List[int]:
        if not hasattr(self, "_creator_classes_defined"):
            try:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                self._creator_classes_defined = True
            except Exception:
                pass

        toolbox = base.Toolbox()
        toolbox.register("attr_num", random.randint, 1, 25)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_num, n=15)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.avaliar_aposta_ga)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=1, up=25, indpb=0.15)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=200)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=40, verbose=False)
        melhor = tools.selBest(pop, k=1)[0]
        aposta_final = sorted(set(melhor))
        while len(aposta_final) < 15:
            candidatos = [n for n in range(1, 26) if n not in aposta_final]
            aposta_final.append(random.choice(candidatos))
            aposta_final = sorted(aposta_final)
        aposta_final = aposta_final[:15]
        return aposta_final

    def avaliar_aposta_ga(self, aposta: List[int]) -> Tuple[float]:
        """
        Score multi-critérios (versão consolidada):
        - Frequência (z-score)
        - Coocorrência leve
        - Clusters (bônus se 2–4 por cluster)
        - Balanceamento por faixas (via _score_balance)
        - Sanidade (pares, soma, runs)
        - Antiviés: penaliza {1,2,3} e excesso de baixos
        - Bônus de extremos e sequência inicial leve
        """
        aposta = sorted(set(int(n) for n in aposta if 1 <= int(n) <= 25))
        if len(aposta) != 15:
            return (0.0,)

        s_ap = set(aposta)

        # 1) Frequência z-score
        freq_arr = np.array([self.frequencias.get(i, 0) for i in range(1, 26)], dtype=float)
        mu = float(freq_arr.mean())
        sd = float(freq_arr.std()) if float(freq_arr.std()) > 1e-9 else 1.0
        zscores = (freq_arr - mu) / sd
        score_freq = float(sum(zscores[n - 1] for n in aposta))
        w_freq = 0.6

        # 2) Coocorrência leve
        score_cooc = 0.0
        for i in range(len(aposta)):
            for j in range(i + 1, len(aposta)):
                score_cooc += float(self.coocorrencias[aposta[i] - 1, aposta[j] - 1])
        w_cooc = 0.06

        # 3) Clusters
        score_cluster = 0.0
        for cluster_nums in self.clusters.values():
            inter = s_ap & set(cluster_nums)
            if 2 <= len(inter) <= 4:
                score_cluster += 1.0
        w_cluster = 2.0

        # 4) Balanceamento por faixas
        score_bal = self._score_balance(aposta)
        w_bal = 1.8

        # 5) Sanidade
        pares = sum(1 for n in aposta if n % 2 == 0)
        soma = sum(aposta)
        penalty = 0.0
        if not (5 <= pares <= 10):
            penalty += 8.0
        if not (160 <= soma <= 220):
            penalty += 8.0
        run_len = self._maior_sequencia_consecutivos(aposta)
        if run_len >= 4:
            penalty += (run_len - 3) * 4.0

        # 6) Antiviés
        if {1, 2, 3}.issubset(s_ap):
            penalty += 12.0
        low, mid, high = self._count_low_mid_high(aposta)
        if low >= 9:
            penalty += (low - 8) * 3.0
        if sum(1 for n in aposta if 1 <= n <= 5) >= 4:
            penalty += 4.0

        # 7) Bônus de extremos
        bonus_extremos = 1.0 if any(n <= 3 for n in aposta) and any(n >= 23 for n in aposta) else 0.0

        # 8) Sequência inicial leve
        seq_inicial = tuple(sorted(aposta)[:3])
        score_seq = float(self.sequencias_iniciais.get(seq_inicial, 0)) * 0.1

        # 9) Agregação
        score_total = (
            w_freq * score_freq +
            w_cooc * score_cooc +
            w_cluster * score_cluster +
            w_bal * score_bal +
            score_seq + bonus_extremos
            - penalty
        )
        return (float(score_total),)

    def gerar_aposta(self, n_apostas: int = 5) -> List[List[int]]:
        """Fallback clássico: GA + (opcional) modelo, com fechamento ao final."""
        apostas = []
        usa_modelo = hasattr(self, "modelo") and self.modelo is not None and len(self.dados) >= 10
        for _ in range(n_apostas):
            aposta_ga = self.gerar_por_algoritmo_genetico()
            if usa_modelo:
                try:
                    aposta_modelo = self.gerar_por_modelo()
                    aposta_final = self.combinar_apostas(aposta_modelo, aposta_ga)
                except Exception:
                    aposta_final = sorted(aposta_ga)
            else:
                aposta_final = sorted(aposta_ga)
            apostas.append(aposta_final)
        return self.aplicar_fechamento(apostas)

    # -------------------------
    # Anti-viés / diversidade forte
    # -------------------------
    def _forca_quebra_123(self, ap: List[int], rng: random.Random) -> List[int]:
        if self._tem_prefixo_123(ap):
            tira = rng.choice([1, 2, 3])
            resto = [x for x in ap if x != tira]
            fora = [n for n in range(1, 26) if n not in resto]
            entra = rng.choice(fora)
            resto.append(entra)
            ap = sorted(resto)
            ap = self._repara(ap, rng)
        return ap

    def _enforce_final_constraints(self, ap: List[int], rng: random.Random) -> List[int]:
        """
        Ajustes finais anti-viés:
        - limita 'low' (1–8) a no máximo 8
        - tenta cobrir extremos (>=1 em 1–3 e >=1 em 23–25)
        - mantém regras básicas (pares, soma, sequências) via _repara
        """
        ap = sorted(ap)
        tries = 0
        while tries < 20:
            low, mid, high = self._count_low_mid_high(ap)
            has_low_ext = any(n <= 3 for n in ap)
            has_high_ext = any(n >= 23 for n in ap)

            ok_low = (low <= 8)
            ok_ext = has_low_ext and has_high_ext

            if ok_low and ok_ext and self._valida_regras_basicas(ap):
                break

            # 1) reduzir low se necessário trocando por mid/high
            if low > 8:
                pool_out = [n for n in ap if 1 <= n <= 8]
                pool_in = [n for n in range(18, 26) if n not in ap] or [n for n in range(9, 18) if n not in ap]
                if pool_out and pool_in:
                    sai = rng.choice(pool_out)
                    entra = rng.choice(pool_in)
                    ap = sorted([x for x in ap if x != sai] + [entra])

            # 2) garantir extremo baixo (1–3)
            if not has_low_ext:
                cand_in = [n for n in range(1, 4) if n not in ap]
                if cand_in:
                    # retira algo fora do extremo baixo
                    pool_out = [n for n in ap if n > 3]
                    if pool_out:
                        sai = rng.choice(pool_out)
                        entra = rng.choice(cand_in)
                        ap = sorted([x for x in ap if x != sai] + [entra])

            # 3) garantir extremo alto (23–25)
            if not has_high_ext:
                cand_in = [n for n in range(23, 26) if n not in ap]
                if cand_in:
                    pool_out = [n for n in ap if n < 23]
                    if pool_out:
                        sai = rng.choice(pool_out)
                        entra = rng.choice(cand_in)
                        ap = sorted([x for x in ap if x != sai] + [entra])

            # 4) repara para respeitar pares/soma/seq
            ap = self._repara(ap, rng)
            tries += 1

        return sorted(ap)

    def _forca_diversidade_lote(
        self,
        lote: List[List[int]],
        min_diff: int,
        rng: random.Random,
        cobertura_execucao: Counter
    ) -> List[List[int]]:
        """
        Reforço final por diversidade e antiviés:
        - quebra {1,2,3}
        - aplica _enforce_final_constraints
        - impõe diversidade mínima vs. já escolhidas
        - evita duplicatas globais
        """
        final: List[List[int]] = []
        vistos: Set[Tuple[int, ...]] = set()

        for ap in lote:
            ap = self._forca_quebra_123(ap, rng)
            ap = self._enforce_final_constraints(ap, rng)

            # diversidade mínima
            tent = 0
            while final and (min(15 - len(set(ap) & set(e)) for e in final) < min_diff) and tent < 25:
                ap = self._mutacao_suave(ap, rng, cobertura_execucao, max_trocas=3, tol_score=0.5, p_aplicar=0.8)
                ap = self._repara(ap, rng)
                ap = self._enforce_final_constraints(ap, rng)
                tent += 1

            # unicidade global
            tent = 0
            key = tuple(sorted(ap))
            while key in vistos and tent < 20:
                ap = self._mutacao_suave(ap, rng, cobertura_execucao, max_trocas=3, tol_score=0.5, p_aplicar=0.8)
                ap = self._repara(ap, rng)
                ap = self._enforce_final_constraints(ap, rng)
                key = tuple(sorted(ap))
                tent += 1

            final.append(sorted(ap))
            vistos.add(key)

        return final

    def gerar_aposta_precisa(self, n_apostas: int = 5, seed: Optional[int] = None) -> List[List[int]]:
        """Versão sem viés: candidatos múltiplos + avaliação + pós-processamento forte."""
        if self.dados is None or len(self.dados) == 0:
            raise RuntimeError("Dados indisponíveis para geração precisa.")

        df = self.dados
        if 'numero' in df.columns:
            df = df.sort_values('numero').reset_index(drop=True)
        elif 'data' in df.columns:
            df = df.sort_values('data').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        historico: List[List[int]] = [
            [int(row[f'B{i}']) for i in range(1, 16)]
            for _, row in df.iterrows()
        ]

        if seed is None:
            try:
                seed = int(df['numero'].max()) + 1
            except Exception:
                seed = len(df) + 1

        n_alvo = max(1, min(int(n_apostas), 10))
        rng_global = random.Random(seed)
        cobertura_execucao = Counter()
        vistos: Set[Tuple[int, ...]] = set()
        apostas_tmp: List[List[int]] = []

        MIN_DIFF = 9
        CAND_POR_POS = 10

        def _ganho_cobertura(ap: List[int]) -> float:
            return sum(1.0 / (1.0 + cobertura_execucao[n]) for n in ap)

        for i in range(n_alvo):
            cand_list: List[List[int]] = []

            # 1) Engine precisa (seeds variados)
            for t in range(4):
                try:
                    geradas = gerar_apostas_precisas(
                        historico, quantidade=1, seed=seed + i*1543 + t*97, cfg=self.cfg_precisa
                    )
                    if geradas:
                        cand_list.append(sorted(set(map(int, geradas[0])))[:15])
                except Exception:
                    pass

            # 2) GA (várias execuções)
            for _ in range(3):
                cand_list.append(self.gerar_por_algoritmo_genetico())

            # 3) Modelo (se houver)
            if hasattr(self, "modelo") and self.modelo is not None and len(self.dados) >= 10:
                try:
                    cand_list.append(self.gerar_por_modelo())
                except Exception:
                    pass

            # 4) Mutações (exploratório — aqui pode ser mais agressivo)
            base_for_mut = cand_list[:]
            for ap in base_for_mut:
                cand_list.append(self._mutacao_suave(ap, rng_global, cobertura_execucao,
                                                 max_trocas=2, tol_score=3.0, p_aplicar=1.0))

            # 5) Reparo + filtros + pré-diversidade local
            candidatos_validos: List[List[int]] = []
            seen_local: Set[Tuple[int, ...]] = set()
            for ap in cand_list:
                ap = self._repara(ap, rng_global)
                if not self._valida_regras_basicas(ap):
                    continue
                ap = self._forca_quebra_123(ap, rng_global)
                if candidatos_validos and min(15 - len(set(ap) & set(e)) for e in candidatos_validos) < 4:
                    continue
                key = tuple(sorted(ap))
                if key in seen_local:
                    continue
                seen_local.add(key)
                candidatos_validos.append(ap)
                if len(candidatos_validos) >= CAND_POR_POS:
                    break

            # fallback aleatório reparado
            while len(candidatos_validos) < max(3, CAND_POR_POS // 2):
                ap = [rng_global.randrange(1, 26) for _ in range(15)]
                ap = self._repara(ap, rng_global)
                ap = self._forca_quebra_123(ap, rng_global)
                key = tuple(sorted(ap))
                if key not in seen_local and self._valida_regras_basicas(ap):
                    seen_local.add(key)
                    candidatos_validos.append(ap)

            # 6) score leve + diversidade contra lote temporário
            def _score_total(ap: List[int]) -> float:
                fit = float(self.avaliar_aposta_ga(ap)[0])  # já inclui balanceamento por faixas
                dist = 0.0 if not apostas_tmp else min(15 - len(set(ap) & set(e)) for e in apostas_tmp)
                cover = _ganho_cobertura(ap)
                pen = 0.0
                low, _, _ = self._count_low_mid_high(ap)
                if low > 8:
                    pen += (low - 8) * 2.5
                if {1, 2, 3}.issubset(set(ap)):
                    pen += 6.0
                return (fit * 1.0) + (dist * 2.5) + (cover * 0.9) - pen

            escolhido = max(candidatos_validos, key=_score_total)

            # evita duplicata global (uma única estratégia padronizada)
            tries = 0
            key = tuple(sorted(escolhido))
            while key in vistos and tries < 12:
                escolhido = self._mutacao_suave(escolhido, rng_global, cobertura_execucao)
                escolhido = self._repara(escolhido, rng_global)
                escolhido = self._forca_quebra_123(escolhido, rng_global)
                key = tuple(sorted(escolhido))
                tries += 1

            vistos.add(key)
            apostas_tmp.append(sorted(escolhido))
            cobertura_execucao.update(escolhido)

        # 7) Cinturão final
        apostas_final = self._forca_diversidade_lote(apostas_tmp, MIN_DIFF, rng_global, cobertura_execucao)
        self.ultima_geracao_precisa = [sorted(ap) for ap in apostas_final]
        return self.ultima_geracao_precisa

    # -------------------------
    # Checks de saúde
    # -------------------------
    def _precheck_precisa(self) -> None:
        if self.dados is None or len(self.dados) < 30:
            raise RuntimeError("Histórico insuficiente para geração precisa (mínimo 30 concursos).")
        for col in [f'B{i}' for i in range(1,16)]:
            if col not in self.dados.columns:
                raise RuntimeError(f"Coluna obrigatória ausente no histórico: {col}")

    def _teste_engine_precisa_startup(self) -> bool:
        self._precheck_precisa()
        _ = self.gerar_aposta_precisa(n_apostas=1, seed=None)
        return True

    # -------------------------
    # Outras utilidades
    # -------------------------
    def gerar_aposta_precisa_com_retry(self, n_apostas: int, seed: Optional[int] = None, retries: int = 2) -> List[List[int]]:
        last_exc: Optional[Exception] = None
        self._precheck_precisa()
        for tent in range(retries + 1):
            try:
                resultado = self.gerar_aposta_precisa(n_apostas=n_apostas, seed=seed)
                self.precise_fail_count = 0
                self.precise_enabled = True
                self.precise_last_error = None
                return resultado
            except Exception as e:
                last_exc = e
                self.precise_fail_count += 1
                self.precise_last_error = str(e)
                try:
                    import time as _t
                    _t.sleep(0.2 * (tent + 1))
                except Exception:
                    pass

        self.precise_enabled = False
        if self.precise_fail_count >= 3:
            try:
                for _admin in ADMIN_USER_IDS:
                    self._notificar_admin_falha_precisa(_admin)
            except Exception:
                pass
        raise last_exc or RuntimeError("Falha desconhecida no engine precisa.")

    def _notificar_admin_falha_precisa(self, admin_id: int) -> None:
        try:
            logger.warning(
                f"[ADMIN ALERT] Falhas seguidas no engine precisa: {self.precise_fail_count} | "
                f"Último erro: {self.precise_last_error}"
            )
        except Exception:
            pass

    def combinar_apostas(self, aposta1: List[int], aposta2: List[int]) -> List[int]:
        """Combina duas apostas de forma inteligente."""
        comuns = set(aposta1) & set(aposta2)
        diferentes = list((set(aposta1) | set(aposta2)) - comuns)
        random.shuffle(diferentes)
        nova_aposta = list(comuns) + diferentes[:15 - len(comuns)]
        return sorted(nova_aposta)

    def aplicar_fechamento(self, apostas: List[List[int]]) -> List[List[int]]:
        """
        Fecha cobertura global (1..25) sem viés:
        - adiciona números faltantes trocando de apostas com maior redundância
        - nunca deixa 'low' (1–8) estourar (>8)
        - preserva extremos (>=1 em 1–3 e >=1 em 23–25) dentro de cada aposta
        - mantém regras básicas (pares, soma, sequências)
        """
        if not apostas:
            return apostas

        todos_numeros = set(range(1, 26))
        cobertura = Counter()
        for ap in apostas:
            cobertura.update(ap)

        faltantes = [n for n in todos_numeros if cobertura[n] == 0]

        # determinismo leve pra facilitar debug
        rng = random.Random(sum(sum(ap) for ap in apostas) + 349)

        def _mantem_extremos(nums: List[int]) -> bool:
            return any(n <= 3 for n in nums) and any(n >= 23 for n in nums)

        def _score_aposta_para_troca(ap: List[int]) -> float:
            return sum(self.frequencias.get(n, 0) for n in ap)

        apostas_idx_ordenadas = sorted(range(len(apostas)), key=lambda i: _score_aposta_para_troca(apostas[i]))

        # Só entra no loop se de fato houver faltantes
        if faltantes:
            for num_in in faltantes:
                trocou = False

                for idx in apostas_idx_ordenadas:
                    ap = apostas[idx]
                    cand_out = [n for n in ap if cobertura[n] > 1 and n != num_in]
                    if not cand_out:
                        continue

                    cand_out.sort(key=lambda n: (self.frequencias.get(n, 0), cobertura[n]), reverse=True)

                    for sai in cand_out:
                        if num_in in ap:
                            break

                        tentativa = sorted([x for x in ap if x != sai] + [num_in])

                        if not _mantem_extremos(tentativa):
                            continue

                        low, _, _ = self._count_low_mid_high(tentativa)
                        if low > 8:
                            continue

                        tentativa = self._repara(tentativa, rng)
                        tentativa = self._enforce_final_constraints(tentativa, rng)
                        if not self._valida_regras_basicas(tentativa):
                            continue

                        cobertura[sai] -= 1
                        cobertura[num_in] += 1
                        apostas[idx] = sorted(tentativa)
                        trocou = True
                        break

                    if trocou:
                        break

                # fallback permissivo (mantendo sanidade)
                if not trocou:
                    for idx in apostas_idx_ordenadas:
                        ap = apostas[idx]
                        cand_out = [n for n in ap if n != num_in]
                        rng.shuffle(cand_out)

                        for sai in cand_out:
                            tentativa = sorted([x for x in ap if x != sai] + [num_in])
                            tentativa = self._repara(tentativa, rng)
                            tentativa = self._enforce_final_constraints(tentativa, rng)
                            if not self._valida_regras_basicas(tentativa):
                                continue

                            cobertura[sai] -= 1
                            cobertura[num_in] += 1
                            apostas[idx] = sorted(tentativa)
                            trocou = True
                            break

                        if trocou:
                            break

        # reforço anti-viés para TODAS as apostas, mesmo sem faltantes
        rng_post = random.Random(sum(sum(ap) for ap in apostas) + 199)
        apostas = [self._enforce_final_constraints(ap[:], rng_post) for ap in apostas]
        return apostas

    def verificar_clusters(self, aposta: List[int]) -> Dict[int, int]:
        """Retorna distribuição da aposta pelos clusters."""
        dist = {}
        for cluster, nums in self.clusters.items():
            dist[cluster] = len(set(aposta) & set(nums))
        return dist

    def gerar_grafico_frequencia(self) -> BytesIO:
        """Gera gráfico de frequência dos números."""
        plt.figure(figsize=(10, 5))
        nums, freqs = zip(*sorted(self.frequencias.items()))
        plt.bar(nums, freqs)
        plt.title('Frequência de Números na Lotofácil')
        plt.xlabel('Número')
        plt.ylabel('Frequência')
        plt.xticks(range(1, 26))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def gerar_aposta_tendencia(self, hot_numbers: int = 8, cold_numbers: int = 7) -> List[int]:
        """Gera aposta baseada em números quentes e frios."""
        # Números quentes (mais sorteados nos últimos 10 concursos)
        last_10 = self.dados.tail(10)
        hot_nums = Counter()
        for _, row in last_10.iterrows():
            hot_nums.update(row[[f'B{i}' for i in range(1,16)]].values)
        hot_pool = [num for num, _ in hot_nums.most_common(hot_numbers)]

        # Números frios (não sorteados há mais de 15 concursos)
        all_drawn = set()
        for _, row in self.dados.tail(15).iterrows():
            all_drawn.update(row[[f'B{i}' for i in range(1,16)]].values)
        cold_pool = [n for n in range(1, 26) if n not in all_drawn][:cold_numbers]

        # Preenche o restante aleatoriamente
        remaining = 15 - len(hot_pool) - len(cold_pool)
        middle_pool = [n for n in range(1, 26) if n not in hot_pool and n not in cold_pool]
        selected = hot_pool + cold_pool + random.sample(middle_pool, max(0, remaining))
        return sorted(selected)


# =========================
# Bloco 4 — Telegram (handlers)
# =========================

# Instância principal do bot
bot = BotLotofacil()

def apenas_admin(func):
    def wrapper(update, context):
        try:
            if not bot.security.is_admin(update.effective_user.id):
                update.message.reply_text("❌ Comando restrito ao administrador.")
                return
            return func(update, context)
        except Exception as e:
            logger.error(f"Erro inesperado em comando admin: {str(e)}")
            update.message.reply_text("❌ Erro interno. Tente novamente mais tarde.")
    return wrapper

def somente_autorizado(func):
    def wrapper(update, context):
        try:
            if not bot.security.is_authorized(update.effective_user.id):
                update.message.reply_text("❌ Você não tem permissão para usar este bot.")
                return
            return func(update, context)
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")
            update.message.reply_text("❌ Ocorreu um erro inesperado. Tente novamente.")
    return wrapper

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(AVISO_LEGAL, parse_mode='HTML')
    update.message.reply_text(
        "🎰 Bot da Lotofácil IA 🎰\n\n"
        "Comandos disponíveis:\n"
        "/meuid - Mostra seu ID do Telegram\n"
        "/aposta - Gera apostas inteligentes\n"
        "/tendencia - Gera aposta baseada em tendências\n"
        "/analise - Mostra análise estatística\n"
        "/status - Mostra status do sistema\n\n"
        "Para solicitar acesso, use o comando /meuid e envie o ID ao administrador do bot.",
        parse_mode='HTML'
    )

@somente_autorizado
def comando_aposta(update: Update, context: CallbackContext) -> None:
    """Versão robusta do comando /aposta com tratamento completo de erros."""
    if not rate_limit(update, "aposta"):
        return

    success_flag = False
    progress_msg, progress_job = None, None

    try:
        chat_id = update.effective_chat.id

        if bot.dados is None or len(bot.dados) == 0:
            safe_send_message(
                context,
                chat_id,
                "❌ Dados indisponíveis. Use /atualizar ou aguarde atualização dos dados."
            )
            return

        # Inicia barra de progresso
        progress_msg, progress_job = _start_progress(context, chat_id)
        context.bot_data['timeout'] = 30

        # n de apostas
        n_apostas = int(context.args[0]) if context.args and context.args[0].isdigit() else 5
        n_apostas = max(1, min(n_apostas, 10))

        # Geração com retry + fallback
        try:
            apostas = bot.gerar_aposta_precisa_com_retry(
                n_apostas=n_apostas,
                seed=None,
                retries=2
            )
            engine_label = "precisa"
            logger.info(f"Geração precisa concluída para {n_apostas} apostas")
        except Exception as e_precisa:
            logger.warning(f"Falha no gerador preciso: {str(e_precisa)}")
            apostas = bot.gerar_aposta(n_apostas)
            engine_label = "clássico"
            logger.info(f"Usando fallback clássico para {n_apostas} apostas")

        # Mensagem de resultado
        mensagem = f"🎲 Apostas recomendadas — engine: {engine_label} 🎲\n\n"
        for i, aposta in enumerate(apostas, 1):
            if not isinstance(aposta, list) or len(aposta) != 15:
                raise ValueError("Formato de aposta inválido")
            pares = sum(1 for n in aposta if n % 2 == 0)
            soma = sum(aposta)
            mensagem += (
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in sorted(aposta))}\n"
                f"Pares: {pares} | Ímpares: {15-pares} | Soma: {soma}\n\n"
            )

        safe_send_message(context, chat_id, mensagem)
        logger.info(f"Apostas geradas: {n_apostas}")
        success_flag = True

    except Exception as e:
        logger.error(f"Erro no comando_aposta: {str(e)}", exc_info=True)
        try:
            safe_send_message(
                context,
                update.effective_chat.id,
                "⚠️ Falha durante a geração. Por favor, tente novamente mais tarde."
            )
        except Exception:
            pass
    finally:
        # Fecha barra de progresso de forma segura
        if progress_job:
            try:
                progress_job.schedule_removal()
            except Exception:
                pass
        if progress_msg:
            final_text = "✅ <b>Geração concluída.</b>" if success_flag else "❌ <b>Falha na geração.</b>"
            _stop_progress(progress_job, progress_msg, final_text)
            
@somente_autorizado
def comando_tendencia(update: Update, context: CallbackContext) -> None:
    if not rate_limit(update, "tendencia"):
        return
    if bot.dados is None or len(bot.dados) == 0:
        update.message.reply_text("❌ Dados indisponíveis. Use /atualizar ou aguarde atualização dos dados.")
        logger.error("Dados indisponíveis ao tentar gerar aposta de tendência.")
        return
    try:
        aposta = bot.gerar_aposta_tendencia()
        pares = sum(1 for n in aposta if n % 2 == 0)
        soma = sum(aposta)
        mensagem = (
            "📈 <b>Aposta Baseada em Tendências</b>\n\n"
            f"<b>Números:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
            f"Pares: {pares} | Ímpares: {15-pares} | Soma: {soma}\n\n"
            "<i>Estratégia: Combina números quentes (últimos sorteios) e frios (ausentes)</i>"
        )
        update.message.reply_text(mensagem, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Erro ao gerar aposta de tendência: {str(e)}")
        update.message.reply_text("❌ Ocorreu um erro ao gerar a aposta. Tente novamente.")

@somente_autorizado
def comando_analise(update: Update, context: CallbackContext) -> None:
    if not rate_limit(update, "analise"):
        return
    if bot.dados is None or len(bot.dados) == 0:
        update.message.reply_text("❌ Dados indisponíveis. Use /atualizar ou aguarde atualização dos dados.")
        logger.error("Dados indisponíveis ao tentar gerar análise estatística.")
        return
    try:
        grafico = bot.gerar_grafico_frequencia()
        update.message.reply_photo(photo=InputFile(grafico), caption='Frequência dos números na Lotofácil')

        freq_completa = [(n, bot.frequencias.get(n, 0)) for n in range(1, 26)]
        freq_ordenada = sorted(freq_completa, key=lambda x: (x[1], x[0]))
        menos_frequentes = [str(n) for n, _ in freq_ordenada[:5]]
        mais_frequentes = [str(n) for n, _ in sorted(freq_completa, key=lambda x: (-x[1], x[0]))[:5]]

        mensagem = (
            "<b>📊 Estatísticas Avançadas</b>\n\n"
            f"<b>Números mais frequentes:</b> {', '.join(mais_frequentes)}\n"
            f"<b>Números menos frequentes:</b> {', '.join(menos_frequentes)}\n\n"
            "<b>Clusters identificados:</b>\n"
        )
        for cluster, nums in bot.clusters.items():
            mensagem += f"Cluster {cluster}: {', '.join(str(n) for n in sorted(nums))}\n"
        update.message.reply_text(mensagem, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Erro ao gerar análise: {str(e)}")
        update.message.reply_text("❌ Ocorreu um erro ao gerar a análise. Tente novamente.")

@apenas_admin
def comando_atualizar(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("🔄 Atualizando dados... Isso pode demorar alguns minutos.")
    try:
        bot.dados = bot.carregar_dados(atualizar=True)
        if bot.dados is None or len(bot.dados) == 0:
            update.message.reply_text("❌ Falha ao atualizar dados. Nenhum dado foi carregado.")
            logger.error("Falha ao atualizar dados: Nenhum dado foi carregado.")
            return

        bot.analisar_dados()
        bot.modelo = bot.construir_modelo()

        try:
            bot._teste_engine_precisa_startup()
            bot.precise_enabled = True
            bot.precise_fail_count = 0
            bot.precise_last_error = None
            update.message.reply_text("✅ Dados e modelo atualizados com sucesso! Engine precisa revalidado e ATIVO.")
        except Exception as e:
            bot.precise_enabled = False
            bot.precise_last_error = str(e)
            update.message.reply_text(
                "✅ Dados e modelo atualizados.\n"
                f"⚠️ Engine precisa desativado após reteste: {e}"
            )
            logger.warning(f"Engine precisa desativado após atualizar: {e}")

    except Exception as e:
        logger.error(f"Erro ao atualizar dados: {str(e)}")
        update.message.reply_text("❌ Falha ao atualizar dados. Verifique os logs.")

@somente_autorizado
def comando_status(update: Update, context: CallbackContext) -> None:
    if not rate_limit(update, "status"):
        return
    if bot.dados is None or len(bot.dados) == 0:
        update.message.reply_text("❌ Dados indisponíveis. Use /atualizar ou aguarde atualização dos dados.")
        logger.error("Dados indisponíveis ao tentar verificar status do sistema.")
        return
    try:
        ultimo = bot.dados.loc[bot.dados['data'].idxmax()]
        precise_state = "✅ Ativo" if getattr(bot, "precise_enabled", False) else "❌ Desativado"
        precise_fail = getattr(bot, "precise_fail_count", 0)
        precise_err  = getattr(bot, "precise_last_error", None)

        status = (
            "<b>📊 Status do Sistema</b>\n\n"
            f"<b>Concursos carregados:</b> {len(bot.dados)}\n"
            f"<b>Último concurso:</b> {ultimo['numero']} ({ultimo['data'].strftime('%d/%m/%Y')})\n"
            f"<b>Modelo IA:</b> {'Carregado' if hasattr(bot, 'modelo') and bot.modelo is not None else 'Não treinado'}\n"
            f"<b>Engine precisa:</b> {precise_state} | Falhas recentes: {precise_fail}\n"
        )
        if precise_err:
            status += f"<b>Último erro:</b> {precise_err[:200]}{'...' if len(precise_err) > 200 else ''}\n"

        status += (
            f"\n<b>Números mais quentes:</b> {', '.join(str(n) for n, _ in bot.frequencias.most_common(3))}\n"
            f"<b>Números mais frios:</b> {', '.join(str(n) for n, _ in bot.frequencias.most_common()[-3:])}"
        )

        update.message.reply_text(status, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Erro ao verificar status: {str(e)}")
        update.message.reply_text("❌ Ocorreu um erro ao verificar o status. Tente novamente.")

@apenas_admin
def comando_inserir(update, context):
    if not rate_limit(update, "inserir"):
        return
    try:
        if not context.args or len(context.args) != 16:
            update.message.reply_text(
                "❌ Uso correto: /inserir YYYY-MM-DD D1 D2 ... D15\n"
                "Exemplo: /inserir 2025-08-08 01 03 05 07 09 10 12 14 17 18 19 20 22 23 25"
            )
            return
        data = context.args[0]
        dezenas = context.args[1:]
        try:
            dezenas_int = [int(d) for d in dezenas]
        except Exception:
            update.message.reply_text("❌ Todas as dezenas devem ser números inteiros entre 1 e 25.")
            return
        if len(dezenas_int) != 15 or any(not 1 <= d <= 25 for d in dezenas_int):
            update.message.reply_text("❌ Dados inválidos. Verifique os números (apenas 15 dezenas, de 1 a 25).")
            return
        try:
            pd.to_datetime(data, format="%Y-%m-%d")
        except Exception:
            update.message.reply_text("❌ Data inválida. Utilize o formato YYYY-MM-DD.")
            return

        # valida duplicidade de data
        try:
            data_dt = pd.to_datetime(data, format="%Y-%m-%d", errors="raise")
        except Exception:
            update.message.reply_text("❌ Data inválida ao normalizar. Utilize o formato YYYY-MM-DD.")
            return

        arq = 'lotofacil_historico.csv'
        if not os.path.exists(arq):
            update.message.reply_text("❌ Arquivo lotofacil_historico.csv não encontrado no servidor.")
            return
        df = pd.read_csv(arq)

        # checa data já existente
        try:
            if 'data' in df.columns and not df.empty:
                df_datas = pd.to_datetime(df['data'], format="%Y-%m-%d", errors="coerce")
                if df_datas.isna().any():
                    alt = pd.to_datetime(df['data'], dayfirst=True, errors="coerce")
                    df_datas = df_datas.fillna(alt)
                if (df_datas.dt.date == data_dt.date()).any():
                    update.message.reply_text(
                        "⚠️ Já existe um concurso com esta data no histórico.\n"
                        "Para evitar duplicidade, a inserção foi cancelada."
                    )
                    return
        except Exception:
            pass

        # bloqueia inserção se combinação de 15 dezenas já existe
        try:
            if not df.empty:
                alvo = frozenset(int(x) for x in dezenas_int)
                cols_b = [f'B{i}' for i in range(1, 16) if f'B{i}' in df.columns]
                if len(cols_b) == 15:
                    repetida = False
                    for _, row in df[cols_b].iterrows():
                        try:
                            s = frozenset(int(row[f'B{i}']) for i in range(1, 16))
                            if s == alvo:
                                repetida = True
                                break
                        except Exception:
                            continue
                    if repetida:
                        update.message.reply_text(
                            "⚠️ Já existe um concurso com exatamente as mesmas 15 dezenas.\n"
                            "A inserção foi cancelada."
                        )
                        return
        except Exception:
            pass

        # próximo número seguro
        if 'numero' in df.columns and not df['numero'].dropna().empty:
            try:
                max_num = pd.to_numeric(df['numero'], errors='coerce').max()
                proximo_numero = int(max_num) + 1 if pd.notna(max_num) else len(df) + 1
            except Exception:
                proximo_numero = len(df) + 1
        else:
            proximo_numero = len(df) + 1

        nova_linha = {'numero': proximo_numero, 'data': data}
        for i, dez in enumerate(dezenas_int, 1):
            nova_linha[f'B{i}'] = dez
        for i in range(1, 6):
            nova_linha[f'repetidos_{i}'] = 0

        df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)

        # recalcula campos de repetidos
        for rep in range(1, 6):
            repetidos = []
            for idx, row in df.iterrows():
                if idx < rep:
                    repetidos.append(0)
                else:
                    nums_atual = set([row.get(f'B{i}', 0) for i in range(1, 16)])
                    nums_anterior = set([df.iloc[idx - rep].get(f'B{i}', 0) for i in range(1, 16)])
                    repetidos.append(len(nums_atual & nums_anterior))
            df[f'repetidos_{rep}'] = repetidos

        df = df.sort_values('numero').reset_index(drop=True)
        backup_csv()
        df.to_csv(arq, index=False, encoding='utf-8')

        update.message.reply_text(
            f"✅ Resultado inserido com sucesso!\nConcurso: {proximo_numero}\nData: {data}\n"
            f"Dezenas: {' '.join(str(d).zfill(2) for d in dezenas_int)}"
        )
    except Exception as e:
        logger.error(f"Erro ao inserir resultado: {str(e)}")
        update.message.reply_text("❌ Falha ao inserir o resultado. Tente novamente.")

def comando_meuid(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    update.message.reply_text(
        f"Seu ID do Telegram é: <b>{user_id}</b>\n\n"
        "Seu ID será utilizado apenas para controle de acesso ao bot. Nenhuma outra informação pessoal é salva ou compartilhada.\n"
        "Envie este número para o administrador do bot para solicitar acesso.",
        parse_mode='HTML'
    )

@apenas_admin
def comando_autorizar(update: Update, context: CallbackContext) -> None:
    try:
        if not context.args or not context.args[0].isdigit():
            update.message.reply_text("❌ Uso correto: /autorizar <ID_do_usuário>")
            return
        user_id = int(context.args[0])
        whitelist_path = "whitelist.txt"
        bot.security.load_whitelist(whitelist_path)
        if user_id in bot.security.whitelist or user_id in bot.security.admins:
            update.message.reply_text(f"✅ O ID {user_id} já está autorizado.")
            return
        with open(whitelist_path, "a") as f:
            f.write(f"{user_id}\n")
        bot.security.whitelist.add(user_id)
        update.message.reply_text(f"✅ Usuário {user_id} autorizado com sucesso.")
        # envia o manual automaticamente
        context.bot.send_message(chat_id=user_id, text=MANUAL_USUARIO, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Erro ao autorizar usuário: {str(e)}")
        update.message.reply_text("❌ Erro ao autorizar usuário.")

@apenas_admin
def comando_remover(update: Update, context: CallbackContext) -> None:
    try:
        if not context.args or not context.args[0].isdigit():
            update.message.reply_text("❌ Uso correto: /remover <ID_do_usuário>")
            return
        user_id = int(context.args[0])
        whitelist_path = "whitelist.txt"
        bot.security.load_whitelist(whitelist_path)
        if user_id not in bot.security.whitelist:
            update.message.reply_text(f"ℹ️ O ID {user_id} não está na whitelist.")
            return
        with open(whitelist_path, "r") as f:
            linhas = f.readlines()
        with open(whitelist_path, "w") as f:
            for linha in linhas:
                if linha.strip() != str(user_id):
                    f.write(linha)
        bot.security.whitelist.discard(user_id)
        update.message.reply_text(f"✅ Usuário {user_id} removido da whitelist.")
    except Exception as e:
        logger.error(f"Erro ao remover usuário: {str(e)}")
        update.message.reply_text("❌ Erro ao remover usuário.")
        
def error_handler(update: Update, context: CallbackContext) -> None:
    """Tratamento robusto de erros, incluindo timeouts."""
    error = context.error
    logger.error(f"Erro no bot: {str(error)}", exc_info=True)

    try:
        if isinstance(error, telegram.error.TimedOut):
            if update and update.message:
                safe_send_message(
                    context,
                    update.message.chat_id,
                    "⌛ O sistema está ocupado. Tentando novamente...",
                    timeout=20
                )

        elif isinstance(error, telegram.error.NetworkError):
            logger.error("Problema de conexão com a API do Telegram")
            if update and update.message:
                safe_send_message(
                    context,
                    update.message.chat_id,
                    "⚠️ Problema temporário de conexão. Por favor, tente novamente em alguns instantes."
                )

        else:
            if update and update.message:
                safe_send_message(
                    context,
                    update.message.chat_id,
                    "❌ Ocorreu um erro inesperado. Os administradores foram notificados."
                )

    except Exception as inner_error:
        logger.error(f"Erro no handler de erros: {str(inner_error)}", exc_info=True)
            
def safe_send_message(context: CallbackContext, chat_id: int, text: str, **kwargs) -> None:
    """Envio seguro de mensagens com tratamento de timeout e retry."""
    max_retries = 3
    base_timeout = 10
    parse_mode = kwargs.pop('parse_mode', 'HTML')

    for attempt in range(max_retries):
        try:
            context.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                timeout=base_timeout * (attempt + 1),
                **kwargs
            )
            return
        except telegram.error.TimedOut:
            if attempt == max_retries - 1:
                logger.error(f"Falha ao enviar mensagem após {max_retries} tentativas")
                raise
            continue
        except telegram.error.NetworkError as e:
            logger.warning(f"Problema de rede ao enviar mensagem (tentativa {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Erro inesperado ao enviar mensagem: {str(e)}")
            raise

def main() -> None:
    """Função principal do bot com monitoramento de recursos completo."""
    try:
        # Ativa timeout global (2 minutos para inicialização)
        signal.alarm(120)

        # 1. Inicialização do Updater
        updater = Updater(
            token=TOKEN,
            use_context=True,
            request_kwargs=REQUEST_KWARGS,
            workers=4
        )
        dp = updater.dispatcher

        # 2. Configuração do sistema de monitoramento
        jq = updater.job_queue
        if jq:
            # Agendamento principal do monitoramento (30 em 30 minutos)
            jq.run_repeating(
                callback=ResourceMonitor.log_resource_usage,
                interval=1800,
                first=10
            )
            
            # Teste rápido do sistema após 5 segundos
            jq.run_once(
                lambda ctx: logger.info("✅ Teste inicial do monitoramento concluído com sucesso"),
                when=5
            )
            logger.info("Monitoramento de recursos ativado (intervalo: 30 minutos)")
        else:
            logger.warning("JobQueue indisponível - monitoramento de recursos desativado")

        # 3. Verificação de saúde inicial do monitoramento
        try:
            stats = ResourceMonitor.get_system_stats()
            if not stats:
                logger.warning("Coleta inicial de recursos retornou vazia")
            else:
                logger.info(f"Teste de saúde OK - Memória: {stats['mem_percent']:.1f}%")
        except Exception as e:
            logger.error(f"Falha no teste inicial do monitoramento: {str(e)}")

        # 4. Registro dos handlers de comandos
        handlers = [
            CommandHandler("start", start),
            CommandHandler("meuid", comando_meuid),
            CommandHandler("aposta", comando_aposta),
            CommandHandler("tendencia", comando_tendencia),
            CommandHandler("analise", comando_analise),
            CommandHandler("atualizar", comando_atualizar),
            CommandHandler("inserir", comando_inserir),
            CommandHandler("status", comando_status),
            CommandHandler("autorizar", comando_autorizar),
            CommandHandler("remover", comando_remover)
        ]

        for handler in handlers:
            dp.add_handler(handler)

        # 5. Configuração do handler de erros
        dp.add_error_handler(error_handler)

        # 6. Inicialização do bot
        logger.info("Iniciando bot com verificações de recursos...")
        updater.start_polling(
            poll_interval=0.5,
            timeout=15,
            drop_pending_updates=True
        )

        logger.info(
            f"✅ Bot em operação normal\n"
            f"• Monitoramento: {'ATIVO' if jq else 'INATIVO'}\n"
            f"• Checks de memória: >{_MAX_MEMORY_ALERT}% (alerta)\n"
            f"• Workers: {updater.dispatcher.workers}\n"
            f"• Timeouts: {REQUEST_KWARGS['connect_timeout']}s connect, {REQUEST_KWARGS['read_timeout']}s read"
        )

        # Desativa timeout após inicialização bem-sucedida
        signal.alarm(0)
        updater.idle()

    except telegram.error.NetworkError as e:
        logger.critical(f"ERRO DE REDE: {str(e)}")
        sys.exit(1)
    except telegram.error.InvalidToken:
        logger.critical("TOKEN INVÁLIDO - Verifique a variável de ambiente TOKEN")
        sys.exit(1)
    except telegram.error.TelegramError as e:
        logger.critical(f"ERRO DO TELEGRAM: {str(e)}")
        sys.exit(1)
    except TimeoutError:
        logger.critical("Timeout na inicialização do bot (excedeu 2 minutos)")
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"ERRO INESPERADO: {str(e)}\n"
            f"Tipo: {type(e).__name__}\n"
            f"Traceback: {traceback.format_exc()}",
            exc_info=True
        )
        sys.exit(1)
    finally:
        logger.info("Encerrando processo do bot...")
        signal.alarm(0)  # Garante que o timeout seja cancelado
        if 'updater' in locals():  # Verifica se a variável existe
            try:
                updater.stop()
                updater.is_idle = False
            except Exception as e:   
                logger.error(f"Erro ao parar updater: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot encerrado pelo usuário")
        sys.exit(0)
    except SystemExit as e:
        logger.error(f"Bot encerrado com código {e.code}")
        raise








