# =========================
# Bloco 1 — IMPORTS & SETUP
# =========================

# Stdlib
import os
import shutil
import pickle
import random
from io import BytesIO
from datetime import datetime
from collections import Counter, defaultdict
import logging
import warnings
from typing import Optional, Dict, List, Tuple, Set

# Terceiros
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import tensorflow as tf

# Matplotlib: backend seguro para servidor/headless (defina ANTES do pyplot)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# (opcional) seaborn não é usado; mantenha descomentado só se for usar de fato
# import seaborn as sns

# Telegram
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext

# Núcleo preciso (score + GRASP + diversidade)
from apostas_engine import gerar_apostas as gerar_apostas_precisas
from apostas_engine import Config as ApostaConfig

# ==== Barra de carregamento para /aposta (JobQueue) ====
from time import time as _now

_PROGRESS_FRAMES = ["▁","▃","▄","▅","▆","▇","█","▇","▆","▅","▄","▃"]

def _progress_tick(context):
    data = context.job.context
    msg = data.get("msg")
    i = data.get("i", 0)
    t0 = data.get("t0", _now())
    frame = _PROGRESS_FRAMES[i % len(_PROGRESS_FRAMES)]
    elapsed = int(_now() - t0)
    try:
        # Indeterminada: repetimos o mesmo char para parecer “barra”
        bar = frame * 20
        msg.edit_text(
            f"⏳ <b>Gerando apostas...</b>\n{bar}\n<i>{elapsed}s</i>",
            parse_mode='HTML'
        )
    except Exception:
        # ignora erros de edição (mensagem apagada, etc.)
        pass
    data["i"] = i + 1

def _start_progress(context, chat_id):
    """Cria a mensagem e agenda atualizações periódicas."""
    msg = context.bot.send_message(chat_id, "⏳ Iniciando geração...", parse_mode='HTML')
    job = context.job_queue.run_repeating(
        _progress_tick,
        interval=0.7,           # atualiza a cada ~700ms
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
    for _fn in ("lotofacil_historico.csv", "whitelist.txt", "modelo_lotofacil_avancado.h5"):
        src = os.path.join(os.getcwd(), _fn)
        dst = os.path.join(DATA_DIR, _fn)
        if (not os.path.exists(dst)) and os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.warning(f"Não foi possível copiar {src} -> {dst}: {e}")
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
    agora = _now()  # <- usa o alias único
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
                # cria arquivo vazio se não existir (idempotente)
                open(file, "a", encoding="utf-8").close()
        except Exception as e:
            logger.error(f"Erro ao carregar whitelist: {e}")

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.admins

    def is_authorized(self, user_id: int) -> bool:
        """Autorizado se estiver na whitelist ou for admin."""
        return user_id in self.whitelist or self.is_admin(user_id)

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
        self.modelo_path = 'modelo_lotofacil_avancado.h5'
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

        # Autoteste simples no startup: se falhar, mantém enabled=False e registra motivo
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

    def carregar_dados(self, atualizar: bool = False) -> Optional[pd.DataFrame]:
        """
        Carrega dados históricos apenas do arquivo local CSV, sempre recalculando os repetidos ao atualizar.
        Após o processamento, sobrescreve o arquivo CSV para garantir persistência das colunas.
        """
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
                # Salva o arquivo CSV atualizado com colunas de repetidos
                processed.to_csv('lotofacil_historico.csv', index=False)
            except Exception as e:
                logger.error(f"Falha ao salvar cache ou CSV: {str(e)}")

        return processed

    def preprocessar_dados(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Versão otimizada e robusta para CSV da Lotofácil (B1-B15 + repetidos_X)"""
        try:
            # Verificação robusta do formato esperado
            required_cols = ['data'] + [f'B{i}' for i in range(1,16)]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Colunas obrigatórias faltantes. Esperado: {required_cols}")
                return None

            # Conversão robusta do campo data (aceita YYYY-MM-DD e DD/MM/YYYY)
            # Primeiro tenta o padrão ISO, depois o padrão brasileiro
            try:
                df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d', errors='raise')
            except Exception:
                try:
                    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='raise')
                except Exception:
                    df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')

            # Remoção de linhas com datas não reconhecidas
            if df['data'].isnull().any():
                logger.warning(f"Linhas descartadas por data inválida: {df['data'].isnull().sum()}")
                df = df.dropna(subset=['data'])

            # Garante o campo 'numero'
            if 'numero' in df.columns:
                df['numero'] = df['numero'].astype(int)
            elif 'concurso' in df.columns:
                df['numero'] = df['concurso'].astype(int)
            else:
                df['numero'] = range(1, len(df)+1)

            # Conversão dos campos de dezenas para inteiro (garante que não haja valores string)
            for i in range(1, 16):
                df[f'B{i}'] = df[f'B{i}'].astype(int)

            # ⚠️ Ordena ANTES de calcular repetidos (garante temporalidade correta)
            if 'numero' in df.columns:
                df = df.sort_values('numero').reset_index(drop=True)
            else:
                df = df.sort_values('data').reset_index(drop=True)

            # Calcula corretamente os repetidos de 1 a 5 concursos anteriores
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

            # Seleciona apenas colunas relevantes
            cols_retorno = ['numero', 'data'] + [f'B{i}' for i in range(1,16)] + [f'repetidos_{j}' for j in range(1,6)]
            return df[cols_retorno]

        except Exception as e:
            logger.error(f"Falha crítica no pré-processamento: {str(e)}\nDados recebidos:\n{df.head()}")
            return None

    def analisar_dados(self) -> None:
        """Realiza análises estatísticas avançadas SEM cache de frequências"""
        # Calcula a frequência real dos números (B1 a B15), garantindo todos de 1 a 25
        contagem = Counter(self.dados.filter(like='B').values.flatten())
        self.frequencias = Counter({n: contagem.get(n, 0) for n in range(1, 26)})
        self.coocorrencias = self.calcular_coocorrencia()
        self.sequencias_iniciais = self.analisar_sequencias_iniciais()
        self.clusters = self.identificar_clusters()

    def calcular_coocorrencia(self) -> np.ndarray:
        """Calcula matriz de coocorrência com pesos temporais (peso maior para sorteios recentes)."""
        cooc = np.zeros((25, 25))
        N = len(self.dados)
        for i in range(1, N):
            nums_atual = set(self.dados.iloc[i][[f'B{j}' for j in range(1,16)]].values)
            nums_anterior = set(self.dados.iloc[i-1][[f'B{k}' for k in range(1,16)]].values)
            # distância até o final (mais recente => dist=1 => peso maior)
            dist = max(1, N - i)
            w = 1.0 / (dist ** 0.5)
            for num1 in nums_atual:
                for num2 in nums_anterior:
                    cooc[num1-1, num2-1] += w
        return cooc

    def analisar_sequencias_iniciais(self) -> Dict[Tuple[int, int, int], int]:
        """Analisa padrões nos primeiros números sorteados"""
        sequencias = defaultdict(int)
        for _, row in self.dados.iterrows():
            nums_ordenados = sorted(row[[f'B{i}' for i in range(1,16)]].values)
            chave = tuple(nums_ordenados[:3])  # Analisa os 3 primeiros números
            sequencias[chave] += 1
        return sequencias
    
    def identificar_clusters(self) -> Dict[int, List[int]]:
        """Identifica clusters dinâmicos com cache (sem warnings de feature names)."""
        cache_file = os.path.join(self.cache_dir, "clusters_cache.pkl")

        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache de clusters corrompido. Recriando... Erro: {str(e)}")

        # Treina com DataFrame e também PREDIZ com DataFrame (mesmas colunas) para evitar warnings
        dados_cluster = self.dados[[f'B{i}' for i in range(1, 16)]]
        kmeans = KMeans(n_clusters=4, random_state=42).fit(dados_cluster)

        clusters: Dict[int, List[int]] = {i: [] for i in range(4)}
        for num in range(1, 26):
            sample = pd.DataFrame([[num] * 15], columns=dados_cluster.columns)  # mantém nomes de features
            cluster = kmeans.predict(sample)[0]
            clusters[cluster].append(num)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(clusters, f)
        except Exception as e:
            logger.error(f"Falha ao salvar cache de clusters: {str(e)}")

        return clusters

    def construir_modelo(self) -> Optional[tf.keras.Model]:
        """Constroi modelo LSTM avançado com otimizações (validação temporal)."""
        if os.path.exists(self.modelo_path):
            try:
                return tf.keras.models.load_model(self.modelo_path)
            except Exception as e:
                logger.warning(f"Modelo corrompido. Recriando... Erro: {str(e)}")
                os.remove(self.modelo_path)
            
        X, y = self.preparar_dados_treinamento()

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(25, activation='sigmoid')
        ])
    
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
        early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.modelo_path, save_best_only=True)

        # ✅ Split temporal (80% treino, 20% validação), sem embaralhar
        n = len(X)
        if n < 10:
            # fallback de segurança (dados muito curtos)
            model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                callbacks=[early, checkpoint],
                shuffle=False,
                verbose=0
            )
        else:
            cut = int(n * 0.8)
            X_train, y_train = X[:cut], y[:cut]
            X_val,   y_val   = X[cut:], y[cut:]
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early, checkpoint],
                shuffle=False,
                verbose=0
            )
    
        return model

    def preparar_dados_treinamento(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para o modelo LSTM com janela temporal"""
        dados_numeros = self.dados[[f'B{i}' for i in range(1,16)]].values
        X, y = [], []
        
        janela = 10  # Analisa 10 concursos anteriores
        
        for i in range(janela, len(dados_numeros)):
            # Dados dos números
            seq_numeros = dados_numeros[i-janela:i]
            
            # Transforma em vetor binário (25 posições)
            seq_bin = np.zeros((janela, 25))
            for j in range(janela):
                for num in seq_numeros[j]:
                    seq_bin[j, num-1] = 1
            
            # Adiciona features adicionais se existirem
            features_extras = []
            for k in range(1,6):
                col_name = f'repetidos_{k}'
                if col_name in self.dados.columns:
                    features_extras.append(self.dados.iloc[i-1][col_name])
                else:
                    features_extras.append(0)
            
            features_extras = np.array(features_extras).reshape(1, -1)
            
            # Combina tudo
            X_seq = np.concatenate([seq_bin, np.tile(features_extras, (janela, 1))], axis=1)
            
            # Target (binário para os números sorteados)
            target = np.zeros(25)
            for num in dados_numeros[i]:
                target[num-1] = 1
                
            X.append(X_seq)
            y.append(target)
        
        return np.array(X), np.array(y)

    def _mutacao_suave(
        self,
        aposta: List[int],
        rng: random.Random,
        cobertura_execucao: Counter,
        max_trocas: int = 2,
        tol_score: float = 3.0,
        p_aplicar: float = 0.5,
    ) -> List[int]:
        """
        Faz 0–2 trocas leves para aumentar diversidade.
        Regras:
          - mantém 15 números únicos
          - mantém pares em [5,10] e soma em [160,220]
          - aceita se score não cair além de tol_score
        """
        if rng.random() > p_aplicar:
            return aposta[:]

        base = aposta[:]
        score_orig = self.avaliar_aposta_ga(base)[0]

        pressao_remover = {n: self.frequencias.get(n, 0) + cobertura_execucao[n] for n in base}
        candidatos_remover = sorted(base, key=lambda n: (-pressao_remover[n], n))

        fora = [n for n in range(1, 26) if n not in base]
        vantagem_incluir = {
            n: -self.frequencias.get(n, 0)
               + float(np.sum(self.coocorrencias[n-1, [x-1 for x in base]])) * 0.05
            for n in fora
        }
        candidatos_incluir = sorted(fora, key=lambda n: (vantagem_incluir[n], n), reverse=True)

        tentativa = base[:]
        trocas = 0
        idx_rem = 0
        idx_inc = 0

        while trocas < max_trocas and idx_rem < len(candidatos_remover) and idx_inc < len(candidatos_incluir):
            sai = candidatos_remover[idx_rem]; idx_rem += 1
            entra = candidatos_incluir[idx_inc]; idx_inc += 1

            if entra in tentativa:
                continue

            nova = [x for x in tentativa if x != sai] + [entra]
            nova.sort()

            pares = sum(1 for n in nova if n % 2 == 0)
            soma = sum(nova)
            if not (5 <= pares <= 10):
                continue
            if not (160 <= soma <= 220):
                continue

            score_novo = self.avaliar_aposta_ga(nova)[0]
            if score_novo + tol_score >= score_orig:
                tentativa = nova
                score_orig = score_novo
                trocas += 1

        return tentativa

    def _maior_sequencia_consecutivos(self, aposta: List[int]) -> int:
        """Retorna o tamanho da maior sequência de consecutivos na aposta."""
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
        """True se {1,2,3} estiver inteiro na aposta."""
        s = set(aposta)
        return {1,2,3}.issubset(s)

    def _diferenca_minima(self, ap: List[int], existentes: List[List[int]], min_diff: int = 4) -> bool:
        """Garante que a aposta difere de TODAS as já geradas por pelo menos min_diff dezenas."""
        s = set(ap)
        for e in existentes:
            comum = len(s & set(e))
            if 15 - comum < min_diff:
                return False
        return True

    def gerar_aposta(self, n_apostas: int = 5) -> List[List[int]]:
        apostas = []
        usa_modelo = hasattr(self, "modelo") and self.modelo is not None and len(self.dados) >= 10

        for _ in range(n_apostas):
            aposta_ga = self.gerar_por_algoritmo_genetico()
            if usa_modelo:
                try:
                    aposta_modelo = self.gerar_por_modelo()
                    aposta_final = self.combinar_apostas(aposta_modelo, aposta_ga)
                except Exception as e:
                    logger.warning(f"Falha em gerar_por_modelo: {e}. Usando só GA para esta aposta.")
                    usa_modelo = False
                    aposta_final = sorted(aposta_ga)
            else:
                aposta_final = sorted(aposta_ga)
            apostas.append(aposta_final)

        return self.aplicar_fechamento(apostas)

    def gerar_por_modelo(self) -> List[int]:
        if not hasattr(self, "modelo") or self.modelo is None:
            raise RuntimeError("Modelo LSTM indisponível.")
        ult = self.dados[[f'B{i}' for i in range(1,16)]].values[-10:]
        t = len(ult)  # pode ser < 10
        X = np.zeros((1, 10, 25 + 5))

        # Preenche só o que existe
        for i in range(t):
            row = ult[-t + i]  # i caminha do mais antigo ao mais recente dentro da janela disponível
            for num in row:
                X[0, i, num - 1] = 1
            for j in range(1, 6):
                col_name = f'repetidos_{j}'
                if col_name in self.dados.columns:
                    # usa a mesma linha temporal correspondente
                    X[0, i, 25 + j - 1] = self.dados.iloc[-t + i][col_name]

        pred = self.modelo.predict(X, verbose=0)[0]
        return sorted([i + 1 for i in np.argsort(pred)[-15:]])

    def gerar_por_algoritmo_genetico(self) -> List[int]:
        """Gera aposta usando algoritmo genético com restrições"""
        # Evita redefinir classes caso já existam
        if not hasattr(self, "_creator_classes_defined"):
            try:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
                self._creator_classes_defined = True
            except Exception:
                pass  # Classes já foram criadas em execuções anteriores

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
        # GARANTE 15 números únicos e ordenados entre 1 e 25
        aposta_final = sorted(set(melhor))
        while len(aposta_final) < 15:
            candidatos = [n for n in range(1, 26) if n not in aposta_final]
            aposta_final.append(random.choice(candidatos))
            aposta_final = sorted(aposta_final)
        # Caso tenha mais que 15 (por alguma operação), corta para 15
        aposta_final = aposta_final[:15]
        return aposta_final
    
    def avaliar_aposta_ga(self, aposta: List[int]) -> Tuple[float]:
        """Função de fitness para o algoritmo genético (penalidades aditivas + mitigação de viés)."""
        # Normaliza e valida a aposta
        aposta = list(set(aposta))
        if len(aposta) != 15:
            return (0.0,)

        score = 0.0

        # 1) Pontuação por frequência histórica
        score += sum(self.frequencias[n] for n in aposta)

        # 2) Pontuação por coocorrência (pares ordenados i != j)
        for i in aposta:
            for j in aposta:
                if i != j:
                    score += self.coocorrencias[i-1, j-1] * 0.1

        # 3) Bônus por clusters (2–4 números por cluster) — incentivo suave
        for cluster in self.clusters.values():
            intersect = set(aposta) & set(cluster)
            if 2 <= len(intersect) <= 4:
                score += 10.0

        # 4) Penalidades aditivas (faixas amplas para reduzir viés global)
        pares = sum(1 for n in aposta if n % 2 == 0)
        soma = sum(aposta)
        penalty = 0.0
        if not (5 <= pares <= 10):
            penalty += 10.0
        if not (160 <= soma <= 220):
            penalty += 10.0

        # 5) Reforço por sequência inicial comum (padrão histórico)
        seq_inicial = tuple(sorted(aposta)[:3])
        score += self.sequencias_iniciais.get(seq_inicial, 0) * 0.5

        # 6) Mitigação de viés estrutural (leve)
        if {1, 2, 3}.issubset(set(aposta)):
            score -= 6.0
        run_len = self._maior_sequencia_consecutivos(aposta)
        if run_len >= 4:
            score -= (run_len - 3) * 4.0

        # 7) Penalidade agregada no final
        score -= penalty
        return (score,)

    def gerar_aposta_precisa(self, n_apostas: int = 5, seed: Optional[int] = None) -> List[List[int]]:
        """
        Gera apostas usando o núcleo preciso (score + GRASP + diversidade) a partir de self.dados,
        garantindo diversidade (sem duplicatas) e variando a semente por aposta.
        """
        if self.dados is None or len(self.dados) == 0:
            raise RuntimeError("Dados indisponíveis para geração precisa.")

        # Histórico do mais antigo ao mais recente
        df = self.dados
        if 'numero' in df.columns:
            df = df.sort_values('numero').reset_index(drop=True)
        elif 'data' in df.columns:
            df = df.sort_values('data').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        historico: List[List[int]] = []
        for _, row in df.iterrows():
            historico.append([int(row[f'B{i}']) for i in range(1, 16)])

        # Semente base (reprodutível por concurso), mas variada por aposta
        if seed is None:
            try:
                seed = int(df['numero'].max()) + 1
            except Exception:
                seed = len(df) + 1

        n_alvo = max(1, min(int(n_apostas), 10))
        apostas_final: List[List[int]] = []
        vistos: Set[Tuple[int, ...]] = set()
        cobertura_execucao = Counter()  # cobertura dentro desta execução

        for i in range(n_alvo):
            rng = random.Random(seed + i*7919 if seed is not None else None)

            obtida: Optional[List[int]] = None
            for tentativa in range(8):
                seed_i = (seed or 0) + (i * 997) + (tentativa * 37)
                try:
                    geradas = gerar_apostas_precisas(
                        historico, quantidade=1, seed=seed_i, cfg=self.cfg_precisa
                    )
                except Exception:
                    continue

                if not geradas:
                    continue

                ap = sorted(map(int, geradas[0]))
                if tuple(ap) not in vistos:
                    obtida = ap
                    break

            if obtida is None:
                try:
                    obtida = sorted(self.gerar_por_algoritmo_genetico())
                except Exception:
                    obtida = apostas_final[0] if apostas_final else [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

            # >>> mitigação de viés + micro-variabilidade controlada <<<
            # 3.1) aplica mutação suave padrão
            obtida = self._mutacao_suave(
                aposta=obtida,
                rng=rng,
                cobertura_execucao=cobertura_execucao,
                max_trocas=2,
                tol_score=3.0,
                p_aplicar=0.5,
            )

            # 3.2) se detectar viés (prefixo 1-2-3, sequência longa) ou baixa diversidade no lote,
            #     força 1-2 mutações adicionais com p=1.0 e max_trocas maiores
            needs_break = self._tem_prefixo_123(obtida) or self._maior_sequencia_consecutivos(obtida) >= 4 \
                          or not self._diferenca_minima(obtida, apostas_final, min_diff=4)
            if needs_break:
                for _ in range(2):  # até 2 tentativas extra para quebrar padrão
                    obtida = self._mutacao_suave(
                        aposta=obtida,
                        rng=rng,
                        cobertura_execucao=cobertura_execucao,
                        max_trocas=3,
                        tol_score=4.0,
                        p_aplicar=1.0,  # força tentar
                    )
                    if self._maior_sequencia_consecutivos(obtida) < 4 and \
                       not self._tem_prefixo_123(obtida) and \
                       self._diferenca_minima(obtida, apostas_final, min_diff=4):
                        break
            # <<< fim mitigação >>>

            apostas_final.append(obtida)
            vistos.add(tuple(obtida))
            cobertura_execucao.update(obtida)
    
        self.ultima_geracao_precisa = apostas_final
        return self.ultima_geracao_precisa

    def _precheck_precisa(self) -> None:
        """Valida pré-condições para o engine preciso."""
        if self.dados is None or len(self.dados) < 30:
            raise RuntimeError("Histórico insuficiente para geração precisa (mínimo 30 concursos).")
        for col in [f'B{i}' for i in range(1,16)]:
            if col not in self.dados.columns:
                raise RuntimeError(f"Coluna obrigatória ausente no histórico: {col}")

    def _teste_engine_precisa_startup(self) -> bool:
        """Tenta gerar 1 aposta para verificar saúde no start."""
        self._precheck_precisa()
        _ = self.gerar_aposta_precisa(n_apostas=1, seed=None)
        return True

    def gerar_aposta_precisa_com_retry(self, n_apostas: int, seed: Optional[int] = None, retries: int = 2) -> List[List[int]]:
        """Wrapper resiliente com retries e contadores de falha."""
        last_exc: Optional[Exception] = None
        self._precheck_precisa()
        for tent in range(retries + 1):
            try:
                resultado = self.gerar_aposta_precisa(n_apostas=n_apostas, seed=seed)
                # sucesso: zera contador
                self.precise_fail_count = 0
                self.precise_enabled = True
                self.precise_last_error = None
                return resultado
            except Exception as e:
                last_exc = e
                self.precise_fail_count += 1
                self.precise_last_error = str(e)
                # pequeno backoff simples
                try:
                    import time as _t
                    _t.sleep(0.2 * (tent + 1))
                except Exception:
                    pass

        # se chegou aqui, esgotou retries: marca como degradado
        self.precise_enabled = False
        # opcional: alerta admin se houver muitas falhas seguidas
        if self.precise_fail_count >= 3:
            try:
                for _admin in ADMIN_USER_IDS:
                    self._notificar_admin_falha_precisa(_admin)
            except Exception:
                pass
        # levanta a última exceção para o caller aplicar fallback
        raise last_exc or RuntimeError("Falha desconhecida no engine precisa.")

    def _notificar_admin_falha_precisa(self, admin_id: int) -> None:
        """Mensagem simples de alerta ao admin (best-effort)."""
        try:
            logger.warning(f"[ADMIN ALERT] Falhas seguidas no engine precisa: {self.precise_fail_count} | Último erro: {self.precise_last_error}")
        except Exception:
            pass

    def combinar_apostas(self, aposta1: List[int], aposta2: List[int]) -> List[int]:
        """Combina duas apostas de forma inteligente"""
        comuns = set(aposta1) & set(aposta2)
        diferentes = list((set(aposta1) | set(aposta2)) - comuns)
        random.shuffle(diferentes)
        
        nova_aposta = list(comuns) + diferentes[:15 - len(comuns)]
        return sorted(nova_aposta)
    
    def aplicar_fechamento(self, apostas: List[List[int]]) -> List[List[int]]:
        todos_numeros = set(range(1, 26))
        cobertura = Counter()
        for ap in apostas:
            cobertura.update(ap)

        faltantes = [n for n in todos_numeros if cobertura[n] == 0]
        if not faltantes:
            return apostas

        for num in faltantes:
            aposta_menor_score = min(apostas, key=lambda a: sum(self.frequencias[n] for n in a))
            candidatos = [n for n in aposta_menor_score if cobertura[n] > 1]
            if not candidatos:
                # se não há redundância, troca qualquer um (para garantir cobertura global)
                candidatos = list(aposta_menor_score)

            substituto = random.choice(candidatos)
            aposta_menor_score.remove(substituto)
            aposta_menor_score.append(num)
            aposta_menor_score.sort()
            cobertura[substituto] -= 1
            cobertura[num] += 1

        return apostas
    
    def verificar_clusters(self, aposta: List[int]) -> Dict[int, int]:
        """Retorna distribuição da aposta pelos clusters"""
        dist = {}
        for cluster, nums in self.clusters.items():
            dist[cluster] = len(set(aposta) & set(nums))
        return dist
    
    def gerar_grafico_frequencia(self) -> BytesIO:
        """Gera gráfico de frequência dos números"""
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
        """Gera aposta baseada em números quentes e frios"""
        # Números quentes (mais sorteados nos últimos 10 concursos)
        last_10 = self.dados.tail(10)
        hot_nums = Counter()
        for _, row in last_10.iterrows():
            hot_nums.update(row[[f'B{i}' for i in range(1,16)]].values)
        hot_pool = [num for num, _ in hot_nums.most_common(hot_numbers)]

        # Números frios (não sorteados há mais de 15 concursos)
        cold_pool = []
        all_drawn = set()
        for _, row in self.dados.tail(15).iterrows():
            all_drawn.update(row[[f'B{i}' for i in range(1,16)]].values)
        cold_pool = [n for n in range(1, 26) if n not in all_drawn][:cold_numbers]

        # Preenche o restante aleatoriamente
        remaining = 15 - len(hot_pool) - len(cold_pool)
        middle_pool = [n for n in range(1, 26) if n not in hot_pool and n not in cold_pool]
        selected = hot_pool + cold_pool + random.sample(middle_pool, remaining)
        
        return sorted(selected)

# Inicializa o bot
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

# Handlers do Telegram
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        AVISO_LEGAL,
        parse_mode='HTML'
    )
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
    if not rate_limit(update, "aposta"):
        return
    if bot.dados is None or len(bot.dados) == 0:
        update.message.reply_text("❌ Dados indisponíveis. Use /atualizar ou aguarde atualização dos dados.")
        logger.error("Dados indisponíveis ao tentar gerar apostas.")
        return

    progress_msg = None
    progress_job = None
    try:
        # --- INICIA A BARRA ---
        chat_id = update.effective_chat.id
        progress_msg, progress_job = _start_progress(context, chat_id)

        n_apostas = int(context.args[0]) if context.args and context.args[0].isdigit() else 5
        n_apostas = max(1, min(n_apostas, 10))

        engine_label = "precisa"
        try:
            apostas = bot.gerar_aposta_precisa_com_retry(n_apostas=n_apostas, seed=None, retries=2)
        except Exception as e_precisa:
            logger.warning(f"Falha no gerador preciso (com retry). Usando fallback clássico. Detalhe: {e_precisa}")
            apostas = bot.gerar_aposta(n_apostas)
            engine_label = "clássico (fallback)"

        # --- PARA A BARRA COM SUCESSO ---
        if progress_job and progress_msg:
            _stop_progress(progress_job, progress_msg, "✅ <b>Geração concluída.</b>")

        # Envia o resultado
        mensagem = f"🎲 Apostas recomendadas — engine: {engine_label} 🎲\n\n"
        for i, aposta in enumerate(apostas, 1):
            pares = sum(1 for n in aposta if n % 2 == 0)
            soma = sum(aposta)
            mensagem += (
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"Pares: {pares} | Ímpares: {15-pares} | Soma: {soma}\n\n"
            )
        update.message.reply_text(mensagem, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Erro ao gerar apostas: {str(e)}")
        # --- PARA A BARRA COM ERRO ---
        if progress_job and progress_msg:
            _stop_progress(progress_job, progress_msg, "❌ <b>Falha na geração.</b>")
        update.message.reply_text("❌ Ocorreu um erro ao gerar as apostas. Tente novamente.")

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

        # Frequência de todos os números de 1 a 25 (zero incluído)
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

        # Revalida engine precisa após atualizar os dados/modelo
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
            update.message.reply_text("❌ Uso correto: /inserir YYYY-MM-DD D1 D2 ... D15\nExemplo: /inserir 2025-08-08 01 03 05 07 09 10 12 14 17 18 19 20 22 23 25")
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

        # ------------------------------------------------------------------
        # ⚠️ SUGESTÃO (opcional): validar duplicidade de data antes de inserir
        # Não remove nada do fluxo original; apenas evita inserir se já existir.
        try:
            data_dt = pd.to_datetime(data, format="%Y-%m-%d", errors="raise")
        except Exception:
            update.message.reply_text("❌ Data inválida ao normalizar. Utilize o formato YYYY-MM-DD.")
            return
        # ------------------------------------------------------------------

        arq = 'lotofacil_historico.csv'
        if not os.path.exists(arq):
            update.message.reply_text("❌ Arquivo lotofacil_historico.csv não encontrado no servidor.")
            return
        df = pd.read_csv(arq)

        # ------------------------------------------------------------------
        # ⚠️ SUGESTÃO (opcional): checar se já existe concurso com a MESMA data
        try:
            if 'data' in df.columns and not df.empty:
                df_datas = pd.to_datetime(df['data'], format="%Y-%m-%d", errors="coerce")
                if df_datas.isna().any():
                    alt = pd.to_datetime(df['data'], dayfirst=True, errors="coerce")
                    df_datas = df_datas.fillna(alt)
                if (df_datas.dt.date == data_dt.date()).any():
                    update.message.reply_text(
                        "⚠️ Já existe um concurso com esta data no histórico.\n"
                        "Para evitar duplicidade, a inserção foi cancelada.\n"
                        "Se for realmente um novo registro da mesma data, ajuste a data ou edite o CSV manualmente."
                    )
                    return
        except Exception:
            # Se algo der errado, não bloqueia a operação; apenas segue.
            pass
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # ✅ (opcional) Bloquear inserção se as 15 dezenas já existirem em algum concurso
        try:
            if not df.empty:
                alvo = frozenset(int(x) for x in dezenas_int)
                # percorre apenas colunas B1..B15 se existirem
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
                            # se alguma linha estiver suja, ignora e continua
                            continue
                    if repetida:
                        update.message.reply_text(
                            "⚠️ Já existe um concurso com exatamente as mesmas 15 dezenas.\n"
                            "A inserção foi cancelada para evitar duplicidade de combinação."
                        )
                        return
        except Exception:
            # Qualquer falha aqui não impede a inserção; é check opcional.
            pass
        # ------------------------------------------------------------------

        # Próximo número seguro mesmo com CSV vazio ou 'numero' nulo
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

        # Recalcula campos de repetidos corretamente
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
        backup_csv()  # Faz backup automático antes de salvar o CSV final
        df.to_csv(arq, index=False, encoding='utf-8')

        update.message.reply_text(
            f"✅ Resultado inserido com sucesso!\nConcurso: {proximo_numero}\nData: {data}\nDezenas: {' '.join(str(d).zfill(2) for d in dezenas_int)}"
        )
    except Exception as e:
        logger.error(f"Erro ao inserir resultado: {str(e)}")
        update.message.reply_text("❌ Falha ao inserir o resultado. Tente novamente.")

def comando_meuid(update: Update, context: CallbackContext) -> None:
    """Handler para comando /meuid (aberto para todos)"""
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
        # Envia o manual do usuário automaticamente
        context.bot.send_message(
            chat_id=user_id,
            text=MANUAL_USUARIO,
            parse_mode='HTML'
        )
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
    logger.error(f"Erro no bot: {str(context.error)}")
    if update and update.message:
        update.message.reply_text("❌ Ocorreu um erro inesperado. Os administradores foram notificados.")

def main() -> None:
    """Função principal para iniciar o bot"""
    try:
        updater = Updater(TOKEN)
        dp = updater.dispatcher
        # Comandos
        dp.add_handler(CommandHandler("start", start))
        dp.add_handler(CommandHandler("meuid", comando_meuid))
        dp.add_handler(CommandHandler("aposta", comando_aposta))
        dp.add_handler(CommandHandler("tendencia", comando_tendencia))
        dp.add_handler(CommandHandler("analise", comando_analise))
        dp.add_handler(CommandHandler("atualizar", comando_atualizar))
        dp.add_handler(CommandHandler("inserir", comando_inserir))
        dp.add_handler(CommandHandler("status", comando_status))
        dp.add_handler(CommandHandler("autorizar", comando_autorizar))
        dp.add_handler(CommandHandler("remover", comando_remover))
        # Tratamento de erros
        dp.add_error_handler(error_handler)
        # Inicia o bot
        updater.start_polling()
        logger.info("Bot iniciado e aguardando comandos...")
        updater.idle()
    except Exception as e:
        logger.error(f"Erro fatal ao iniciar o bot: {str(e)}")

if __name__ == "__main__":

    main()









