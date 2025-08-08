import os
import requests
import pandas as pd
import numpy as np
import random
import hashlib
import pickle
from datetime import datetime
import shutil
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext
import logging
from io import BytesIO
import warnings
from typing import Optional, Dict, List, Tuple, Set
from time import time

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

def backup_csv():
    origem = 'lotofacil_historico.csv'
    if os.path.exists(origem):
        destino = f"lotofacil_historico_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy2(origem, destino)
        logger.info(f"Backup automático criado: {destino}")

# Configuração inicial
warnings.filterwarnings("ignore", message="oneDNN custom operations are on")
logging.basicConfig(
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TOKEN", "").strip()  # TOKEN virá das variáveis da Railway
if not TOKEN:
    raise RuntimeError("TOKEN não definido no ambiente.")

# Diretório persistente no Volume da Railway
DATA_DIR = os.getenv("DATA_DIR")  # ex.: /data
if DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)
    # Copia arquivos iniciais para o volume se não existirem
    for _fn in ["lotofacil_historico.csv", "whitelist.txt", "modelo_lotofacil_avancado.h5"]:
        src = os.path.join(os.getcwd(), _fn)
        dst = os.path.join(DATA_DIR, _fn)
        if (not os.path.exists(dst)) and os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Aviso: não foi possível copiar {src} -> {dst}: {e}")
    # Muda diretório de trabalho para o volume
    os.chdir(DATA_DIR)

ADMIN_USER_IDS = [5344714174]  # IDs dos administradores

# Rate limit por usuário/comando
rate_limit_map = {}

def rate_limit(update, comando, segundos=8):
    user_id = update.effective_user.id
    agora = time()
    if user_id not in rate_limit_map:
        rate_limit_map[user_id] = {}
    if comando in rate_limit_map[user_id]:
        if agora - rate_limit_map[user_id][comando] < segundos:
            update.message.reply_text("⏳ Aguarde alguns segundos antes de usar novamente.")
            return False
    rate_limit_map[user_id][comando] = agora
    return True

class SecurityManager:
    def __init__(self):
        self.whitelist: Set[int] = set()
        self.admins: Set[int] = set(ADMIN_USER_IDS)
        self.load_whitelist()
    def load_whitelist(self, file: str = "whitelist.txt") -> None:
        try:
            if os.path.exists(file):
                with open(file, "r") as f:
                    self.whitelist = {int(line.strip()) for line in f if line.strip().isdigit()}
        except Exception as e:
            logger.error(f"Erro ao carregar whitelist: {str(e)}")
    def is_admin(self, user_id: int) -> bool:
        return user_id in self.admins
    def is_authorized(self, user_id: int) -> bool:
        return user_id in self.whitelist or self.is_admin(user_id)

class DataFetcher:
    """Gerencia obtenção de dados com fallback e cache"""
    API_URLS = [
        "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil",
        "https://api-loterias.herokuapp.com/api/v1/lotofacil"
    ]
    
    @staticmethod
    def fetch_data(url: str, timeout: int = 10) -> Optional[Dict]:
        """Tenta obter dados de uma URL com tratamento de erros"""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Falha ao acessar {url}: {str(e)}")
        return None
    
    @classmethod
    def get_latest_data(cls) -> Optional[Dict]:
        """Tenta obter dados de múltiplas fontes"""
        for url in cls.API_URLS:
            data = cls.fetch_data(url)
            if data and cls.validate_data(data):
                return data
        return None
    
    @staticmethod
    def validate_data(data: Dict) -> bool:
        """Valida estrutura dos dados recebidos"""
        required_keys = {'numero', 'data', 'dezenas'}
        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data['dezenas'], list) or len(data['dezenas']) != 15:
            return False
        try:
            return all(1 <= int(n) <= 25 for n in data['dezenas'])
        except (ValueError, TypeError):
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

            # Calcula corretamente os repetidos de 1 a 5 concursos anteriores
            for rep in range(1, 6):
                repetidos = []
                for idx, row in df.iterrows():
                    if idx < rep:
                        repetidos.append(0)
                    else:
                        nums_atual = set([row[f'B{i}'] for i in range(1, 16)])
                        nums_anterior = set([df.iloc[idx - rep][f'B{i}'] for i in range(1, 16)])
                        repetidos.append(len(nums_atual & nums_anterior))
                df[f'repetidos_{rep}'] = repetidos

            # Ordena por número do concurso (do menor para o maior) e reseta o índice
            df = df.sort_values('numero').reset_index(drop=True)

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
        """Calcula matriz de coocorrência com pesos temporais"""
        cooc = np.zeros((25, 25))
        for i in range(1, len(self.dados)):
            nums_atual = set(self.dados.iloc[i][[f'B{j}' for j in range(1,16)]].values)
            nums_anterior = set(self.dados.iloc[i-1][[f'B{k}' for k in range(1,16)]].values)
            
            for num1 in nums_atual:
                for num2 in nums_anterior:
                    cooc[num1-1, num2-1] += 1 / (i ** 0.5)  # Peso decrescente
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
        """Identifica clusters dinâmicos com cache"""
        cache_file = os.path.join(self.cache_dir, "clusters_cache.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache de clusters corrompido. Recriando... Erro: {str(e)}")
        
        dados_cluster = self.dados[[f'B{i}' for i in range(1,16)]]
        kmeans = KMeans(n_clusters=4, random_state=42).fit(dados_cluster)
        clusters = {i: [] for i in range(4)}
        for num in range(1, 26):
            cluster = kmeans.predict([[num]*15])[0]
            clusters[cluster].append(num)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(clusters, f)
        except Exception as e:
            logger.error(f"Falha ao salvar cache de clusters: {str(e)}")
        
        return clusters
    
    def construir_modelo(self) -> Optional[tf.keras.Model]:
        """Constroi modelo LSTM avançado com otimizações"""
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
        
        model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early, checkpoint],
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
    
    def gerar_aposta(self, n_apostas: int = 5) -> List[List[int]]:
        """Gera apostas otimizadas com múltiplas estratégias"""
        apostas = []
        for _ in range(n_apostas):
            # 1. Geração por modelo neural
            aposta_modelo = self.gerar_por_modelo()
            
            # 2. Geração por algoritmo genético
            aposta_ga = self.gerar_por_algoritmo_genetico()
            
            # 3. Combinação inteligente
            aposta_final = self.combinar_apostas(aposta_modelo, aposta_ga)
            apostas.append(aposta_final)
        
        # Aplica fechamento matemático
        return self.aplicar_fechamento(apostas)
    
    def gerar_por_modelo(self) -> List[int]:
        """Gera aposta usando o modelo LSTM"""
        ultimos_numeros = self.dados[[f'B{i}' for i in range(1,16)]].values[-10:]
        X = np.zeros((1, 10, 25 + 5))  # 25 números + 5 features
        
        for i in range(10):
            for num in ultimos_numeros[i]:
                X[0, i, num-1] = 1
            for j in range(1,6):
                col_name = f'repetidos_{j}'
                if col_name in self.dados.columns:
                    X[0, i, 25 + j - 1] = self.dados.iloc[-i][col_name] if i > 0 else 0
                else:
                    X[0, i, 25 + j - 1] = 0
        
        pred = self.modelo.predict(X, verbose=0)[0]
        return sorted([i+1 for i in np.argsort(pred)[-15:]])
    
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
        """Função de fitness para o algoritmo genético"""
        aposta = list(set(aposta))
        if len(aposta) != 15:
            return (0,)
        
        score = 0
        
        # Pontua por frequência histórica
        score += sum(self.frequencias[n] for n in aposta)
        
        # Pontua por coocorrência
        for i in aposta:
            for j in aposta:
                if i != j:
                    score += self.coocorrencias[i-1, j-1] * 0.1
        
        # Pontua por clusters (2-4 números por cluster)
        for cluster in self.clusters.values():
            intersect = set(aposta) & set(cluster)
            if 2 <= len(intersect) <= 4:
                score += 30
        
        # Penaliza se não seguir padrões estatísticos
        pares = sum(1 for n in aposta if n % 2 == 0)
        soma = sum(aposta)
        
        if not (6 <= pares <= 8):
            score *= 0.5
        if not (185 <= soma <= 215):
            score *= 0.5
            
        # Verifica sequência inicial
        seq_inicial = tuple(sorted(aposta)[:3])
        score += self.sequencias_iniciais.get(seq_inicial, 0) * 0.5
        
        return (score,)
    
    def combinar_apostas(self, aposta1: List[int], aposta2: List[int]) -> List[int]:
        """Combina duas apostas de forma inteligente"""
        comuns = set(aposta1) & set(aposta2)
        diferentes = list((set(aposta1) | set(aposta2)) - comuns)
        random.shuffle(diferentes)
        
        nova_aposta = list(comuns) + diferentes[:15 - len(comuns)]
        return sorted(nova_aposta)
    
    def aplicar_fechamento(self, apostas: List[List[int]]) -> List[List[int]]:
        """Aplica sistema de fechamento para cobertura de números"""
        todos_numeros = list(range(1, 26))
        cobertura = Counter()
        
        # Prioriza números menos sorteados nas apostas
        for aposta in apostas:
            cobertura.update(aposta)
        
        # Garante que todos números tenham chance
        for num in todos_numeros:
            if cobertura[num] == 0:
                # Substitui o número menos útil em uma das apostas
                aposta_menor_score = min(apostas, key=lambda a: sum(self.frequencias[n] for n in a))
                substituto = random.choice([n for n in aposta_menor_score if cobertura[n] > 1])
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
    try:
        n_apostas = int(context.args[0]) if context.args and context.args[0].isdigit() else 5
        n_apostas = max(1, min(n_apostas, 10))
        apostas = bot.gerar_aposta(n_apostas)
        mensagem = "🎲 Apostas recomendadas 🎲\n\n"
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
        update.message.reply_text("✅ Dados e modelo atualizados com sucesso!")
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
        status = (
            "<b>📊 Status do Sistema</b>\n\n"
            f"<b>Concursos carregados:</b> {len(bot.dados)}\n"
            f"<b>Último concurso:</b> {ultimo['numero']} ({ultimo['data'].strftime('%d/%m/%Y')})\n"
            f"<b>Modelo IA:</b> {'Carregado' if hasattr(bot, 'modelo') and bot.modelo is not None else 'Não treinado'}\n"
            f"<b>Números mais quentes:</b> {', '.join(str(n) for n, _ in bot.frequencias.most_common(3))}\n"
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
        arq = 'lotofacil_historico.csv'
        if not os.path.exists(arq):
            update.message.reply_text("❌ Arquivo lotofacil_historico.csv não encontrado no servidor.")
            return
        df = pd.read_csv(arq)
        proximo_numero = int(df['numero'].max()) + 1 if 'numero' in df.columns else len(df) + 1
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
                    nums_atual = set([row[f'B{i}'] for i in range(1, 16)])
                    nums_anterior = set([df.iloc[idx - rep][f'B{i}'] for i in range(1, 16)])
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

def error_handler(update: Update, context: CallbackContext) -> None:
    """Handler para erros não tratados"""
    logger.error(f"Erro no bot: {str(context.error)}")
    if update and update.message:
        update.message.reply_text("❌ Ocorreu um erro inesperado. Os administradores foram notificados.")

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