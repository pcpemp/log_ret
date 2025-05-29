# analysis_module.py

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import traceback

def get_stock_data(ticker, start_date, end_date):
    # ... (mesma função de antes) ...
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"Erro ao buscar dados históricos: Nenhum dado encontrado para o ticker '{ticker}' no período especificado.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if 'Close' not in data.columns:
             if 'Adj Close' in data.columns: data['Close'] = data['Adj Close']
             else:
                 print(f"Erro ao buscar dados históricos: Coluna 'Close' não encontrada nos dados baixados após achatar colunas para '{ticker}'.")
                 return None

        initial_rows_before_dropna = len(data)
        data = data.dropna(subset=['Close'])
        if len(data) < initial_rows_before_dropna:
            print(f"Aviso ao buscar dados históricos: {initial_rows_before_dropna - len(data)} linhas removidas devido a NaNs na coluna 'Close' para '{ticker}'.")

        if data.empty:
             print(f"Erro ao buscar dados históricos: Nenhum dado válido para o ticker '{ticker}' no período especificado após tratamento de NaNs.")
             return None

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        return data[['Close']]

    except Exception as e:
        print(f"Erro inesperado ao obter dados históricos para o ticker '{ticker}': {e}")
        traceback.print_exc()
        return None

def get_stock_info(ticker):
    # ... (mesma função de antes) ...
    """
    Busca informações adicionais para um ticker no Yahoo Finance.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return info
    except Exception as e:
        print(f"Erro ao obter informações para o ticker '{ticker}': {e}")
        return None


# Retorna o DataFrame para plotagem, componentes para projeção E dados de resíduos E taxa de crescimento
def calculate_log_regression(data):
    """
    Calcula a regressão logarítmica, o canal de regressão baseado nos resíduos máximos/mínimos,
    e a taxa de crescimento anualizada.
    Retorna o DataFrame para plotagem, componentes para projeção E dados de resíduos E taxa de crescimento.
    """
    # Adicionado None extra no retorno de erro
    if data is None or data.empty or 'Close' not in data.columns:
        print("Erro: Dados inválidos ou vazios para calcular a regressão.")
        return None, None, None, None, None, None, None, None

    df = data.copy()
    df['log_close'] = np.log(df['Close'] + 1e-9)
    df['time'] = np.arange(len(df))

    initial_rows = len(df)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_close', 'time']).copy()
    df_clean.index = pd.to_datetime(df_clean.index)
    df_clean = df_clean.sort_index()

    # Mínimo de pontos necessários para OLS (2 parâmetros: intercepto e coef. do tempo)
    # Adicionado None extra no retorno de erro
    if len(df_clean) < 2:
        print(f"Erro: Dados insuficientes para calcular regressão após limpeza. Mínimo de 2 pontos necessários, encontrados {len(df_clean)}.")
        return None, None, None, None, None, None, None, None

    if len(df_clean) < initial_rows:
        print(f"Aviso: {initial_rows - len(df_clean)} linhas removidas durante a limpeza após transformação logarítmica.")
        df_clean.loc[:, 'time'] = np.arange(len(df_clean))

    # Inicializa a taxa de crescimento como None, caso a regressão falhe
    annualized_growth_rate = None

    try:
        # 3. Ajustar o modelo nos dados LIMPOS
        model = smf.ols('log_close ~ time', data=df_clean).fit()

        # --- CALCULAR TAXA DE CRESCIMENTO ANUALIZADA ---
        # O coeficiente de 'time' é a inclinação (beta_1)
        beta_1 = model.params['time']
        # Estimativa de 252 dias de negociação por ano
        annualized_growth_factor = np.exp(beta_1 * 252)
        annualized_growth_rate = (annualized_growth_factor - 1) * 100 # Em porcentagem
        # --- FIM CALCULAR TAXA DE CRESCIMENTO ---


        # Obter a linha central da regressão (em escala logarítmica) para os dados limpos
        df_clean.loc[:, 'log_predicted_close'] = model.predict(df_clean['time'])

        df_clean.loc[:, 'log_residuals'] = df_clean['log_close'] - df_clean['log_predicted_close']

        max_log_residual = df_clean['log_residuals'].max()
        min_log_residual = df_clean['log_residuals'].min()

        df_clean.loc[:, 'log_upper_outer_channel'] = df_clean['log_predicted_close'] + max_log_residual
        df_clean.loc[:, 'log_lower_outer_channel'] = df_clean['log_predicted_close'] + min_log_residual
        df_clean.loc[:, 'log_upper_inner_channel'] = df_clean['log_predicted_close'] + max_log_residual / 2
        df_clean.loc[:, 'log_lower_inner_channel'] = df_clean['log_predicted_close'] + min_log_residual / 2

        df_clean.loc[:, 'predicted_close'] = np.exp(df_clean['log_predicted_close'])
        df_clean.loc[:, 'upper_outer_channel'] = np.exp(df_clean['log_upper_outer_channel'])
        df_clean.loc[:, 'lower_outer_channel'] = np.exp(df_clean['log_lower_outer_channel'])
        df_clean.loc[:, 'upper_inner_channel'] = np.exp(df_clean['log_upper_inner_channel'])
        df_clean.loc[:, 'lower_inner_channel'] = np.exp(df_clean['log_lower_inner_channel'])

        df_plot = df_clean[[
            'Close',
            'predicted_close',
            'upper_outer_channel',
            'lower_outer_channel',
            'upper_inner_channel',
            'lower_inner_channel'
        ]]

        log_residuals_series = df_clean['log_residuals']

        current_log_residual = log_residuals_series.iloc[-1] if not log_residuals_series.empty else None
        current_actual_price = df_clean['Close'].iloc[-1] if not df_clean.empty else None


        # Retornamos uma tupla com todos os resultados necessários, incluindo a taxa de crescimento
        return df_plot, model, max_log_residual, min_log_residual, log_residuals_series, current_log_residual, current_actual_price, annualized_growth_rate

    except Exception as e:
        print(f"Erro no cálculo da regressão ou canais para o ticker '{ticker}': {e}")
        traceback.print_exc()
        # Retornamos None para todos os valores em caso de erro
        return None, None, None, None, None, None, None, None


def project_log_channel(model, max_log_residual, min_log_residual, df_clean_length):
    # ... (função project_log_channel permanece a mesma) ...
    """
    Projeta as linhas do canal de regressão logarítmica para o futuro.
    Retorna os valores projetados em preço absoluto.
    """
    if model is None or max_log_residual is None or min_log_residual is None or df_clean_length is None:
        return None

    days_in_3_months = 63
    days_in_6_months = 126
    days_in_9_months = 189
    days_in_12_months = 252

    current_time_index = df_clean_length - 1
    projection_time_indices = {
        'Atual': current_time_index, # Corresponde ao último ponto histórico
        'Em 3 meses': current_time_index + days_in_3_months,
        'Em 6 meses': current_time_index + days_in_6_months,
        'Em 9 meses': current_time_index + days_in_9_months,
        'Em 12 meses': current_time_index + days_in_12_months,
    }

    line_names = [
        'Canal Exterior Superior',
        'Canal Interior Superior',
        'Linha Central',
        'Canal Interior Inferior',
        'Canal Exterior Inferior',
    ]

    projected_prices = {line: [] for line in line_names}

    for period, time_index_value in projection_time_indices.items():
        exog_predict_df = pd.DataFrame({'time': [time_index_value]})
        log_pred_center = model.predict(exog_predict_df)[0]

        log_pred_upper_outer = log_pred_center + max_log_residual
        log_pred_lower_outer = log_pred_center + min_log_residual
        log_pred_upper_inner = log_pred_center + max_log_residual / 2
        log_pred_lower_inner = log_pred_center + min_log_residual / 2 # CORRIGIDO NESTA FUNÇÃO


        pred_center = np.exp(log_pred_center)
        pred_upper_outer = np.exp(log_pred_upper_outer)
        pred_lower_outer = np.exp(log_pred_lower_outer)
        pred_upper_inner = np.exp(log_pred_upper_inner)
        pred_lower_inner = np.exp(log_pred_lower_inner) # CORRIGIDO: Usando log_pred_lower_inner

        projected_prices['Canal Exterior Superior'].append(pred_upper_outer)
        projected_prices['Canal Interior Superior'].append(pred_upper_inner)
        projected_prices['Linha Central'].append(pred_center)
        projected_prices['Canal Interior Inferior'].append(pred_lower_inner)
        projected_prices['Canal Exterior Inferior'].append(pred_lower_outer)

    projected_df_prices = pd.DataFrame(projected_prices, index=projection_time_indices.keys()).T

    return projected_df_prices