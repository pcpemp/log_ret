# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import analysis_module # Importa o módulo de análise

# --- Funções Auxiliares de Formatação ---
def format_market_cap(market_cap):
    """Formata Market Cap em Bilhões."""
    if market_cap is None:
        return "Não disponível"
    return f"${market_cap / 1e9:,.2f} Bilhões".replace(',', '.')

def format_price(price):
    """Formata preço com 2 casas decimais."""
    if pd.isna(price):
         return "-"
    return f'{price:,.2f}'.replace(',', '.')

def format_percentage(percentage):
    """Formata porcentagem com 2 casas decimais e sinal."""
    if pd.isna(percentage):
        return "-"
    return f'{percentage:+.2f}%'.replace(',', '.')

def format_growth_rate(rate):
    """Formata taxa de crescimento anualizada em porcentagem."""
    if rate is None or pd.isna(rate):
        return "Não calculada"
    # Reusa format_percentage, mas sem o sinal de + se for positivo (opcional, mas comum para crescimento)
    # Ou mantém o sinal com format_percentage
    return f'{rate:.2f}%'.replace(',', '.') # Exemplo sem sinal + para positivo

# --- Configurações da Página ---
st.set_page_config(
    page_title="Análise de Ações com Canal de Regressão Logarítmica",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inicialização de st.session_state ---
# Inicializa as variáveis que armazenarão os resultados da análise
if 'analysis_performed' not in st.session_state:
    st.session_state.analysis_performed = False
    st.session_state.stock_info = None
    st.session_state.regression_data = None
    st.session_state.model = None
    st.session_state.max_log_residual = None
    st.session_state.min_log_residual = None
    st.session_state.log_residuals = None
    st.session_state.current_residual = None
    st.session_state.current_actual_price = None
    st.session_state.projection_table_prices = None
    st.session_state.annualized_growth_rate = None # Adicionado


# --- Título da Aplicação ---
st.title("📈 Análise de Ações com Canal de Regressão Logarítmica")
st.markdown("""
Esta aplicação visualiza o preço histórico de uma ação juntamente com um canal
de regressão logarítmica baseado nos desvios extremos observados nos dados,
projeta esses valores para o futuro e mostra a distribuição dos resíduos históricos.
""")

# --- Interface do Usuário (Sidebar) ---
st.sidebar.header("Parâmetros de Análise")

ticker = st.sidebar.text_input("Insira o Ticker da Ação", "PETR4.SA").upper()
today = date.today()
default_start_date = today - timedelta(days=5*365)
start_date = st.sidebar.date_input("Data de Início", default_start_date)
end_date = st.sidebar.date_input("Data de Fim", today)

if start_date >= end_date:
    st.sidebar.error("Erro: A data de início deve ser anterior à data de fim.")

analyze_button = st.sidebar.button("Analisar Ação")

# --- Lógica Principal (Quando o botão é clicado) ---
# Este bloco SÓ RODA quando o botão é clicado. Faz a análise e SALVA no session_state.
if analyze_button:
    # Limpar resultados anteriores no session_state
    st.session_state.analysis_performed = False
    st.session_state.stock_info = None
    st.session_state.regression_data = None
    st.session_state.model = None
    st.session_state.max_log_residual = None
    st.session_state.min_log_residual = None
    st.session_state.log_residuals = None
    st.session_state.current_residual = None
    st.session_state.current_actual_price = None
    st.session_state.projection_table_prices = None
    st.session_state.annualized_growth_rate = None # Limpa também a taxa de crescimento

    if start_date >= end_date:
        st.error("Corrija as datas antes de analisar.")
    elif not ticker:
         st.error("Por favor, insira um ticker de ação.")
    else:
        with st.spinner(f"Buscando dados e informações para {ticker}..."):
            # Obter informações adicionais do ticker
            st.session_state.stock_info = analysis_module.get_stock_info(ticker)

            # Obter dados históricos da ação
            stock_data = analysis_module.get_stock_data(ticker, start_date, end_date)

            if stock_data is None:
                st.error(f"Não foi possível obter dados históricos para o ticker '{ticker}'. Verifique o ticker e o período.")
            elif len(stock_data) < 2:
                 st.warning(f"Aviso: Foram encontrados apenas {len(stock_data)} dias de dados válidos. Mínimo necessário é 2 para regressão. Não é possível calcular o canal.")
                 if not stock_data.empty:
                      st.line_chart(stock_data['Close']) # Mostra o gráfico de preço se houver algum dado
            else:
                # Calcular Regressão Logarítmica e Canal (Recebe múltiplos retornos)
                # Recebe a taxa de crescimento também
                (
                    st.session_state.regression_data,
                    st.session_state.model,
                    st.session_state.max_log_residual,
                    st.session_state.min_log_residual,
                    st.session_state.log_residuals,
                    st.session_state.current_residual,
                    st.session_state.current_actual_price,
                    st.session_state.annualized_growth_rate # Recebe a taxa de crescimento
                ) = analysis_module.calculate_log_regression(stock_data)

                if st.session_state.regression_data is None: # Verifica se o primeiro retorno (df_plot) é None
                     st.error("Erro ao calcular o canal de regressão. Certifique-se de que há dados suficientes e válidos para o período.")
                else:
                    # Calcular a tabela de projeção (preços absolutos) APENAS UMA VEZ aqui
                    st.session_state.projection_table_prices = analysis_module.project_log_channel(
                        st.session_state.model,
                        st.session_state.max_log_residual,
                        st.session_state.min_log_residual,
                        len(st.session_state.regression_data) # Passamos o comprimento dos dados limpos
                    )

                    # Se chegamos aqui com sucesso e temos dados para projeção e preço atual
                    if st.session_state.projection_table_prices is not None and st.session_state.current_actual_price is not None:
                         st.session_state.analysis_performed = True
                    else:
                        st.warning("Análise de projeção ou preço atual falhou após cálculo da regressão.")
                        st.session_state.analysis_performed = False # Marca como não realizada se projeção falhou


# --- Lógica de Exibição (Este bloco roda a CADA rerun se analysis_performed for True) ---
# Tudo aqui acessa os dados do session_state, persistindo entre reruns de widgets.
if st.session_state.analysis_performed:
    # Exibir informações adicionais da empresa (se disponíveis)
    if st.session_state.stock_info:
        company_name = st.session_state.stock_info.get('longName', ticker)
        market_cap = st.session_state.stock_info.get('marketCap')
        st.subheader(f"{company_name} ({ticker})")
        if market_cap is not None:
            st.markdown(f"**Capitalização de Mercado:** {format_market_cap(market_cap)}")
        # Exibir a taxa de crescimento anualizada aqui
        st.markdown(f"**Taxa de Crescimento Anualizada (Regressão Log):** {format_growth_rate(st.session_state.annualized_growth_rate)}")

    else:
         st.subheader(f"Análise para {ticker}")
         # Exibir a taxa de crescimento mesmo se info da empresa falhar, se estiver calculada
         if st.session_state.annualized_growth_rate is not None:
             st.markdown(f"**Taxa de Crescimento Anualizada (Regressão Log):** {format_growth_rate(st.session_state.annualized_growth_rate)}")


    # 3. Visualização Gráfica do Preço e Canais (Plotly)
    st.subheader(f"Gráfico de Preço e Canal de Regressão Logarítmica")

    fig = go.Figure()

    # --- Adicionar traces com as NOVAS CORES E ESTILOS ---
    # Preço de Fechamento (Preto Cheio)
    fig.add_trace(go.Scattergl(
        x=st.session_state.regression_data.index,
        y=st.session_state.regression_data['Close'],
        mode='lines',
        name='Preço de Fechamento',
        line=dict(color='black', width=1, dash='solid')
    ))

    # Linha Central da Regressão (Preto Pontilhado)
    fig.add_trace(go.Scattergl(
        x=st.session_state.regression_data.index,
        y=st.session_state.regression_data['predicted_close'],
        mode='lines',
        name='Linha Central do Canal',
        line=dict(color='black', dash='dot', width=2)
    ))

    # Canal Exterior Superior (Azul Pontilhado)
    fig.add_trace(go.Scattergl(
        x=st.session_state.regression_data.index,
        y=st.session_state.regression_data['upper_outer_channel'],
        mode='lines',
        name='Canal Exterior Superior',
        line=dict(color='blue', width=1, dash='dot'),
        showlegend=True
    ))

    # Canal Interior Superior (Azul Cheio)
    fig.add_trace(go.Scattergl(
        x=st.session_state.regression_data.index,
        y=st.session_state.regression_data['upper_inner_channel'],
        mode='lines',
        name='Canal Interior Superior',
        line=dict(color='blue', width=1, dash='solid'),
        showlegend=True
    ))

    # Canal Interior Inferior (Vermelho Cheio)
    fig.add_trace(go.Scattergl(
         x=st.session_state.regression_data.index,
         y=st.session_state.regression_data['lower_inner_channel'],
         mode='lines',
         name='Canal Interior Inferior',
         showlegend=True,
         line=dict(color='red', width=1, dash='solid')
     ))

    # Canal Exterior Inferior (Vermelho Pontilhado)
    fig.add_trace(go.Scattergl(
         x=st.session_state.regression_data.index,
         y=st.session_state.regression_data['lower_outer_channel'],
         mode='lines',
         name='Canal Exterior Inferior',
         showlegend=True,
         line=dict(color='red', width=1, dash='dot')
     ))
    # --- Fim NOVOS ESTILOS DE LINHA ---


    # Configurações do Layout do Gráfico de Preço
    fig.update_layout(
        title=f"Canal de Regressão Logarítmica<br><sup>Período: {start_date} a {end_date}</sup>",
        xaxis_title="Data",
        yaxis=dict(
            title="Preço (Escala Logarítmica)",
            type='log'
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        height=500
    )

    # Exibir o gráfico de preço
    st.plotly_chart(fig, use_container_width=True)


    # --- 4. Visualização da Distribuição dos Resíduos ---
    if st.session_state.log_residuals is not None and not st.session_state.log_residuals.empty:
         st.subheader("Distribuição dos Resíduos Logarítmicos")

         fig_residuals = px.histogram(st.session_state.log_residuals, nbins=50, labels={'value': 'Valor do Resíduo Logarítmico', 'count': 'Frequência'}, title='Histograma dos Resíduos Históricos')

         if st.session_state.current_residual is not None:
             fig_residuals.add_vline(x=st.session_state.current_residual, line_dash="dash", line_color="red", line_width=2, annotation_text=f"Resíduo Atual: {st.session_state.current_residual:.4f}", annotation_position="top right")

         fig_residuals.update_layout(xaxis_title='Valor do Resíduo Logarítmico', yaxis_title='Frequência')

         st.plotly_chart(fig_residuals, use_container_width=True)

    else:
         st.warning("Não foi possível gerar o histograma de resíduos (dados insuficientes ou erro no cálculo).")


    # --- 5. Projetar os Valores e Exibir a Tabela ---
    if st.session_state.projection_table_prices is not None and st.session_state.current_actual_price is not None:
        st.subheader("Projeção do Canal de Regressão Logarítmica")

        view_option = st.radio(
            "Visualização da Tabela:",
            ('Preço Absoluto', 'Variação Percentual'),
            horizontal=True,
            key="projection_view_option"
        )

        if view_option == 'Preço Absoluto':
            st.dataframe(
                st.session_state.projection_table_prices.applymap(format_price),
                use_container_width=True
            )
            st.markdown("""
            <small><i>Os valores na tabela são os preços absolutos projetados.</i></small>
            """, unsafe_allow_html=True)

        elif view_option == 'Variação Percentual':
            if st.session_state.current_actual_price == 0:
                 st.error("Não é possível calcular variação percentual: o preço atual é zero.")
            else:
                # Calcular a variação percentual em relação ao preço atual REAL salvo
                projection_table_percent = (
                    (st.session_state.projection_table_prices - st.session_state.current_actual_price) / st.session_state.current_actual_price
                ) * 100

                st.dataframe(
                    projection_table_percent.applymap(format_percentage),
                    use_container_width=True
                )
                st.markdown("""
                <small><i>Os valores na tabela são a variação percentual em relação ao último preço de fechamento observado.</i></small>
                """, unsafe_allow_html=True)

        st.markdown("""
        <small><i>Os valores projetados são baseados na extensão linear da regressão logarítmica,
        assumindo aproximadamente 21 dias de negociação por mês.
        Isso não é uma previsão baseada em eventos futuros ou análise fundamental.</i></small>
        """, unsafe_allow_html=True)

    else:
         st.warning("Não foi possível calcular a projeção do canal.")


# --- Informações Adicionais (Opcional) ---
# Este bloco roda a cada rerun
st.markdown("""
---
**Observação:** A análise de regressão logarítmica visualiza a tendência histórica
do preço em escala logarítmica. O canal cinza reflete a amplitude dos desvios
históricos em torno dessa tendência. A projeção estende essa tendência
e amplitude para o futuro. O histograma mostra a distribuição estatística
desses desvios. Essas ferramentas são para análise técnica e não devem ser
interpretadas como recomendações de investimento.
""")