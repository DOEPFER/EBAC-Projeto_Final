import streamlit         as st

import numpy as np
import pandas            as pd

from io                  import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
# from sklearn import metrics

# from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline

# from sklearn.metrics import accuracy_score, classification_report

import pickle

from pathlib import Path

custom_params = {"axes.spines.right": False, "axes.spines.top": False}

@st.cache_data(show_spinner=True)
def load_file(file_data):
    try:
        return pd.read_csv(file_data, sep=',')
    except:
        return st.write('Não foi possível ler o arquivo')
    
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# initial treatment
def initial_treatment(dataframe=None):

    dataframe = dataframe.copy()

    accepted_columns = {
        'sexo',
        'posse_de_veiculo',
        'posse_de_imovel',
        'qtd_filhos',
        'tipo_renda',
        'educacao',
        'estado_civil',
        'tipo_residencia',
        'idade', 
        'tempo_emprego',
        'qt_pessoas_residencia',
        'renda',
        'mau'
    }

    present_columns = list(set(dataframe.columns) & accepted_columns)
    
    dataframe = dataframe[present_columns].copy()

    # dataframe.drop(labels='index', axis=1, inplace=True)
    # dataframe.set_index(keys=[dataframe.index, pd.to_datetime(dataframe['data_ref'])], inplace=True)
    # dataframe.drop(labels='data_ref', axis=1, inplace=True)
    # dataframe.sort_index(level=1, inplace=True)
    
    dataframe['sexo'] = dataframe['sexo'].astype('category')
    dataframe['posse_de_veiculo'] = dataframe['posse_de_veiculo'].astype('bool')
    dataframe['posse_de_imovel'] = dataframe['posse_de_imovel'].astype('bool')
    dataframe['qtd_filhos'] = dataframe['qtd_filhos'].astype('int8')
    dataframe['tipo_renda'] = dataframe['tipo_renda'].astype('category')
    dataframe['educacao'] = dataframe['educacao'].astype('category')
    dataframe['estado_civil'] = dataframe['estado_civil'].astype('category')
    dataframe['tipo_residencia'] = dataframe['tipo_residencia'].astype('category')
    dataframe['idade'] = dataframe['idade'].astype('int8')
    dataframe['tempo_emprego'] = dataframe['tempo_emprego'].astype('float32')
    dataframe['qt_pessoas_residencia'] = dataframe['qt_pessoas_residencia'].astype('int8')
    
    return dataframe

# missing values
def missing_values(dataframe=None, columns=None, methods=None):
    
    # dataframe = dataframe.copy()
        
    for column, method in zip(columns, methods):
        if dataframe[column].isna().sum():
            
            _call = getattr(
                dataframe[dataframe[column].isna() == False][column], 
                method
            )
            
            value = _call()
            
            dataframe.loc[dataframe[column].isna(), column] = value
            
    return dataframe

# missing treatment
def missing_treatment(dataframe=None):
    
    dataframe = dataframe.copy()
    
    return missing_values(dataframe=dataframe, columns=['idade', 'tempo_emprego', 'renda'], methods=['mode', 'mean', 'median'])

# outlier detect
def outlier_detect(dataframe=None, column=None, cumulate=0.99):
    
    _bin_count = dataframe[column].value_counts()
    _bin_count.sort_values(ascending = False, inplace = True)
    _bin_count = _bin_count.to_frame()
    _bin_count['freq'] = _bin_count / _bin_count.sum()
    _bin_count['freq_cum'] = _bin_count['freq'].cumsum()

    _bin_interval = _bin_count[_bin_count['freq_cum'] >= cumulate].iloc[1:]

    column_outliers = pd.Series(data=False, index=dataframe.index, name=column)

    if _bin_interval.empty:
        return column_outliers
    
    column_outliers[dataframe[column].isin(_bin_interval.index)] = True

    return column_outliers

# outlier treatment
def outlier_treatment(dataframe=None):
    
    dataframe = dataframe.copy()
    
    dataframe['tempo_emprego_bin'] = round(dataframe['tempo_emprego'], ndigits=0)
    dataframe['renda_bin'] = round(dataframe['renda'], ndigits=-2)
    
    columns_outliers = {
        'sexo':'sexo', 
        'posse_de_veiculo':'posse_de_veiculo', 
        'posse_de_imovel':'posse_de_imovel', 
        'qtd_filhos':'qtd_filhos', 
        'tipo_renda':'tipo_renda', 
        'educacao':'educacao', 
        'estado_civil':'estado_civil', 
        'tipo_residencia':'tipo_residencia', 
        'idade':'idade', 
        'tempo_emprego_bin':'tempo_emprego', 
        'qt_pessoas_residencia':'qt_pessoas_residencia', 
        'renda_bin':'renda'
    }
    
    columns = list(columns_outliers.keys())

    dataframe_outliers = pd.DataFrame()
    for column in columns:
        dataframe_outliers[column] = outlier_detect(dataframe=dataframe, column=column)
    
    _outliers = pd.concat(objs=[dataframe, dataframe_outliers.add_suffix(suffix='_outlier', axis=1)], axis=1)
    
    outliers_count = (dataframe_outliers.sum() > 0)
    columns = outliers_count[outliers_count == True].index

    for column in columns:
        column_outlier_true = _outliers[f'{column}_outlier'].value_counts()[True]

        sample = dataframe[_outliers[f'{column}_outlier'] == False][columns_outliers[column]].sample(
            n = column_outlier_true,
            replace = True,
            random_state = 100
        )

        condition = (_outliers[f'{column}_outlier'] == True)

        dataframe.loc[condition, columns_outliers[column]] = sample.values
        
    dataframe.drop(labels=['tempo_emprego_bin', 'renda_bin'], axis=1, inplace=True)
    
    return dataframe

# iv
def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')

    rótulo_evento = tab.columns[0]
    rótulo_nao_evento = tab.columns[1]

    tab['pct_evento'] = tab[rótulo_evento]/tab.loc['total',rótulo_evento]
    tab['ep'] = tab[rótulo_evento]/tab.loc['total',rótulo_evento]
    
    tab['pct_nao_evento'] = tab[rótulo_nao_evento]/tab.loc['total',rótulo_nao_evento]
    tab['woe'] = np.log(tab.pct_evento/tab.pct_nao_evento)
    tab['iv_parcial'] = (tab.pct_evento - tab.pct_nao_evento)*tab.woe
    return tab['iv_parcial'].sum()

# information value
def information_value(dataframe=None):
    
    dataframe = dataframe.copy()
    
    metadados = pd.DataFrame(dataframe.dtypes, columns=['dtype'])
    metadados['ausentes'] = dataframe.isna().sum()
    metadados['unicos'] = dataframe.nunique()
    metadados['papel'] = 'covariavel'
    metadados.loc['mau','papel'] = 'resposta'
    
    for var in metadados[metadados.papel=='covariavel'].index:
        if  (metadados.loc[var, 'unicos'] > 6):
            metadados.loc[var, 'IV'] = IV(pd.qcut(dataframe[var], 5, duplicates='drop'), dataframe["mau"])
        else: 
            metadados.loc[var, 'IV'] = IV(dataframe[var], dataframe["mau"])
    
    filter = list(metadados[metadados['IV'] >= 0.025].index.values)
    
    return dataframe[filter]

# dummies
def get_dummies_treatment(dataframe=None):
    
    dataframe = dataframe.copy()
    
    return pd.get_dummies(dataframe)

# train
def train_model(file_train):

    st.write('## Treinamento')
    dataframe = pd.read_feather(file_train)
     
    X = dataframe
    y = dataframe['mau']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

    initial = FunctionTransformer(initial_treatment)
    missing = FunctionTransformer(missing_treatment)
    outlier = FunctionTransformer(outlier_treatment)
    informationvalue = FunctionTransformer(information_value)
    dummies = FunctionTransformer(get_dummies_treatment)
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    logistic = LogisticRegression(max_iter=200)

    pipeline = Pipeline(
        steps=[
            ('initial', initial),
            ('missing', missing),
            ('outlier', outlier),
            ('informationvalue', informationvalue),
            ('dummies', dummies),
            ('scaler', scaler), 
            ('pca', pca),
            ('logistic', logistic)
        ]
    )
    
    st.write('Iniciando treinamento')
    pipeline.fit(X_train, y_train)

    st.write('Treinamento concluído')
    
    st.write(f'Acurácia de {pipeline.score(X_test, y_test):.2%} na amostra de testes')

    # salvar modelo
    model_path = Path('model/model_final.pkl')
    pickle.dump(pipeline, open(model_path, 'wb'))
    st.write('Modelo salvo')

def test_model(file_test, model):

    st.write('## Scoring')

    data_raw = load_file(file_test)
    data = data_raw.copy()

    st.write('Dados enviados')
    st.markdown("---")
    st.write(data_raw)        

    st.write('Iniciando scoring')
    probability = model.predict_proba(data)
    data['probality_false'] = probability[:, 0]
    data['probality_true'] = probability[:, 1]
    data['mau_predict'] = model.predict(data)

    st.write('Scoring concluído')

    st.write('## Credit Scoring')
    st.markdown("---")
    st.write(data)        

    _xlsx = to_excel(data)
    st.write('Realize o download do scoring em excel')
    st.download_button(label='Arquivo EXCEL', data=_xlsx, file_name='scoring.xlsx')

def train_page():

    with placeholder.container():
        st.write(f"## :orange[Enviar arquivo para treinamento]")
        st.write("Envie arquivo para realizar o treinamento do modelo")
        file = st.file_uploader("Arquivo para treinamento", type = ['feather'], key='f_train')

    return file

def test_page():

    with placeholder.container():

        st.write(f"## :green[Modelo existente]")
        st.write("Atençao, já existe um modelo treinado")

        if st.button(label='Realizar novo treinamento'):
            file = train_page()
            
            model_path = Path('model/model_final.pkl')
            if model_path.exists():
                model_path.unlink()

            return file
            
        st.write("## Enviar arquivo para teste")
        file = st.file_uploader('Arquivo para teste', type = ['csv'], key='f_test')

        return file

def first_page():

    try:
         with open('model/model_final.pkl', 'rb') as model:
              model_final = pickle.load(model)
    except FileNotFoundError:
         file, action = train_page(), 'train'
    else:
         file, action = test_page(), 'test'

    if file is not None and action == 'train':
        train_model(file_train=file)
        file = None
        first_page()
    elif file is not None and action == 'test':
        test_model(file_test=file, model=model_final)
        file = None
        # first_page()

st.set_page_config(page_title = 'Credit scoring', layout="wide", initial_sidebar_state='expanded')
placeholder = st.sidebar.empty()

def main():

    st.write('# Credit scoring')
    first_page()

if __name__ == '__main__':
	main()