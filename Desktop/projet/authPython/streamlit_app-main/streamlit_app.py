#--------- import libraries
from requirements import *
from functions_download_data import get_cleaned_data, get_raw_data, get_raw_eval_df
import os
from streamlit_extras.metric_cards import style_metric_cards
import base64
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import functools
from keras.wrappers.scikit_learn import KerasRegressor
import graphs

# -------------- General Config --------

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="SHIS SLA Project : Predictive Analytics Application", page_icon=":bar_chart:", layout="wide")
# [theme]
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

#--------- Load Data ----------
#------------- functions--------------

from pandas import DataFrame
from pandas import concat
from numpy import asarray
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score, KFold


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    n_vars = 1
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
    return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
 return data[:-n_test, :], data[-n_test:, :]


 # fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]




# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        #st.write('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


# pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# models
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

def pipeline_models():
    pipelines = []
    pipelines.append(('scaledRF' , (Pipeline([('scaled' , StandardScaler()),('LR' ,RandomForestRegressor())]))))
    pipelines.append(('scaledXGB' , (Pipeline([('scaled' , StandardScaler()),('KNN' ,xgb.XGBRegressor())]))))
    pipelines.append(('scaledLGB' , (Pipeline([('scaled' , StandardScaler()),('DT' ,lgb.LGBMRegressor())]))))
    pipelines.append(('scaledRidge' , (Pipeline([('scaled' , StandardScaler()),('SVC' ,Ridge())]))))
    return pipelines


# fit an random forest model and make a one step prediction
def random_forest_forecast_bis(model, train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    #model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]




# walk-forward validation for univariate data
def walk_forward_validation_bis(model, data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast_bis(model,history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        #st.write('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions









def run_times_series_model(classifier_name,X):
     #classifier_name = st.sidebar.selectbox(
     #                   'Select classifier',('','Autoregression (AR)','Moving Average (MA)', 'Autoregressive Moving Average (ARMA)','Holt Winter’s Exponential Smoothing (HWES)'))
    if classifier_name == 'Autoregression (AR)':
            # AR example
            C = st.number_input("C (test size)", 2, 10, step=1, key="C")
            lag = st.number_input("lag (lag)", 2, 10, step=1, key="lag")
            p= st.number_input("p (time prediction)", 2, 10, step=1, key="p")
            if st.button('run model', help='Be certain to check the parameters on the sidebar'):
                X = X.squeeze()
                train, test = X[1:len(X)-C], X[len(X)-C:]
                n = len(X)
                res = AutoReg(X, lags = lag).fit()
                preds = res.model.predict(res.params, start=len(X), end=len(X)+p)
                # make predictions
                predictions = res.predict(start=len(train), end=len(train)+len(test)+p, dynamic=False)
                data = {'Test': test, 'Prediction': predictions}
    
                # Create DataFrame
                data = pd.DataFrame(data)
                st.write(data)
                st.line_chart(data)
    if classifier_name == 'Moving Average (MA)':
        st.write("Please Wait : Method under construction")
    if classifier_name == 'Autoregressive Moving Average (ARMA)':
        st.write("Please Wait : Method under construction")
    if classifier_name == 'Holt Winter’s Exponential Smoothing (HWES)':
        st.write("Please Wait : Method under construction")
    #if classifier_name == 'Autoregression (AR)':
    #    st.write("Please Wait : Method under construction")

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'LSTM method':
        units = st.slider('units', 10, 100)
        epochs = st.slider('epochs',10,150)
        batch_size = st.slider('batch_size',10,150)
        if st.button('run model', help='Be certain to check the parameters on the sidebar'):
            params['units'] = units
            params['epochs'] = epochs
            params['batch_size'] = batch_size
    return params

def get_graph(clf_name,df):
    if clf_name == 'LSTM method':
        fig, ax = plt.subplots()
        df.plot(ax=ax, xticks=df.index, rot=95,fontsize = 10)
        ax.set_xticklabels(df.index)
        plt.xlabel('Year')
        plt.ylabel('Number of values')
        plt.title('LSTM Model prediction')
        plt.legend()
    return fig



def get_classifier(clf_name, params, x_train,y_train):
    clf = None
    if clf_name == 'LSTM method':
        clf = Sequential()
        clf.add(LSTM(units =params['units'],return_sequences=True,input_shape=(x_train.shape[1],1)))
        clf.add(Dropout(0.2))
        clf.add(LSTM(units = params['units'],return_sequences=True))
        clf.add(Dropout(0.2))
        clf.add(LSTM(units = params['units'],return_sequences=True))
        clf.add(Dropout(0.2))
        clf.add(LSTM(units=params['units']))
        clf.add(Dropout(0.2))
        clf.add(Dense(units=1))
        clf.compile(optimizer = 'adam',loss='mean_squared_error')
        clf.fit(x_train,y_train, epochs=params['epochs'], batch_size =params['batch_size'],verbose=2)

        #clf = SVC(C=params['C'])
    return clf

def run_deep_time_serie_bis(classifier_name_height,val):
     
    if classifier_name_height == 'LSTM method':
        
        n = val.shape[0]
        train_set = val.values
        test_set = val.values
        sc = MinMaxScaler()
        train_set_scaled = sc.fit_transform(train_set)
        x_train = []
        y_train = []
        per = 1
        for i in range(per,n):
            x_train.append(train_set_scaled[i-per:i,0])
            y_train.append(train_set_scaled[i,0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        params = add_parameter_ui(classifier_name_height)
        clf = get_classifier(classifier_name_height, params,x_train,y_train)
        pred = clf.predict(x_train)
        pred = sc.inverse_transform(pred)
        df = pd.DataFrame(train_set, columns=['Actual'])
        pred  = np.insert(pred, 0, 0)
        #st.write(pred)
        df['Predicted']=pred 
        #df['Year']=Y
        #df = df.set_index('Year')
        #df.drop(df.index[-1], inplace=True)
        df = df.iloc[:-2 , :]
        df[df<0]=0
        st.write(df)
        st.pyplot(get_graph(classifier_name_height,df))





def create_model(optimizer='adam', dropout=0.2, activation='relu', kernel_initializer='normal'):
    model = Sequential()
    model.add(Dense(units = 15, activation = activation, input_dim = 15, kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(units = 11, activation = activation))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1, activation = activation))

    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    return model
# ----------- Data -------------------


@st.cache_data
def get_raw_data():
    """
    This function return a pandas DataFrame with the raw data.
    """

    raw_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'Immunisation_Coverage_Target.csv'))
    return raw_df


@st.cache_data
def get_cleaned_data():
    """
    This function return a pandas DataFrame with the cleaned data.
    """

    clean_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'Immunisation_Coverage_Target.csv'))
    clean_data['date']=pd.to_datetime(clean_data['periodcode'], format='%Y%m')
    clean_data['dayofweek'] = clean_data['date'].dt.dayofweek
    clean_data['quarter'] = clean_data['date'].dt.quarter
    clean_data['month'] = clean_data['date'].dt.month
    clean_data['dayofyear'] = clean_data['date'].dt.dayofyear
    clean_data['dayofmonth'] = clean_data['date'].dt.day
    clean_data = clean_data.rename(columns={'Immunisation under 1 year coverage':'y'})
    clean_data =clean_data[['organisationunitid','organisationunitname','date','dayofmonth','dayofweek','dayofyear','month','quarter','y']]
    return clean_data


@st.cache_data
def get_raw_eval_df():
    """
    This function return a pandas DataFrame with the dataframe and the machine learning models along with it's metrics.
    """

    raw_eval_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'model_evaluation_reg.csv'))
    return raw_eval_df


#@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x})
#@st.cache_data
def load_models_df(dataframe):
    df_evaluated = dataframe.copy()
    models_list = os.listdir(os.path.join(os.path.abspath(''), 'models_reg'))
    rep = {"pipe": "model", "pickle": "h5"}
    for index, row in df_evaluated.iterrows():
        # check if the file_name is in our models directory
        if row['pipe_file_name'] in models_list:
            # now, load the model.
            with open(os.path.join(os.path.abspath(''), 'models_reg', row['pipe_file_name']), 'rb') as fid:
                model_trained = pickle.load(fid)
            
            # for the keras model, we have to load the model separately and add into the pipeline or transformed target object.
            if row['name'] == 'NeuralNetwork':
                model_keras = load_model(os.path.join(os.path.abspath(''), 'models_reg', functools.reduce(lambda a, kv: a.replace(*kv), rep.items(), row['pipe_file_name'])))
                # check if the target transformer it is active
                if row['custom_target']:
                    # reconstruct the model inside a kerasregressor and add inside the transformed target object
                    model_trained.regressor.set_params(model = KerasRegressor(build_fn=create_model, verbose=0))
                    # add the keras model inside the pipeline object
                    model_trained.regressor_.named_steps['model'].model = model_keras
                else:
                    model_trained.named_steps['model'].model = model_keras

            df_evaluated.loc[index, 'model_trained'] = model_trained

    # we have to transform our score column to bring it back to a python list
    df_evaluated['all_scores_cv'] = df_evaluated['all_scores_cv'].apply(lambda x: [float(i) for i in x.strip('[]').split()])
    
    return df_evaluated.sort_values(by='rmse_cv').reset_index(drop=True)

@st.cache_data
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')




@st.cache_data
def split(dataframe):
    df = dataframe.copy()
    #df= df.set_index(['date'])
    x = df.drop(columns=['y','organisationunitid','organisationunitname'], axis=1)
    y = df['y']
    y_train, y_test= np.split(y, [int(.60 *len(y))])
    x_train, x_test= np.split(x, [int(.60 *len(x))])    
    #x = df.drop(columns=['rent amount (R$)'], axis=1)
    #y = df['rent amount (R$)']
    # check if the random state it is equal to when it was trained, this is very important.
    #x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state=0)
    date_train=x_train['date']
    date_test=x_test['date']
    date_x=x['date']
    x_train= x_train.drop(columns=['date'], axis=1)
    x_test= x_test.drop(columns=['date'], axis=1)

    return x, y, x_train, x_test, y_train, y_test,date_train, date_test,date_x


raw_df = get_raw_data()
clean_df = get_cleaned_data()

raw_eval_df = get_raw_eval_df()
eval_df = load_models_df(raw_eval_df)
x, y, x_train, x_test, y_train, y_test,date_train, date_test,date_x = split(clean_df)




#--------- Authentification page -------------
#------ Users authentification ----------
names = ["Seydou Sylla","Christopher Mvlase","Hans Naude","Benjamin Mayasi","Elmarie Claasen","Moeti Mphoso","Pooben Dass"]
usernames = ["ssylla","cmvlase","hnaude","bmayasi","eclaasen","mmphoso","pdass"]
#load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file: 
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,"Data_Science", "abcdef")
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False : 
    st.error("Username/Password is incorrect ")

if authentication_status == None : 
    st.error(" please enter you Username and Password  ")




if authentication_status == True :

    selected = option_menu(
            menu_title = None, # required
            options = ["Home", "Dashbord", "Predictive Models", "Contacts"], # required 
            icons = ["house","book", "book", "envelope"], # optional
            menu_icon = "cast",    #optional 
            default_index = 0, 
            orientation = "horizontal") 
    if selected == "Dashbord":
        data = clean_df.copy()
        select_model_orgid = st.sidebar.selectbox('Select Organisation Unit ID ',['']+[i for i in  data['organisationunitid'].unique()])
        select_model_orgmame = st.sidebar.selectbox('Select Organisation Unit  Name', ['']+[i for i in data['organisationunitname'].unique()]  )
        date_beg = st.sidebar.date_input("Select the Begininng Date:",datetime.date(2023, 3, 1))
        date_end = st.sidebar.date_input("Select the  end Date:", datetime.date(2023, 3, 1))
        if st.sidebar.button('Select Data', help='Be certain to check the parameters on the sidebar'):
            data = data.query("organisationunitid == @select_model_orgid & organisationunitname ==@select_model_orgmame & date >= @date_beg  & date <= @date_end ")
            if not data.empty:

                st.dataframe(data)
                col1, col2,col3= st.columns(3)
                col1.metric("Number of rows", data.shape[0])
                col2.metric("Mean", data['y'].mean().round(2) )
                col3.metric("Standard Deviance", data['y'].std().round(2))
                style_metric_cards(border_left_color='#6A1A41', background_color='#009AA6')
                height, width, margin = 450, 1500, 10
                st.subheader(' Distribution')
                    # Create line chart
                fig, ax = plt.subplots()
                ax.plot(data['date'], data['y'])
                    # Add labels and legend
                ax.set_xlabel('date')
                ax.set_ylabel('Value')
                ax.legend()
                    # Display chart using streamlit
                st.pyplot(fig)
                    # Create a download link for the CSV file
                    # Define the CSV filename
                CSV_FILENAME = "my_data.csv"
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{CSV_FILENAME}">Download CSV file</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.write('Data is empty Please select others paramaters to continuous ')

    if selected == "Predictive Models":
        data = clean_df.copy()
        select_model_orgid1 = st.sidebar.selectbox('Select Organisation Unit ID ',['']+[i for i in  data['organisationunitid'].unique()])
        select_model_orgmame1 = st.sidebar.selectbox('Select Organisation Unit  Name', ['']+[i for i in data['organisationunitname'].unique()]  )

        date_beg1 = st.sidebar.date_input("Select the Begininng Date:",datetime.date(2023, 3, 1))
        date_end1 = st.sidebar.date_input("Select the  end Date:", datetime.date(2023, 3, 1))
        if st.sidebar.checkbox('Select Data', help='Be certain to check the parameters on the sidebar'):
            data = data.query("organisationunitid == @select_model_orgid1 & organisationunitname ==@select_model_orgmame1 & date >= @date_beg1  & date <= @date_end1 ")

            if not data.empty:
                jobs_selected = st.radio('Select what your job run',('','Compiled Models','Fablab Machine Learning  Models'))
                if jobs_selected == 'Compiled Models':

                    with st.expander("See explanation"):
                        st.markdown("In this Section, you have use many compiled methods" )
                        st.markdown("- You have possibilty to select one method and  you see the graph on the results" )
                        st.markdown("Models are compiled in the background on a specific data set. If datasets are changed, predictions may not be effective." )
                        st.markdown("**You can use the lab to choose the appropriate model for the chosen dataset. **" )


                    with st.expander("Available Models"):
                        st.subheader('Available Models')
                        st.dataframe(eval_df.drop(columns=['all_scores_cv', 'pipe_file_name', 'model_trained']).sort_values(by=['rmse_testset']))
                        select_model_meval = st.selectbox('Select the Model',[i for i in eval_df.sort_values(by=['rmse_testset'])['name'].unique()]  )
                        #select_custom_features_meval = st.select_slider('Create Custom Features?',[False, True])
                        select_custom_features_meval= False
                        select_custom_target_meval = st.select_slider('Perform Target Transformation?',[False, True])
                        model_trained_meval = eval_df.loc[(eval_df['name'] == select_model_meval) & (eval_df['custom_features'] == select_custom_features_meval) & (eval_df['custom_target'] == select_custom_target_meval)]['model_trained'].iloc[0]
                        # -------------- figs -----------------

                        height, width, margin = 450, 1500, 30

                        st.subheader('Distribution of the Target Variable')
                        data_pred = {'Test': y_test, 'Prediction': model_trained_meval.predict(x_test)}
        
                        # Create DataFrame
                        data_pred = pd.DataFrame(data_pred)
                        #st.write(data_pred)
                        st.line_chart(data_pred)


                        fig = graphs.plot_distplot(
                            y_real=y_test, 
                            y_predict=model_trained_meval.predict(x_test),
                            height=height, 
                            width=width, 
                            margin=margin,
                            title_text='Predicted and Real Value'
                        )
                        st.plotly_chart(fig)

                        st.subheader('Distribution of the Residuals')

                        # predict the values of the entire data
                        prediction = model_trained_meval.predict(x)
                        # calculate the residual
                        resid = prediction - y

                        # create a copy to not alter the original data
                        df_plot = clean_df.copy()
                            # create a column to identify the data regarding to train or test
                        df_plot['split'] = 'train'
                        df_plot.loc[x_test.index, 'split'] = 'test'
                        df_plot['prediction'] = prediction
                        df_plot['resid'] = resid

                        # plot the residual plot with the histograms
                        fig = graphs.plot_scatter(data=df_plot, x='prediction', y='resid', residual=True, height=height, width=width, margin=margin, title_text='Residuals per Split')
                            
                        st.plotly_chart(fig)
                        st.subheader('Boxplot of RMSE in Cross Validation')

                        fig = graphs.plot_boxplot(data=eval_df, x=None, y=None, model_name=select_model_meval, custom_feature=select_custom_features_meval, custom_target=select_custom_target_meval, single_box=True, title_text='Cross Validation with 5 Folds', height=height, width=width, margin=margin)

                        st.plotly_chart(fig)
                        data_pred['date']=date_test
                        #st.write(len(date_test),len(y_test),x_test.shape,x.shape)
                        #st.dataframe(data_pred)
                        #st.dataframe(x_test)
                        st.download_button(
                            label="Download data as CSV",
                            data=convert_df_to_csv(data_pred),
                            file_name='prediction_data.csv',
                            mime='text/csv',
                            )

                if jobs_selected == 'Fablab Machine Learning  Models':
                    
                    #jobs_time= st.radio('Select your Time Series Methods',('pipeline','Classical Times Series Methods','Deep Times Series Methods'))
                    #if jobs_time=="pipeline" :
                    with st.expander("See explanation"):
                        st.markdown("In this Section, you have use three items" )
                        st.markdown("1. You have possibilty to run many machine learning models at time and  you see the graph on the best model" )
                        st.markdown("2.You have possibility to run classical model time series (AR, MA, ARMA,...) " )
                        st.markdown("3.You have possibility to run a deep learning model (LSTM, Prophet) " )



                    with st.expander("Pipeline Machine Learning methods"):
                        st.write(data)
                        data2 = series_to_supervised(data['y'], n_in=10)
                        #st.write(data2)
                        # evaluate
                        #mae, y_exp, yhat = walk_forward_validation(data2, 10)
                        #st.write('MAE: %.3f' % mae)
                        #d = {'Expected': y_exp, 'Predicted': yhat}
                        #chart_data = pd.DataFrame(data=d)
                        #st.write(chart_data)
                        #st.line_chart(chart_data)

                        train, test = train_test_split(data2, 12)
                        x_train, y_train = train[:, :-1], train[:, -1]
                        model_name = []
                        results = []
                        results_mean = []
                        models =[]
                        pipelines = pipeline_models()
                        for pipe ,model in pipelines:
                            kfold = KFold(n_splits=10)
                            crossv_results = cross_val_score(model , x_train ,y_train ,cv =kfold , scoring='neg_mean_squared_error')
                            results.append(crossv_results)
                            model_name.append(pipe)
                            models.append(model)
                            results_mean.append(crossv_results.mean())
                            msg = "%s: %f (%f)" % (model_name, crossv_results.mean(), crossv_results.std())
                            #st.write(msg)
                        # Compare different Algorithms
                        fig = plt.figure()
                        fig.suptitle('Algorithm Comparison')
                        ax = fig.add_subplot(111)
                        plt.boxplot(results)
                        ax.set_xticklabels(model_name)
                        st.pyplot(fig)
                        #st.write(results_mean)
                        minindex = np.argmin(results_mean)
                        st.write(models[minindex])
                        # evaluate
                        mae, y_exp, yhat = walk_forward_validation_bis(models[minindex],data2, 10)
                        st.write('MAE: %.3f' % mae)
                        d = {'Expected': y_exp, 'Predicted': yhat}
                        chart_data = pd.DataFrame(data=d)
                        #st.write(chart_data)
                        st.line_chart(chart_data)
                        #with st.expander("See explanation"):
                        #maxindex = np.argmax(results_mean)
                        #st.write(models[maxindex])
                            # evaluate
                        #mae, y_exp, yhat = walk_forward_validation_bis(models[maxindex],data2, 10)
                        #st.write('MAE: %.3f' % mae)
                        #d = {'Expected': y_exp, 'Predicted': yhat}
                        #chart_data = pd.DataFrame(data=d)
                        #st.write(chart_data)
                        #st.line_chart(chart_data)


    

                    #if jobs_time=="Classical Times Series Methods" :
                    with st.expander("Classical Times Series Methods"):
                        st.write(data)
                        classifier_name = st.selectbox(
                            'Select classifier',('','Autoregression (AR)','Moving Average (MA)', 'Autoregressive Moving Average (ARMA)','Holt Winter’s Exponential Smoothing (HWES)'))
                        X = data['y']  
                        Y= data["date"]         
                        run_times_series_model(classifier_name,X)


                    with st.expander("Deep Times Series Methods"):
                    #if jobs_time== "Deep Times Series Methods":
                        #st.write(type(Y))
                        X = data['y']  
                        Y= data["date"] 
                        X = X.to_frame()
                        Y = Y.to_frame()
                        classifier_name_height = st.selectbox(
                            'Select other Time Series method',('','LSTM method','Prophet method'))
                        run_deep_time_serie_bis(classifier_name_height,X)
                    

            else:
                st.write('Data is empty Please select others paramaters to continuous ')




        
        


     
         

    authenticator.logout("Logout","sidebar")
    #st.sidebar.title(f"Welcome {name}")