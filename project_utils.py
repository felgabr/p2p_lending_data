#########################################################################
################                IMPORTS                  ################
#########################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##########################################################################
################                FUNCTIONS                 ################
##########################################################################

def convert_column_type(dataframe: 'pd.Dataframe', dtype: 'np.dtype', columns: list, excl_list: list = None) -> 'pd.Dataframe':
    '''
    Convert Pandas DataFrame column(s) type.
    
    Args:
        dataframe: Pandas DataFrame containing the specified columns.
        dtype: Numpy new type.
        columns: A list of column to be converted.
        excl_list: Optional exception list.
    
    Returns:
        Pandas DataFrame: Containing casted columns.
    '''
    if excl_list:
        for column in columns:
            if column not in excl_list:
                dataframe[column] = dataframe[column].astype(dtype)
    else:
        for column in columns:
            dataframe[column] = dataframe[column].astype(dtype)
    return dataframe


def plot_learning_curve(estimator,
                        dataframe,
                        target,
                        train_sizes = np.linspace(0.05, 1.0, 10),
                        cv = 7,
                        scoring = None,
                        n_jobs = -1,
                        save_dir: str = None):
    '''
    Função para plotar a curva de aprendizagem de um modelo supervisionado, classificação binária ou regressão.
    
    Learning Curves mostram a performance de treino e validação de um estimador ao variarmos a quantidadede amostras de treino utilizadas.
    É geralmente construida a partir de hyperparâmetros ótimos fornecidos por uma validation curve ou definidos a partir de uma busca de
    hyperparâmetros.
    
    Dado que a alteração no número de observações utilizadas para treinamento impacta o erro cometido pelo modelo, as Learning Curves
    fornecem informações
    sobre o nível de viés e variância de um modelo com hyperparâmetros fixados.
    
    A função varia a porcentagem de observações utilizadas para treino no range definido, plotando os resultados de treino e teste para cada
    valor, junto da incerteza associada e performance do teste.
    
    O comportamento da distância entre as duas curvas, treino e teste, fornece informações sobre o comportamento do erro de variância do
    modelo.
    
    - Caso as duas curvas permaneçam muito próximas por todas porcentagens de amostras utilizadas para treino, isso significa que esse
    modelo tem baixíssimo erro de variância, mesmo com menos observações ou menos regularização.
    
    - Caso as duas curvas estejam muito separadas porém tenham convergido para um mesmo ponto, isso significa que o modelo atual apresenta
    baixíssimo erro de variância porém esse erro cresce sensivelmente ao diminuirmos as observações utilizadas para treino ou diminuindo a
    regularização.
    
    - Caso as duas curvas estejam muito separadas e permanecem muito separadas, sem convergir para um mesmo ponto, isso siginifica que o
    modelo apresenta alto erro de variância e o quanto de performance podeser ganho reduzindo esse erro, aproximando as duas curvas para seu
    ponto de convergência ao aumentar o número de observações, regularizar ou aplicar outras técnicas.
    
    O ponto de convergência observado entre as curvas fornece uma noção do viés do modelo, ao indicar qual seria a performance do modelo ao
    removermos seu erro de variância.
    
    Author:
        Gabriel Sant'Anna
        
    Args:
        estimator: Estimador instanciado a ser utilizado para modelar o problema de classificação ou regressão ().
        dataframe: Dataframe que contém dados (Pandas DataFrame).
        target: Coluna que tem o target (Pandas Series).
        train_sizes: Array com a sequência das porcentagens de amostras utilizadas para treinamento (Numpy Array).
        cv: Quantidade de folds utilizados para cross-validation (int).
        scoring: Métrica de avaliação da performance, dadas pelos nomes das métricas documentadas no sklearn (string).
        n_jobs: Quantidade de processadores utilizados no processo (int).
        save_dir: Optional, path to save the chart file.
    '''
    from sklearn.model_selection import learning_curve
    
    # Defines scoring metric
    scoring_used = 'roc_auc' if (type(scoring) == type(None) and target.nunique() == 2) else (
                   'neg_root_mean_squared_error' if (type(scoring) == type(None) and target.nunique() > 2) else
                   scoring)
    
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
                                                            dataframe, 
                                                            target,
                                                            train_sizes = train_sizes,   
                                                            cv = cv,
                                                            scoring = scoring_used,
                                                            n_jobs = n_jobs)
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color='#111111',  label='Training score')
    plt.plot(train_sizes, test_mean, color='#111111', label='Cross-validation score')

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='#DDDDDD')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='#DDDDDD')
    
    # Create plot
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring_used)
    plt.legend(loc='best')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    plt.show()


def plot_validation_curve(estimator,
                          dataframe,
                          target,
                          param_name,
                          param_range,
                          cv = 7,
                          scoring = None,
                          n_jobs = -1,
                          save_dir: str = None):
    '''
    Função para plotar a curva de validação de um modelo supervisionado, classificação binária ou regressão.
    
    Validation Curves permitem analisar a influência de um único hyperparâmetro - geralmente o principal e mais significativo hyperparâmetro
    do modelo no trade-off entre viés/variância.
    Fornece uma noção baseline da performance do modelo sobre diferentes valores de um hyperparâmetro.
    
    Exemplos de hyperparâmetros mais usados:
        - Profundidade máxima em Árvores de decisão;
        - Número de vizinhos em KNN;
        - Peso da regularização Lasso ou Ridge em Modelos Lineares;
        - Peso do Hyperparâmetro C para regularização em SVMs;
        
    A função varia o hyperparâmetro indicado no range definido, plotando resultados de treino/teste para cada valor, junto da incerteza
    associada a performance do teste.
    
    A altura das curvas em cada ponto do range indicam sobre o erro de viés cometido pelo estimador.
    
    A distância entre a curva de treino e a curva de teste indicam sobre o erro sobre a variância cometido pelo estimador.
    
    Author:
        Gabriel Sant'Anna
        
    Args:
        estimator: Estimador instanciado a ser utilizado para modelar o problema de classificação ou regressão ().
        dataframe: Dataframe que contém dados (Pandas DataFrame).
        target: Coluna que tem o target (Pandas Series).
        param_name: Nome do hyperparâmetro avaliado na validation curve (strig).
        param_range: Intervalo de valores de avaliação do hyperparâmetro escolhido (list).
        cv: Quantidade de folds utilizados para cross-validation (int).
        scoring: Métrica de avaliação da performance, dadas pelos nomes das métricas documentadas no sklearn (string).
        n_jobs: Quantidade de processadores utilizados no processo (int).
        save_dir: Optional, path to save the chart file.
    '''
    from sklearn.model_selection import validation_curve
    
    # Defines scoring metric
    scoring_used = 'roc_auc' if (type(scoring) == type(None) and target.nunique() == 2) else (
                   'neg_root_mean_squared_error' if (type(scoring) == type(None) and target.nunique() > 2) else
                   scoring)
    
    # Create CV training and test scores for various params values
    train_scores, test_scores = validation_curve(estimator, 
                                                 dataframe, 
                                                 target,
                                                 param_name = param_name,
                                                 param_range = param_range,
                                                 cv = cv,
                                                 scoring = scoring_used,
                                                 n_jobs = n_jobs)
    
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label='Training score', color='black')
    plt.plot(param_range, test_mean, label='Cross-validation score', color='dimgrey')

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='gray')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color='gainsboro')

    # Create plot
    plt.title('Validation Curve')
    plt.xlabel(param_name)
    plt.ylabel(scoring_used)
    plt.tight_layout()
    plt.legend(loc='best')
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    plt.show()


def get_low_variance_columns(dframe=None,
                             columns=None,
                             skip_columns=[],
                             thresh=0.0,
                             autoremove=False):
    '''
    Wrapper for sklearn VarianceThreshold for use on pandas dataframes.
    
    Author:
        Jason Wolosonovich
    '''
    from sklearn.feature_selection import VarianceThreshold
    
    print('Finding low-variance features.')
    try:
        # get list of all the original df columns
        all_columns = dframe.columns

        # remove `skip_columns`
        remaining_columns = all_columns.drop(skip_columns)

        # get length of new index
        max_index = len(remaining_columns) - 1

        # get indices for `skip_columns`
        skipped_idx = [all_columns.get_loc(column)
                       for column
                       in skip_columns]

        # adjust insert location by the number of columns removed
        # (for non-zero insertion locations) to keep relative
        # locations intact
        for idx, item in enumerate(skipped_idx):
            if item > max_index:
                diff = item - max_index
                skipped_idx[idx] -= diff
            if item == max_index:
                diff = item - len(skip_columns)
                skipped_idx[idx] -= diff
            if idx == 0:
                skipped_idx[idx] = item

        # get values of `skip_columns`
        skipped_values = dframe.iloc[:, skipped_idx].values

        # get dataframe values
        X = dframe.loc[:, remaining_columns].values

        # instantiate VarianceThreshold object
        vt = VarianceThreshold(threshold=thresh)

        # fit vt to data
        vt.fit(X)

        # get the indices of the features that are being kept
        feature_indices = vt.get_support(indices=True)

        # remove low-variance columns from index
        feature_names = [remaining_columns[idx]
                         for idx, _
                         in enumerate(remaining_columns)
                         if idx
                         in feature_indices]

        # get the columns to be removed
        removed_features = list(np.setdiff1d(remaining_columns,
                                             feature_names))
        print('Found {0} low-variance columns.'
              .format(len(removed_features)))

        # remove the columns
        if autoremove:
            print('Removing low-variance features.')
            # remove the low-variance columns
            X_removed = vt.transform(X)

            print('Reassembling the dataframe (with low-variance '
                  'features removed).')
            # re-assemble the dataframe
            dframe = pd.DataFrame(data=X_removed,
                                  columns=feature_names)

            # add back the `skip_columns`
            for idx, index in enumerate(skipped_idx):
                dframe.insert(loc=index,
                              column=skip_columns[idx],
                              value=skipped_values[:, idx])
            print('Succesfully removed low-variance columns.')

        # do not remove columns
        else:
            print('No changes have been made to the dataframe.')

    except Exception as e:
        print(e)
        print('Could not remove low-variance features. Something '
              'went wrong.')
        pass

    return dframe, removed_features


def sklearn_vif(exogs, data):
    '''
    Assume we have a list of exogenous variable [X1, X2, X3, X4].
    To calculate the VIF and Tolerance for each variable, we regress
    each of them against other exogenous variables. For instance, the
    regression model for X3 is defined as:
                        X3 ~ X1 + X2 + X4
    And then we extract the R-squared from the model to calculate:
                    VIF = 1 / (1 - R-squared)
                    Tolerance = 1 - R-squared
    The cutoff to detect multicollinearity:
                    VIF > 10 or Tolerance < 0.1

    Author:
        steven

    Args:
        exogs (list): List of exogenous/independent variables.
        data (Pandas DataFrame): Df storing all variables.

    Returns:
        VIF and Tolerance DataFrame for each exogenous variable.
    
    '''    
    from sklearn.linear_model import LinearRegression
    
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif


def train_test_split_sorted(X, y, test_size, dates):
    '''
    Splits X and y into train and test sets, with test set separated by most recent dates.
    
    Author:
        Glyph

    Example:
    --------
    >>> from sklearn import datasets

    # Fake dataset:
    >>> gen_data = datasets.make_classification(n_samples=10000, n_features=5)
    >>> dates = np.array(pd.date_range('2016-01-01', periods=10000, freq='5min'))
    >>> np.random.shuffle(dates)
    >>> df = pd.DataFrame(gen_data[0])
    >>> df['date'] = dates
    >>> df['target'] = gen_data[1]

    # Separate:
    >>> X_train, X_test, y_train, y_test = train_test_split_sorted(df.drop('target', axis=1), df['target'], 0.33, df['date'])

    >>> print('Length train set: {}'.format(len(y_train)))
    Length train set: 8000
    >>> print('Length test set: {}'.format(len(y_test)))
    Length test set: 2000
    >>> print('Last date in train set: {}'.format(X_train['date'].max()))
    Last date in train set: 2016-01-28 18:35:00
    >>> print('First date in test set: {}'.format(X_test['date'].min()))
    First date in test set: 2016-01-28 18:40:00
    '''
    from math import ceil

    n_test = ceil(test_size * len(X))

    sorted_index = [x for _, x in sorted(zip(np.array(dates), np.arange(0, len(dates))), key=lambda pair: pair[0])]
    train_idx = sorted_index[:-n_test]
    test_idx = sorted_index[-n_test:]

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def classification_report_df(y_true: 'array', y_pred: 'array', target_names: list = None) -> 'pd.Dataframe':
    '''
    Build a Dataframe report showing the main binary classification metrics.
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        target_names: Optional, display names matching the labels (same order).

    Returns:
        Pandas DataFrame: Containing calculated metrics.
    '''
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true,
                                   y_pred,
                                   target_names=target_names,
                                   output_dict=True)
    
    return pd.DataFrame(report).transpose().apply(lambda x: x.apply('{0:.2f}'.format))


def get_performance(score: 'pd.Series', target: 'pd.Series') -> 'pd.Dataframe + str':
    '''
    "It stands for Kolmogorov–Smirnov which is named after Andrey Kolmogorov and Nikolai Smirnov.
    It compares the two cumulative distributions and returns the maximum difference between them.
    It is a non-parametric test which means you don't need to test any assumption related to the distribution of data.
    In KS Test, Null hypothesis states null both cumulative distributions are similar.
    Rejecting the null hypothesis means cumulative distributions are different.
    In data science, it compares the cumulative distribution of events and non-events and KS is where there is a maximum
    difference between the two distributions. In simple words, it helps us to understand how well our predictive model is
    able to discriminate between events and non-events."
    
    "Lift is a measure of the effectiveness of a predictive model calculated as the ratio between the results obtained with
    and without the predictive model.
    
    Example:
        Now the lift for the decile 1 is 5.84. This means that a campaign based on decile 1 selection
        can be 5.84 times more successful than a campaign based on random selection."
    
    Args:
        score: Pandas Series of given score.
        target: Pandas Series of dependent variable.        
    
    Author:
        Felipe Gabriel - Adapted from Deepanshu Bhalla
    '''
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_columns', None)
    
    # Create a auxiliary DataFrame with score and true label
    df = pd.concat([score.reset_index(drop=True).rename('score'),
                    target.reset_index(drop=True).rename('target')], axis=1)
    
    df['bucket'] = pd.qcut(df.score, 10) # Discretize into deciles
    df['target0'] = 1 - df.target # Flag nonevents
    grouped = df.groupby('bucket', as_index=False) # Group by buckets
    
    df_eval = pd.DataFrame() # Create empty DataFrame    
    
    df_eval['min_score'] = grouped.min().score # Get min score in each bucket
    df_eval['max_score'] = grouped.max().score # Get max score in each bucket
    
    df_eval['event'] = grouped.sum().target # Get number of events in each bucket
    df_eval['nonevent'] = grouped.sum().target0 # Get number of nonevents in each bucket
    
    df_eval['dec_pop'] = df_eval.event + df_eval.nonevent # Total population in the decile
    
    df_eval = df_eval.sort_values(by='min_score', ascending=False)\
              .reset_index(drop=True) # Sort from first to last decile
    
    df_eval['dec_eventrate'] = df_eval.event / df_eval.dec_pop # % of event in each decile
    avg_event_rate = df_eval.event.sum() / df_eval.dec_pop.sum() # % of event in the whole population
        
    df_eval['event_rate'] = (df_eval.event / df.target.sum()) # Calculate event rate
    df_eval['nonevent_rate'] = (df_eval.nonevent / df.target0.sum()) # Calculate nonevent rate
    
    df_eval['cum_eventrate'] = round((df_eval.event / df.target.sum()).cumsum(), 4) # Calc Gain
    df_eval['cum_noneventrate'] = round((df_eval.nonevent / df.target0.sum()).cumsum(), 4) # Calc cumulative nonevent rate
    
    df_eval['lift'] = round(df_eval.dec_eventrate / avg_event_rate, 2) # lift over avg
    
    df_eval['ks'] = np.round(df_eval.cum_eventrate - df_eval.cum_noneventrate, 3) * 100 # Calc KS
    
    # Add Total row at the end
    df_eval.loc[len(df_eval.index)] = [df_eval.min_score.min(), df_eval.max_score.max(), df_eval.event.sum(),\
                                   df_eval.nonevent.sum(), df_eval.dec_pop.sum(), avg_event_rate, 1, 1, 1, 1,\
                                   1, df_eval.ks.max()]
    
    # Decile column
    df_eval.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'total']
    df_eval.index.rename('decile', inplace=True)
    
    # Formating
    to_int = ['min_score', 'max_score', 'event', 'nonevent', 'dec_pop']
    to_perc = ['dec_eventrate', 'event_rate', 'nonevent_rate', 'cum_eventrate', 'cum_noneventrate']
    
    df_eval[to_int]= df_eval[to_int].astype('Int64')
    df_eval[to_perc]= df_eval[to_perc].apply(lambda x: x.apply('{0:.2%}'.format))
    
    # Highlight KS
    ks = "KS is " + str(round(max(df_eval['ks']), 2)) + "%" + " at decile " +\
         str((df_eval.index[df_eval['ks'] == max(df_eval['ks'])][0]))
    
    return df_eval, ks


def shap_feature_importance(df: 'pd.DataFrame', shap_values: list) -> 'pd.Dataframe':
    '''
    Retrieves SHAP feature importance for binary classification, calculates the mean for the absolute values.
    Args:
        data: pandas DataFrame used to calculate the shape values.
        shap_values: A list of shap value arrays with one array for each class.
    '''
    import numpy as np
    import pandas as pd
    
    # Create DataFrame
    feat_import = pd.DataFrame(list(zip(df.columns, np.abs(shap_values)[1].mean(0))),
                                      columns=['feature_name', 'feature_importance_value'])
    
    # Sort feature importance descending
    feat_import = feat_import.iloc[(-np.abs(feat_import['feature_importance_value'].values)).argsort()]\
                  .reset_index(drop=True)
    
    # Get ranking
    feat_import.index += 1
    
    return feat_import


def plot_ks(df_eval: 'pd.DataFrame', save_dir: str = None):
    '''
    Plots Kolmogorov-Smirnov Chart from 2 cumulative distributions.
    
    It will help understand how good our model is in differentiating the customers, the KS statistic
    is the highest difference between the cumulative target and the cumulative nontarget.
    Higher KS value means that the model is good at separating the two classes.
    
    Args:
       df_eval: Pandas DataFrame generated from get_performance function.
       save_dir: Optional, path to save the chart file.
    '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Points of KS line
    x = [df_eval.index[df_eval['ks'] == max(df_eval['ks'])][0],
     df_eval.index[df_eval['ks'] == max(df_eval['ks'])][0]]     
    y = [df_eval.cum_eventrate[df_eval['ks'] == max(df_eval['ks'])].apply(lambda x: float(x.strip('%'))).iloc[0],
         df_eval.cum_noneventrate[df_eval['ks'] == max(df_eval['ks'])].apply(lambda x: float(x.strip('%'))).iloc[0]]
    
    # Plot lines of cumulative event rate, cumulative nonevent rate and KS in this order
    ax.plot(list(df_eval.index)[:-1],
            df_eval['cum_eventrate'].iloc[:-1].apply(lambda x: float(x.strip('%'))),
            color='darkslateblue',
            label='cum_eventrate')
    ax.plot(list(df_eval.index)[:-1],
            df_eval['cum_noneventrate'].iloc[:-1].apply(lambda x: float(x.strip('%'))),
            color='lightseagreen',
           label='cum_noneventrate')
    ax.plot(x, y,'r--', color='yellowgreen', label='KS = '+str(round(max(df_eval['ks']), 2)))

    # Axis Formatting
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    # Title and legend
    ax.set_title('Kolmogorov-Smirnov Chart')
    ax.set_xlabel('Decile')

    plt.legend(loc = 'lower right')
    plt.show()
    if save_dir is not None:
        fig.savefig(save_dir, bbox_inches='tight', dpi=600)
        
        
def plot_roc_curve(y_true: 'pd.Series', y_score: 'pd.Series', save_dir: str = None):
    '''
    Plots the ROC curve and show AUC score for binary classification.
    
    Args:
        y_true: True binary labels.
        y_score: Target scores, can either be probability estimates of the positive class, confidence values.
        or non-thresholded measure of decisions.
        save_dir: Optional, path to save the chart file.
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Compute ROC_AUC score
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_score[:,1])
    
    # Calculate true/false positive rate and thresholds
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score[:,1], pos_label=1)

    # Plot roc curve
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc, color='darkslateblue')
    plt.plot([0, 1], [0, 1], '--', label='Random', color='gray')#color='lightseagreen'

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.title('Receiver Operating Characteristic')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc = 'lower right')
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    plt.show()
    
    
def plot_calibration_curve(y_true: 'np.array', y_pred: 'np.array', caly_pred: 'np.array' = None, save_dir: str = None):
    '''
    "Calibration curves (also known as reliability diagrams) compare how well the
    probabilistic predictions of a binary classifier are calibrated. It plots the
    true frequency of the positive label against its predicted probability, for
    binned predictions."
    
    Plots calibration Curve.
    
    Args:
        y_true: True binary labels.
        y_pred: predictions from uncalibrated model.
        caly_pred: Optional, predictions from calibrated model.
        save_dir: Optional, path to save the chart file.
    '''
    from sklearn.calibration import calibration_curve
    
    un_prob_true, un_prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    # Plot ideal curve 
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    
    # Plot Uncalibrated curve
    plt.plot(un_prob_true, un_prob_pred, marker='.', color='darkslateblue', label='Uncalibrated')
    
    if caly_pred is not None:
        cal_prob_true, cal_prob_pred = calibration_curve(y_true, caly_pred, n_bins=10)
        
        plt.plot(cal_prob_true, cal_prob_pred, marker='.', color='lightseagreen', label='Calibrated')
    
    # Title and legend
    plt.title('Calibration Curve')
    plt.ylabel('Fraction of Positives')
    plt.xlabel('Mean Predicted Probability')

    plt.legend(loc = 'lower right')
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight', dpi=600)
    plt.show()