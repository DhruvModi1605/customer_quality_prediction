'''function returns a cross-table of feature w.r.t the target variable'''
def get_cross_table(data, target_var, feature, variable_type = 'nominal'):
    cr_tab_temp = pd.crosstab(data[target_var], data[feature], margins = True)
    for i in range(0, len(set(data[target_var]))):
        cr_tab_temp.iloc[i] = 100*cr_tab_temp.iloc[i]/cr_tab_temp.loc['All']
    cr_tab_temp = pd.DataFrame(cr_tab_temp).applymap("{:.2f}".format)
    rslt_df = cr_tab_temp
    if variable_type == 'nominal':
        rslt_df = cr_tab_temp.drop(columns = ['All']).sort_values(by = 0, axis = 1)
        rslt_df['All'] = cr_tab_temp['All']
    return rslt_df


'''Function to plot the box-plot'''
def plot_boxplot(df, column, by):    
    unique_cat = np.sort(df[by].unique())
    cat_data = [df[df[by] == i][column] for i in unique_cat]
    plt.figure(figsize=[4,5])
    plt.boxplot(cat_data,labels = unique_cat)
    plt.grid(axis='y', alpha = 0.75)
    plt.xticks(rotation = 'vertical', fontsize=10)
    plt.yticks(fontsize=15)
    title = 'Boxplot of ' + column + ' by '+ by  
    plt.title(title,fontsize=15)
    plt.show()


'''function to remove one of the highly correlated (threshold) columns'''
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold)and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset

    
'''function to perform the chi2 test'''
def chi2_test(data, target_var, thold_for_pvalue = 0.05):
    df_target = data[target_var]
    df_target = pd.DataFrame(df_target)
    
    data = data.loc[:,data.columns != target_var]

    feature_list = list(data.columns.values)
    features_without_chi2 = []
    
    pval_table = []
    for var in feature_list:
        crosstab = pd.crosstab(df_target[target_var], data[var], margins = False)
        crosstab = pd.DataFrame(crosstab)

        if crosstab.min().min() <= 5:
            features_without_chi2.append(var)
            feature_list = list(set(feature_list) - set(list(var.split(" "))))
        
    for var in feature_list:
        obs = np.array(pd.crosstab(df_target[target_var], (data[var]), margins = False))
        chi2, p, dof, expected = stats.chi2_contingency(obs)
        pval_table.append(p)
                   
    pval_table = pd.DataFrame(pval_table)
    feature_list = pd.DataFrame(feature_list)
    pval_table = pd.concat([feature_list, pval_table], axis = 1)
    pval_table.columns = ['features', 'pvalue']
    pval_table.index = pval_table.features
    pval_table['pvalue'] = pval_table['pvalue'].apply(pd.to_numeric)
    pval_table = pval_table.sort_values(by = 'pvalue')
    
    pval_table_shortlisted = pval_table.loc[pval_table['pvalue'] <= thold_for_pvalue]
    pval_table_shortlisted.reset_index(inplace=True, drop=True)
    pval_cols = pval_table_shortlisted['features'].tolist()
    
    return pval_cols, pval_table, features_without_chi2


'''Function which performs features selection test: Krushal Wallis on input Categorical and numeric output data'''
def krushkal_wallis_test(data, features, target_var, alpha):
    cols_to_keep = []
    cols_to_drop = []
    for feature in features:
        list_of_samples = []
        categories = list(data[feature].unique())
        for category in categories:
            if category != 'success':
                data_category = data[data[feature] == category][target_var].reset_index(drop = True)
                list_of_samples.append(data_category)
        stat, p = kruskal(*list_of_samples)
        if p > alpha: 
            cols_to_drop.append(feature)
        else: #one of the categories has different distribution
            cols_to_keep.append(feature)
    return cols_to_keep, cols_to_drop



''' Function to iterate over classification models and obtain results'''
def train_classification_model(data, target_var, look_for, model_dict):

    #defining X_train and y_train
    X_train = data.drop(columns = target_var)
    y_train = data[target_var]
    # set empty dicts and list
    result_pr = {}
    result_roc = {}
    models = []
    predictions_df = data[[target_var]]
    
    print("Using average_precision (PR-AUC) for Scoring because the response rate is just 11% (Imbalanced), hence we are insterested in knowing how well our model can identify response (class: 1)")

    #fitting models and getting cross validation result and best params
    for index, model in enumerate(look_for):
        start = time.time()
        print()
        print('+++++++ Start New Model ++++++++++++++++++++++')
        print('Estimator is {}'.format(model_dict[index]))
        model.fit(X_train, y_train)
        print('---------------------------------------------')
        print('best params {}'.format(model.best_params_))
        print('best score (average_precision) is {0:.4f}'.format(model.best_score_))

        #getting cross-validation result
        report = model.cv_results_
        cv_result = []
        for i in range(0,n_splits):
            key = 'split' + str(i) + '_test_score'
            cv_result.append(report[key][model.best_index_])

        #statistics of best param cross validation result
        min_score = round(np.min(cv_result), 4)
        max_score = round(np.max(cv_result), 4)
        mean_score = round(np.mean(cv_result), 4)
        std_score = round(np.std(cv_result), 4)
        print("Best param CV test results (average_precision): mean: {} | std : {} | min : {} | max : {}".format(mean_score,std_score,min_score,max_score))

        #Model performance on complete train set
        y_train_pred = model.predict_proba(X_train)[:,1]
        predictions_df[model_dict[index]] = y_train_pred 
        roc_auc = round(roc_auc_score(y_train, y_train_pred), 4)
        average_precision = round(average_precision_score(y_train, y_train_pred), 4)
        print('---------------------------------------------')
        print('For complete train set:')
        print('average_precision is {} and ROC_AUC is {}'.format(average_precision, roc_auc))
        end = time.time()
        print('It lasted for {} sec'.format(round(end - start, 3)))
        print('++++++++ End Model +++++++++++++++++++++++++++\n\n')
        models.append(model.best_estimator_)
        result_pr[index] = average_precision
        result_roc[index] = roc_auc
        
    return models, predictions_df, result_pr, result_roc



''' Function to iterate over Regression models and obtain results'''
def train_regression_model(data, target_var, look_for, model_dict):

    #defining X_train and y_train
    X_train = data.drop(columns = target_var)
    y_train = data[target_var]
    # set empty dicts and list
    result_r2 = {}
    result_adj_r2 = {}
    result_mae = {}
    models = []
    predictions_df = data[[target_var]]
    delta_df = pd.DataFrame()
    
    print("Using R2 for scoring")

    #fitting models and getting cross validation result and best params
    for index, model in enumerate(look_for):
        start = time.time()
        print()
        print('+++++++ Start New Model ++++++++++++++++++++++')
        print('Estimator is {}'.format(model_dict[index]))
        model.fit(X_train, y_train)
        print('---------------------------------------------')
        print('best params {}'.format(model.best_params_))
        print('best score (r2) is {0:.4f}'.format(model.best_score_))

        #getting cross-validation result
        report = model.cv_results_
        cv_result = []
        for i in range(0,n_splits):
            key = 'split' + str(i) + '_test_score'
            cv_result.append(report[key][model.best_index_])

        #statistics of best param cross validation result
        min_score = round(np.min(cv_result), 4)
        max_score = round(np.max(cv_result), 4)
        mean_score = round(np.mean(cv_result), 4)
        std_score = round(np.std(cv_result), 4)
        print("Best param CV test results (r2): mean: {} | std : {} | min : {} | max : {}".format(mean_score,std_score,min_score,max_score))

        #Model performance on complete train set
        y_train_pred = model.predict(X_train)
        predictions_df[model_dict[index]] = y_train_pred 
        delta_df[model_dict[index]] = y_train_pred - y_train
        mae = round(mean_absolute_error(y_train, y_train_pred), 4)
        r2 = round(r2_score(y_train, y_train_pred), 4)
        
        #adj_r2
        n = len(X_train) #number of cases
        p = len(X_train.columns)  #number of features 
        adj_r2 = round(1-(1-r2)*(n-1)/(n-p-1), 4)
        
        print('---------------------------------------------')
        print('For complete train set:')
        print('r2 = {} \nadj_r2 = {} \nMAE = {}'.format(r2, adj_r2, mae))
        end = time.time()
        print('It lasted for {} sec'.format(round(end - start, 3)))
        print('++++++++ End Model +++++++++++++++++++++++++++\n\n')
        models.append(model.best_estimator_)
        result_mae[index] = mae
        result_r2[index] = r2
        result_adj_r2[index] = adj_r2
    
    #error plot
    sns.displot(delta_df, kind="kde")
    plt.xlabel('error')
    plt.ylabel('distribution')
    plt.legend(list(model_dict.values()))
    plt.show()

    #plot for performance comparisons 
    if(len(look_for) > 1): 
        plt.clf()
        plt.plot(model_dict.values(), result_r2.values(), c='r')
        plt.plot(model_dict.values(), result_adj_r2.values(), c='b')
        plt.xlabel('Models')
        plt.xticks(rotation=45)
        plt.ylabel('R2 and adj_R2')
        plt.title('Result of Grid Search')
        plt.legend(['R2', 'Adj_R2'])
        plt.show()
        
    return models, predictions_df, delta_df, result_r2, result_adj_r2, result_mae


'''Functions helps in visualizing percentage of class 1 (responded) in each bucket of deciles'''
def get_decile_wise_distribution(dataframe, target_var):
    
    #columns for which lift chart will be plotted
    columns_to_plot_lift_chart = list(dataframe.drop(columns = [target_var]).columns)
    
    dict_xPoints = {}
    dict_yPoints = {}
    dict_class_1_dist = {}

    for column in columns_to_plot_lift_chart:
        dict_xPoints[str(column)] = []
        dict_yPoints[str(column)] = []
        dict_class_1_dist[str(column)] = []

        results = pd.DataFrame([dataframe[target_var], dataframe[column]]).T
        results.columns = [target_var, 'prediction']
        results = results.sort_values(by = 'prediction', ascending = False)

        results_prediction = results['prediction']
        results_target = results[target_var]
        var_name = target_var
        tuple_cat_decile = pd.qcut(results_prediction, 10, retbins = True, labels = False, duplicates='drop')
        list_cut_offs = list(tuple_cat_decile[1])

        df_actual_var = pd.DataFrame(results_target, columns = [var_name]) #to make array a data frame with a column name
        df_predicted_var = pd.DataFrame(results_prediction, columns = ['Probability_class_1']) # Raw
        df_actual_var['Probability_class_1'] = df_predicted_var['Probability_class_1']

        dict_binned_train = {'cat_train_prob_class_1' : list_cut_offs[1:10]}
        feature = ['cat_train_prob_class_1']

        df_binned = pd.DataFrame(pd.Series([tuple_cat_decile[0]]), columns = ['cat_train_prob_class_1']).iloc[0, 0]
        df_binned = pd.DataFrame(df_binned)
        df_binned[target_var] = results[target_var]    
        df_binned = df_binned.apply(pd.to_numeric)

        cumulative_perc_class_1 = 0

        for i in range(0,10):
            perc_class_1 = len(df_binned.loc[(df_binned['prediction'] == i) & (df_binned[target_var] == 1),])/len(df_binned)
            perc_class_1 = perc_class_1/(len(df_binned.loc[df_binned[target_var] == 1, ])/len(df_binned))
            cumulative_perc_class_1 = cumulative_perc_class_1 + perc_class_1
            dict_xPoints[str(column)].append(i)
            dict_yPoints[str(column)].append(cumulative_perc_class_1)
            dict_class_1_dist[str(column)].append(perc_class_1)

        plt.plot(dict_xPoints[str(column)], dict_class_1_dist[str(column)], label=str(column))
    plt.legend()
    plt.xlabel('Deciles')
    plt.ylabel('percent class 1 (' + target_var +') identified')
    plt.show()
    
    return dict_class_1_dist



'''Getting Optimal cut-off for Model output'''
def get_optimal_cutoff(prediction, target_var):
    y_pred = {}
    dict_f1_index = {}
    df_target = pd.DataFrame(prediction[target_var])

    low = 0
    high = 1
    step = 0.005

    probability_prediction = prediction['probability_class_1']

    for i in np.arange(low, high , step):
        temp = (probability_prediction >= i).astype(int)
        y_pred['prediction_' + str(i)] = pd.DataFrame(temp.values, columns = ['prediction_using_' + str(i)])
        dict_f1_index['f1_at_' + str(i)] = metrics.fbeta_score(df_target[target_var], y_pred['prediction_' + str(i)], beta = 1)

    probability_cutoff = list(np.arange(low, high, step))
    f1_score = list(dict_f1_index.values())
    optimum_cutoff = float(max(dict_f1_index, key = dict_f1_index.get).split("_")[-1])
    return optimum_cutoff


'''Get confusion matix and Precision, Recall at cut_off'''
def get_confusion_matrix(data_prediction, model_output, target_var, cutoff):
    #using optimum cut-off to predict class
    data_prediction['pred_class'] = np.where(data_prediction[model_output] > cutoff, 1, 0)

    #confusion matrix
    conf_matrix = metrics.confusion_matrix(data_prediction[target_var], data_prediction['pred_class'])* 100 / len(data_prediction)
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("\nPrecision is {} and recall is {}, at f1-score: {}".format(round(precision,4),round(recall,4),cutoff))

    print("\nConfusion Matrix (percentage):\n++++++++++++++++++++++++++++++")
    print("Note: Row is Actual and columns is Predicted")
    print(pd.DataFrame(conf_matrix))
    
        
'''Function to plot prediction interval for regression outputs''' 
def plot_prediction_interval(y, y_pred, target_var, is_train = True, interval = 0):
    sum_errs = np.sum((y - y_pred)**2)
    stdev = np.sqrt(1/(len(y)-2) * sum_errs)
    
    print("Prediction Interval: ")
    print("\nA prediction interval for a single future observation is an interval that will, with a specified degree of confidence, contain a future randomly selected observation from a distribution")
    
    # calculate prediction interval
    if is_train:
        interval = 1.96 * stdev
        print('\nPrediction interval is calculated as 1.96 * std_dev...')
    print('Prediction Interval: ± %.3f' % interval)
    
    sorted_df = pd.DataFrame()
    sorted_df['y_pred'] = y_pred
    sorted_df['y'] = y
    
    sorted_df['flag'] = [1 if ((rows.y >= (rows.y_pred - interval)) & (rows.y <= (rows.y_pred + interval))) else 0 for idx, rows in sorted_df.iterrows()]
    print("\nPercentage of points lying in 95% prediction interval for this population: {0:.2f}%".format((sorted_df['flag'].sum()*100/len(sorted_df))))
    
    sorted_df.sort_values(by = ['y','y_pred'], inplace = True)
    sorted_df.reset_index(inplace = True)
    sorted_df.drop(columns = 'index', inplace = True)
    sorted_df.reset_index(inplace = True)
    
    #prediction interval plot
    plt.figure(figsize=[60,40])
    plot_act = plt.scatter(sorted_df.index, sorted_df['y'])
    plot_pred = plt.scatter(sorted_df.index, sorted_df['y_pred'], color='red')
    plot_error = plt.errorbar(sorted_df.index, sorted_df['y_pred'], yerr=interval, color='red', fmt='o')
    plt.grid(axis='y', alpha = 0.75)
    plt.xlabel('case')
    plt.ylabel(target_var)
    plt.legend(['Actual', 'predicted', 'Error bar'], prop={'size': 60})
    plt.show()
    
    #scatter plot of sorted target_var vs predicted target_var
    plt.figure(figsize=[10,8])
    plot_pred = sns.scatterplot(data = sorted_df, x = 'index', y = 'y_pred')
    plot_act = sns.scatterplot(data = sorted_df, x = 'index', y = 'y')
    plt.grid(axis='y', alpha = 0.75)
    plt.xlabel('case')
    plt.ylabel(target_var)
    plt.title("Plot of Sorted " + target_var + " and corresponding predicted_"+ target_var)
    plt.legend(['predicted','actual'])
    plt.show()
    
        
'''PSI Function to compare two distributions'''
#Population stability Index.
def calculate_psi_continuous(expected, actual, buckets = 5):

    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input
    
    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return value

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)    

    psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

    return psi_value
