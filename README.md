---------------------------------------------------------------------------------------------
 Files 
---------------------------------------------------------------------------------------------

 problem_statement.pdf: Defines the problem to be solved.

 requirements.txt: This file has all the required standard python packages used in building this project

---------------------------------------------------------------------------------------------
 codes
---------------------------------------------------------------------------------------------

 modeling_functions.py: A python file having all the functions which were used repeatledly during Modelling this project. Below are the functions.

	1. get_cross_table: function returns a cross-table of feature w.r.t the target variable
	2. plot_boxplot: Function to plot the box-plot
	3. correlation: function to remove one of the highly correlated (threshold) column
	4. chi2_test: function to perform the chi2 test
	5. krushkal_wallis_test:Function which performs features selection test: Krushal Wallis on input Categorical and numeric output data
	6. train_classification_model: Function to iterate over classification models and obtain results
	7. train_regression_model: Function to iterate over Regression models and obtain results
	8. get_decile_wise_distribution(dataframe, target_var): Functions helps in visualizing percentage of class 1 (responded) in each bucket of deciles
	9. get_optimal_cutoff: Getting Optimal cut-off for Model outpu
	10. get_confusion_matrix: Get confusion matix and Precision, Recall at cut_off
	11. plot_prediction_interval: Function to plot prediction interval for regression outputs
	12. calculate_psi_continuous: PSI (Population stability Index:) Function to compare two distributions and to see if population is stable.


 Below flow to be followed sequentially.

 1_responde_model.ipynb: First a response Model is built based on binary classification keeping target variable as 'responded'. To predict who will respond.

 2_profit_model.ipynb: After building above the model, the model to predict 'Profit' is built for customers who would respond. 
		       So, that we should only contact people likely to respond and will be profitable to the firm.
                       (to identify the profitable customers, cutoff decided basis prediction interval of regression output)

 3_predicting_output.ipynb: Here, customers in 'testingCandidate.csv' data are tagged using above two models. 
		     	    First, responding customers are identified and then profitable customers out of responding customers are identified.
		            The interesection of two (respond and profit) are marked as 1 and these customers should be marketed.
		            Here, PSI was generated for the batch output with testing time, to see if distribution is still holding true (validation). 

---------------------------------------------------------------------------------------------
 data 
---------------------------------------------------------------------------------------------

 i.e. Input Files

 training_csv.csv: Training data used to build the above two models.

 testingCandidate_csv.csv: The batch file for which the output is required to be generated.

---------------------------------------------------------------------------------------------
 output 
---------------------------------------------------------------------------------------------
 testingCandidate_with_output.csv (output columns: 'should_market'. (1 means yes and 0 means no))


---------------------------------------------------------------------------------------------
 pickles 
---------------------------------------------------------------------------------------------

 pickle_reponse_model: pickles from response model are stored here.

 pickle_profit_model: pickles from profit model are stored here.


