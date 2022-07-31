#we kept those libraries as they are needed by more than one method
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import chisquare
import fairlearn.metrics as met
import sklearn.metrics as msl
from sklearn.metrics import *

#Improvements: we removed the following functions as they are either redundent or not needed
#import dalex as dx #not needed - removed
#import pandas as pd #remove as it is redundent


#Improvements: the following imports were moved to methods as they are only used there 
#from random import randrange #moved to the understand_shap method
#from scipy.stats import chi2_contingency #moved to representation
#from tabulate import tabulate moved to the representation method
#from sklearn.metrics import confusion_matrix # moved to the ability method
#from sklearn.metrics import precision_score #moved to the predictive method



class FairDetect:
    """
    A class to detect the existence of bias depending on certain binary sensitive 
    group/feature in machine learning models.
    
    Attributes
    ---------- 
    my_dataframe :
        Dataframe containing the records and data we want to analyse for bias.
    
    model :
        Model is an object obtained after applying the classification Machine Learning algorithm.
        It can use any classification algorithm for binary predictions.
    
    X_test (DataFrame) :
        DataFrame given the test columns for each record inside the X_test set, except the target variable.    
    
    y_test (DataFrame) :
        DataFrame given for test target variable.    


        
    
    Methods
    -------
    create_labels(self,sensitive):
        Allows to create labels for the sensitive groups.
        
    representation(self,sensitive,labels,predictions):
        Compares the representation of the sensitive variable and its 
        association with the target and displays the results in charts.
    
    ability(self,sens_df,labels,XXXXX):
        Compares the metrics disparities for the sensitive features.
    
    ability_plots(self,labels,XXXXXXX):
        Graphically represents the metrics dispersion.
    
    ability_metrics(self,XXXXXXXX):
        Using hypoyhesis testings, it examines the metrics disparities within the  
        sensitive features.
    
    predictive(self,labels,sens_df):
        Compares the distribution within the dataset with the distribution 
        within the prediction set.    
    
    identify_bias(self,sensitive,labels):
        It allows to observe in a complete way all the analysis related to the
        existence of bias in relation to the sensitive group.
    
    understand_shap(self,labels,sensitive,affected_group,affected_target):
        Quantifies the contribution that each feature makes to the prediction 
        made by the model individually and separately, with each of the categories
        of the sensitive features as pre-specified, allowing us to see their 
        marginal contribution to the model.        
    
    """
    def __init__(self,my_df, model, X_test, y_test):
        """
        Constructs all the necessary attributes for the person object
        
        Parameters
        ----------        
        my_dataframe :
            Dataframe containing the records and data we want to analyse for bias.
        model :
            Model is an object obtained after applying the classification Machine Learning algorithm.
            It can use any classification algorithm for binary predictions.
    
        X_test (DataFrame) :
            DataFrame given the test columns for each record inside the X_test set, except the target variable.    
    
        y_test (DataFrame) :
            DataFrame given for test target variable.    

        """
        
        self.my_dataframe = my_df
        self.model,self.X_test,self.y_test= model, X_test, y_test          
                
        
        
        
    def create_labels(self,sensitive):
        """ 
        Allows to create labels for the sensitive groups.
        
        The user must enter in the input parameters the name of the column which  
        contains the values that we consider sensitive and for which we want to
        perform a potential discrimination analysis. Then we can assign different
        labels (i.e. 0 for Female, 1 for Male) to each value of the variable.
        
        Parameters
        ----------
        sensitive : 
            Column of the dataframe object of analysis which has sensitive values.
        
        Returns
        -------
        sensitive_label (dict):
            Dictionary that maps each of the user-entered inputs (labels) to the 
            values of the column selected as sensitive.

        """
        #Improvement: Raising an error if the user tries to assign a string to the sensitive variable. Our program will               throw a ValueError with the message below
        
        self.sensitive = self.my_dataframe[sensitive]
        data_arr = self.sensitive
        
        try:
            self.sensitive.astype(int)
            
        except:
            raise ValueError('Sensitive value should be numerical or categorical')
            
            
        sensitive_label = {}
        for i in set(self.X_test[sensitive]):
            text = "Please Enter Label for Group" +" "+ str(i)+": "
            label = input(text)
            sensitive_label[i]=label
        return(sensitive_label)


    def representation(self,sensitive,labels,predictions):
        """
        Compares the representation of the sensitive variable and its 
        association with the target and displays the results in charts.
        
        This function generates two histograms comparing, in the first one,
        the frequency with which each of the values of the sensitive feature
        appears in the dataset and, in the second one, the frequency of each
        of the values of the target. It then generates a frequency distribution
        table that allows us to compare the weight of each of the values of the
        target variable for each of the values of the sensitive feature. Also
        it generates the p-value and performs an Hypothesis Testing in order 
        to find out if there's any significant relation between the sensitive
        value and the target variable.
        
        Parameters
        ----------
        sensitive : 
            Column of the dataframe object of analysis which has sensitive values.
        labels (str) :
            Names previously assigned from method create_labels() to each of the values 
            of the sensitive column.
        predictions :
            Predictions of the target variable previously computed for all the
            records.
        
        Returns
        -------
        cont_table :
            Table of frequencies in percentage. Relates the frequency of each value
            of the target variable for each of the sensitive feature values.
        sens_df (dict) :
            Dictionary which stores separate tables for each of the values of the
            labels of the sensitive feature, and relates them with the target.
        fig :
            Two histograms with frequencies of the sensitive and the target variables
        p :
            P-value computed using a chi-square test for the contingency table, 
            indicating level of relation between target and sensitive variables.

        """
        from tabulate import tabulate
        from scipy.stats import chi2_contingency
        
        full_table = self.X_test.copy()
        sens_df = {}

        for i in labels:
            full_table['p'] = predictions
            full_table['t'] = self.y_test
            sens_df[labels[i]] = full_table[full_table[sensitive]==i]

        contigency_p = pd.crosstab(full_table[sensitive], full_table['t']) 
        cp, pp, dofp, expectedp = chi2_contingency(contigency_p) 
        contigency_pct_p = pd.crosstab(full_table[sensitive], full_table['t'], normalize='index')

        #Improvement: Simplify the code; more readable - one loop instead of two

        sens_rep = {}
        labl_rep = {}
        # Improvement:Simplification   of   the   value_counts().sum()   function   into count()
        # achieving the same result, as can be seen in our project.
        for i in labels:
            sens_rep[labels[i]] = (self.X_test[sensitive].value_counts()/self.X_test[sensitive].count())[i]
            labl_rep[str(i)] = (self.y_test.value_counts()/self.y_test.count())[i]


        fig = make_subplots(rows=1, cols=2)

        for i in labels:
            fig.add_trace(go.Bar(
            showlegend=False,
            x = [labels[i]],
            y= [sens_rep[labels[i]]]),row=1,col=1)

            fig.add_trace(go.Bar(
            showlegend=False,
            x = [str(i)],
            y= [labl_rep[str(i)]],
            marker_color=['orange','blue'][i]),row=1,col=2)

        c, p, dof, expected = chi2_contingency(contigency_p)
        cont_table = (tabulate(contigency_pct_p.T, headers=labels.values(), tablefmt='fancy_grid'))

        return cont_table, sens_df, fig, p



    def ability(self,sens_df,labels,predictions,sen_feat_test):
        """
        Compares the metrics disparities for the sensitive features.

        
        Allows us to obtain the confusion matrix that relates the sensitive 
        variable to the target variable. We can contrast the values observed in 
        the current classification for the target variable, grouped by 
        sensitive variable, with the predicted values for them. Once the matrix 
        is obtained, we calculate the rates of true positives, false positives,
        true negatives and false negatives.
        
        Parameters
        ----------
        labels (str) :
            Names previously assigned to each of the values of the sensitive column.
        sens_df (dict) :
            Dictionary which stores separate tables for each of the values of the
            labels of the sensitive variable, and relates them with the target.
        sen_feat_test(list):
            A list with a sample of the X_test for the sensitive feature. It is required 
            to execute the MetricFrame method from FairLearn Library.
            
        Returns
        -------
        metric_frame:
            A Fairlearn object that contains a collection of disaggregated metric values, 
            and this metric_frame has attributes like: 
            
            by_group: 
                Return the collection of metrics evaluated for each subgroup.
            
            control_levels: 
                Return a list of feature names which are produced by control features.
            
            overall:
                Return the underlying metrics evaluated on the whole dataset.

            sensitive_levels:
                Return a list of the feature names which are produced by sensitive features.
            
            default metrics:
                true_positive_rate_m, false_positive_rate_m, true_negative_rate_m, false_negative_rate_m,
                selection_rate, accuracy_score
        
        
        """
        from sklearn.metrics import confusion_matrix
        
        #Improvement: this improvement makes generating metrics more generic, not limited to TP, TN, FP, FN.
        #Just add to the dictionary and it will calculate. This will make it much easier to edit the code  
        #in case the business requires different metric assessment
        metric_frame = met.MetricFrame(metrics={"true_positive_rate_m": met.true_positive_rate,
                                                    "false_positive_rate_m": met.false_positive_rate,
                                                    "true_negative_rate_m": met.true_negative_rate,
                                                    "false_negative_rate_m":  met.false_negative_rate,
                                                    "selection_rate": met.selection_rate,
                                                    "accuracy_score": msl.accuracy_score 
                                                    },
                                           sensitive_features=sen_feat_test,
                                           y_true=self.y_test,
                                           y_pred=predictions)


        return(metric_frame)

        #this is Ryan's code
        """sens_conf = {}
        for i in labels:
            sens_conf[labels[i]] = confusion_matrix(list(sens_df[labels[i]]['t']), list(sens_df[labels[i]]['p']), labels=[0,1]).ravel()

        ### true positive, false positive, true negatives and false negatives rates
        ### names are abbreviated in order to homogenise the code.
        TPR = {}
        FPR = {}
        TNR = {}
        FNR = {}

        for i in labels:
            TPR[labels[i]] = (sens_conf[labels[i]][3]/(sens_conf[labels[i]][3]+sens_conf[labels[i]][2]))
            FPR[labels[i]] = (sens_conf[labels[i]][1]/(sens_conf[labels[i]][1]+sens_conf[labels[i]][0]))
            TNR[labels[i]] = 1 - FPR[labels[i]]
            FNR[labels[i]] = 1 - TPR[labels[i]]

        return(TPR, FPR, TNR, FNR)
    """


    def ability_plots(self,labels,metric_frame):
        """ 
        Graphically represents the metrics dispersion.
        
        It graphically represents the distribution of metrics between 
        each of the values of the sensitive feature, in relation to the 
        predictions made on the target variable.
        
        Parameters
        ----------
        labels (str) :
            Names previously assigned to each of the values of the sensitive column.
        metric_frame:
            A Fairlearn object that contains a collection of disaggregated metric values, 
            and this metric_frame has attributes like: 
            
            by_group: 
                Return the collection of metrics evaluated for each subgroup.
            
            default metrics:
                true_positive_rate_m, false_positive_rate_m, true_negative_rate_m, false_negative_rate_m,
                selection_rate, accuracy_score
            
        Displays/Prints
        ---------------
            Column charts, showing how the metrics in the metric_frame are distributed 
            among the different values of the sensitive variable.
        
        """

        #Improvement: plotting the chart using the by_group from the metric_frame, to make it more generic. So
        #whaever metrics, they will be printed
        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3,3], #put this in a forloop
            legend=False,
            figsize=[12, 8],
            title="Show all metrics",
            )
        
        #This is Ryan's code  
        """     
        
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("True Positive Rate", "False Positive Rate", "True Negative Rate", "False Negative Rate"))

        x_axis = list(labels.values())
        fig.add_trace(
            go.Bar(x = x_axis, y=list(TPR.values())),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FPR.values())),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(TNR.values())),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x = x_axis, y=list(FNR.values())),
            row=2, col=2
        )

        fig.update_layout(showlegend=False,height=600, width=800, title_text="Ability Disparities")
        fig.show()
        """
            
    def ability_metrics(self,metric_frame):

        """
        Using hypoyhesis testings, it examines the metrics disparities within the  
        sensitive feature.
        
        Using a chi-square test, obtains the p-value for each of the previously 
        calculated rates (TPR, FPR, TNR, FNR). Subsequently, it performs a 
        Hypothesis Testing comparing one by one, each of the p-values with
        differents Significance Levels (0.01, 0.05 and 0.1) indicating to the 
        user whether the Null Hypothesis is rejected or not for each of the SL.
        
        Parameters
        ----------
        metric_frame:
            A Fairlearn object that contains a collection of disaggregated metric values, 
            and this metric_frame has attributes like: 
            
            by_group: 
                Return the collection of metrics evaluated for each subgroup.
            
            default metrics:
                true_positive_rate_m, false_positive_rate_m, true_negative_rate_m, false_negative_rate_m
        
        Displays/Prints
        ---------------        
            As a result, we obtain four texts, one for each of the rates calculated 
            in the previous metrics (TPR, FPR, TNR and FNR). In each of the texts, 
            we indicate whether, by contrasting the p-value with the Significance 
            Level, we reject or not the Null Hypothesis, concluding whether or not 
            there is disparity in each of the previously mentioned categories. 
            Depending on the p-value, we can, in each of the rates, reject the null
            hypothesis with a confidence level of 99%, 95% or 90%. Depending on the 
            confidence with which we can reject, if applicable, the Null Hypothesis,
            the rejection will be indicated.
            In the case of obtaining p-values greater than a SL of 0.1, the 
            Null Hypothesis is accepted indicating the non-existence of disparities.
            
        """
        #Improvement: calculating the chisqaure for the TP,TN,FP and FN in a loop using the dictionary of 
        #the metric and storing the p-values in a dic_p_val dictionary so that we can access later in a loop
        #to compare to confidence level and alpha
        
        dic_met = metric_frame.by_group.to_dict(orient='list')
        dic_p_val = {}
        list_test = ['true_positive_rate_m','false_positive_rate_m','true_negative_rate_m','false_negative_rate_m']
        for key,items in dic_met.items():
            if key in list_test:
                p_val =chisquare(list(np.array(list(items))*100))[1]
                dic_p_val[key]=p_val
        #Ryan's code
        """
        TPR_p = chisquare(list(np.array(list(TPR.values()))*100))[1]
        FPR_p = chisquare(list(np.array(list(FPR.values()))*100))[1]
        TNR_p = chisquare(list(np.array(list(TNR.values()))*100))[1]
        FNR_p = chisquare(list(np.array(list(FNR.values()))*100))[1]
        """        
        #Improvement: looping over the p-values dictionary dic_p_val, and decide to accept/reject the HO
        #based on confidence levels. Code is reduced to 9 lines instead of 40 lines
        
        for key,items in dic_p_val.items():
            if items <= 0.01:
                print("With 99% Confidence Level, Reject H0: "+ key +" with p= ", items)
            elif items <= 0.05:
                print("With 95% Confidence Level, Reject H0: "+ key +" with p= ", items)
            elif items <= 0.1:
                print("With 90% Confidence Level, Reject H0: "+ key +" with p= ", items)
            else:
                print("Accept H0: " + key +" Disparity is Not Detected. p= ", items)        
             
        """
        if TPR_p <= 0.01:
            print("*** Reject H0: Significant True Positive Disparity with p=",TPR_p)
        elif TPR_p <= 0.05:
            print("** Reject H0: Significant True Positive Disparity with p=",TPR_p)
        elif TPR_p <= 0.1:
            print("*  Reject H0: Significant True Positive Disparity with p=",TPR_p)
        else:
            print("Accept H0: True Positive Disparity Not Detected. p=",TPR_p)

        if FPR_p <= 0.01:
            print("*** Reject H0: Significant False Positive Disparity with p=",FPR_p)
        elif FPR_p <= 0.05:
            print("** Reject H0: Significant False Positive Disparity with p=",FPR_p)
        elif FPR_p <= 0.1:
            print("*  Reject H0: Significant False Positive Disparity with p=",FPR_p)
        else:
            print("Accept H0: False Positive Disparity Not Detected. p=",FPR_p)

        if TNR_p <= 0.01:
            print("*** Reject H0: Significant True Negative Disparity with p=",TNR_p)
        elif TNR_p <= 0.05:
            print("** Reject H0: Significant True Negative Disparity with p=",TNR_p)
        elif TNR_p <= 0.1:
            print("*  Reject H0: Significant True Negative Disparity with p=",TNR_p)
        else:
            print("Accept H0: True Negative Disparity Not Detected. p=",TNR_p)

        if FNR_p <= 0.01:
            print("*** Reject H0: Significant False Negative Disparity with p=",FNR_p)
        elif FNR_p <= 0.05:
            print("** Reject H0: Significant False Negative Disparity with p=",FNR_p)
        elif FNR_p <= 0.1:
            print("*  Reject H0: Significant False Negative Disparity with p=",FNR_p)
        else:
            print("Accept H0: False Negative Disparity Not Detected. p=",FNR_p)


        """

    def predictive(self,labels,sens_df):
        """
        Compares the distribution within the dataset with the distribution 
        within the prediction set.
        
        It calculates the precision score for each of the categories of the 
        sensitive variable and stores these values in a dictionary. Based on the 
        calculated values, it generates a column chart to observe the differences
        and computes the p-value using a chi-square test.
        
        Parameters
        ---------- 
        labels (str) :
            Names previously assigned to each of the values from method 
            create_labels() of the sensitive column.
            
        sens_df (dict) :
            Dictionary which stores separate tables for each of the values of the
            labels of the sensitive feature, and relates them with the target.        
        
        Returns
        ------- 
        precision_dic (dict) : 
            Dictionary which stores, for each of the different categories inside the 
            sensitive variable, its precision score.
        fig :
            Column chart showing the precision score for each of the categories of 
            the sensitive feature.
        pred_p (float) :
            P-value obtained with the two values of the precision_dic dictionary. It 
            will be used to analyse if there's any model exacerbation of biases.
        
        """
        from sklearn.metrics import precision_score

        precision_dic = {}
        for i in labels:
            precision_dic[labels[i]] = precision_score(sens_df[labels[i]]['t'],sens_df[labels[i]]['p'])

        fig = go.Figure([go.Bar(x=list(labels.values()), y=list(precision_dic.values()))])

        pred_p = chisquare(list(np.array(list(precision_dic.values()))*100))[1]

        return(precision_dic,fig,pred_p)




    def identify_bias(self,sensitive,labels):
        """
        It allows to observe in a complete way all the analysis related to the
        existence of bias in relation to the sensitive feature.
        
        Runs the previous defined methods grouping the outputs: representation, 
        ability, ability_plots, ability_metrics and predictive. It shows the user
        the conclusions reached in each of the previous methods in a grouped way,
        allowing to simplify the use by calling a single method. For more details, 
        reference is made to the documentation for each of the methods mentioned.

        Parameters
        ----------
        sensitive : 
            Column of the dataframe object of analysis which has sensitive values.
        
        labels (str) :
            Names previously assigned to each of the values from method 
            create_labels() of the sensitive column.
            
        predictions :
            Predictions of the target variable previously computed for all the
            records.
            
        
        Displays/Prints
        ---------------    
        rep_fig :
            Two histograms with frequencies of the sensitive and the target variables.
            They are derived from the "representation" method and use as inputs the 
            results achieved in the "representation" method.
        
        cont_table :
            Table of frequencies in percentage. Relates the frequency of each value
            of the target variable for each of the sensitive variable values. 
            Derived from the "representation" method.
            
            Using the rep_p obtained in the "representation" method, peforms an 
            Hypothesis testing and tells us if there's significante relation between 
            the sensitive and the target variables.
            
        fig :    
            Four column charts, each showing how TP, FP, TN and FN are distributed 
            among the different values of the sensitive feature. Derived from the 
            "ability_plots" method.
            
            Prints four texts, one for each of the rates calculated (TPR, FPR, TNR and
            FNR). In each of the texts, we indicate whether, by contrasting the 
            p-value with the Significance Level, we reject or not the Null Hypothesis,
            concluding whether or not there is disparity in each of the previously
            mentioned categories. Depending on the p-value, we can, in each of the
            rates, reject the null hypothesis with a confidence level of 99%, 95% or
            90%. Depending on the confidence with which we can reject, if applicable,
            the Null Hypothesis, the rejection will be indicated. In the case of
            obtaining p-values greater than a SL of 0.1, the Null Hypothesis is accepted
            indicating the non-existence of disparities. Derived from the "ability_metrics" method.
          
        pred_fig :
            Column chart showing the precision score for each of the categories of 
            the sensitive feature. Derived from the "predictive" method.
            
            Using the pred_p obtained in the "predictive method" performs an Hypothesis
            testing and prints if Significant Predictive disparity exists.

        Raises
        -------  
        KeyError:
            When the user writes another variable name inside the function that is
            not the same as the sensitive variable. The sensitive variable name does
            not appear inside the dataframe. Data type non suported by the function.
            The number of labels does not match the number of different categories 
            of the target variable.
        """
        
    
        predictions = self.model.predict(self.X_test)
        cont_table,sens_df,rep_fig,rep_p = self.representation(sensitive,labels,predictions)
        
        print("REPRESENTATION")
        rep_fig.show()

        print(cont_table,'\n')
        
        
        
        
        #Improvement: better user message relating the confidence level to the p-value
        if rep_p <= 0.01:
            print("With 99% Confidence Level, Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
        elif rep_p <= 0.05:
            print("With 95% Confidence Level, Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
        elif rep_p <= 0.1:
            print("With 90% Confidence Level, Reject H0: Significant Relation Between",sensitive,"and Target with p=",rep_p)
        else:
            print("Accept H0: No Significant Relation Between",sensitive,"and Target Detected. p=",rep_p)


        precision_dic, pred_fig, pred_p = self.predictive(labels,sens_df)
        print("\n\nPREDICTIVE")
        pred_fig.show()
        
        #Improvement: better user message relating the confidence level to the p-value
        if pred_p <= 0.01:
            print("With 99% Confidence Level, Reject H0: Significant Predictive Disparity with p=",pred_p)
        elif pred_p <= 0.05:
            print("With 95% Confidence Level, Reject H0: Significant Predictive Disparity with p=",pred_p)
        elif pred_p <= 0.1:
            print("With 90% Confidence Level, Reject H0: Significant Predictive Disparity with p=",pred_p)
        else:
            print("Accept H0: No Significant Predictive Disparity. p=",pred_p)
            
        print("\n\nABILITY")
        
        #Improvement: Instead of creating 4 variables for the TP,TN,FP and FN, the improvement is to create a metric_frame
        #that support more group of metrics and plot as many metrics as added in the metric_frame dictionary, without 
        #adding more code or complexity
        
        sen_feat_test = self.X_test[sensitive]
        metric_frame = self.ability(sens_df,labels,predictions,sen_feat_test)
        self.ability_metrics(metric_frame)       
        self.ability_plots(labels,metric_frame)

    def understand_shap(self,labels,sensitive,affected_group,affected_target):
        """
        Quantifies the contribution that each feature makes to the prediction 
        made by the model individually and separately, with each of the categories
        of the sensitive features as pre-specified, allowing us to see their 
        marginal contribution to the model.
        
        It uses the library SHAP (SHapley Additive exPlanations). This library
        applies the Shapley value, a solution concept in cooperative game theory
        named in honor of Lloyd Shapley. It uses calculations from the field of 
        game theory to find out which variables have the most influence on the 
        predictions of machine learning techniques, allowing us to isolate and
        identify the causes of the bias.
        
        Parameters
        ----------
        labels (str) :
            Names previously assigned to each of the values from method 
            create_labels() of the sensitive column.
            
        sensitive : 
            Column of the dataframe object of analysis which has sensitive values.
            
        affected_group (int):
            Category of the sensitive variable for which we have identified the bias
            and on which we want to focus our analysis.
            
        affected_target (int):
            Category of the target variable that we want to focus on in the analysis.
            
        Displays/Prints
        --------------- 
        Model Importance Comparison (fig):
            It returns two bar charts. In the left graph, we have an overall 
            comparison in which we can check the marginal contribution, that is, 
            the importance, of each of the features for the model according to the 
            category of the selected sensitive variable. In the right graph, we can
            check the weight that each of the features has in the specific case of
            the parameters that we have set in the function as affected_group and
            affected_target.
            
        Average Comparison to True Class Members (fig) :
            Column chart comparing how each of the attributes affects the group misplaced
            differently in relation to other records within the same group. We can
            see the differences in attributes that our affected group have in contrast
            to the records of the class of which they should be a part if they were 
            not incorrectly classified.
            
        Average Comparison to All Members (fig) :
            Column chart comparing how each of the attributes affects the group misplaced
            differently in relation to all the other records.
        
        Affected Decision Process (fig) :
            We select, at random, a record included in the affected_group incorrectly
            classified in the affected_target. It shows us a graph in which we can
            see, for that record, the importance that each of the features has had
            for that record to be classified incorrectly.

        Raises
        ------- 
        KeyError:
            When the user writes another variable name inside the function that is
            not the same as the sensitive variable. 
            The sensitive variable name does not appear inside the dataframe. 
            Data type not suported by the function.
            The number of labels does not match the number of different categories 
            of the target variable.
        
        IndexError:
            When the int include in our affected_group or in our affected_target does not appear in our dataset.
        """
        from random import randrange
        import shap
        explainer = shap.Explainer(self.model)

        full_table = self.X_test.copy()
        full_table['t'] = self.y_test
        full_table['p'] = self.model.predict(self.X_test)
        full_table

        shap_values = explainer(self.X_test)
        sens_glob_coh = np.where(self.X_test[sensitive]==list(labels.keys())[0],labels[0],labels[1])

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]
        shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
        #sens_mis_coh = np.where(affected_class[sensitive]==list(labels.keys())[0],labels[0],labels[1])


        figure,axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
        plt.subplots_adjust(right=1.4,wspace=1)

        print("Model Importance Comparison")
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        shap.plots.bar(shap_values.cohorts(sens_glob_coh).abs.mean(0),show=False)
        plt.subplot(1, 2, 2) # row 1, col 2 index 1
        shap_values2 = explainer(affected_class.drop(['t','p'],axis=1))
        shap.plots.bar(shap_values2)
        #shap.plots.bar(shap_values2)

        full_table['t'] = self.y_test
        full_table['p'] = self.model.predict(self.X_test)
        #full_table=full_table[['checking_account','credit_amount','duration','sex','t','p']]

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

        truclass = full_table[full_table.t == full_table.p]
        tru_class = truclass[(truclass[sensitive] == affected_group) & (truclass.t == affected_target)]

        x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
        affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-tru_class.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

        #plt.figsize([10,10])
        #plt.bar(x_axis,affect_character)

        fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])

        print("Affected Attribute Comparison")
        print("Average Comparison to True Class Members")
        fig.show()

        misclass = full_table[full_table.t != full_table.p]
        affected_class = misclass[(misclass[sensitive] == affected_group) & (misclass.p == affected_target)]

        #truclass = full_table[full_table.t == full_table.p]
        tru_class = full_table[(full_table[sensitive] == affected_group) & (full_table.p == affected_target)]

        x_axis = list(affected_class.drop(['t','p',sensitive],axis=1).columns)
        affect_character = list((affected_class.drop(['t','p',sensitive],axis=1).mean()-full_table.drop(['t','p',sensitive],axis=1).mean())/affected_class.drop(['t','p',sensitive],axis=1).mean())

        #plt.figsize([10,10])
        #plt.bar(x_axis,affect_character)

        fig = go.Figure([go.Bar(x=x_axis, y=affect_character)])
        print("Average Comparison to All Members")
        fig.show()

        print("Random Affected Decision Process")
        explainer = shap.Explainer(self.model)
        shap.plots.waterfall(explainer(affected_class.drop(['t','p'],axis=1))[randrange(0, len(affected_class))],show=False)


    

    