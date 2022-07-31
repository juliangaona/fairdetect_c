import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExtendedDataframe(pd.DataFrame): #class ExtendedDataframe extends pandas.DataFrame
    """
    Complete and correct way to extend the properties of Dataframes.
    
    Attributes
    ----------     
    pd.DataFrame (DataFrame) :
            Dataframe containing the records and data we want to analyse for bias.        
    
    
    Methods
    -------
    printmd(self,string_to_print) :
        Method that allows us to print the data in markdown format.
        
    summary_report(self) :
        Displays a complete summary report using the pandas-profiling Library.
        
    check_numeric_and_categorical(self) :
        Provides a count of the number of numerical and categorical variables as
        well as a list of variables in each type.

    convert_categorical_to_numeric(self) :
        Converts the categorical variables into numeric ones.  
        
    eda_visualize_for_df(self) :        
        Allows us to visualize the frequency distribution for each variable at once.
        
    check_for_binary(self) :
        Identifies columns with binary fields (1,0).  
        
    eda_descriptive_for_df(self) :    
        Performs all the statistical EDA for the dataset at once.    
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs all the necessary attributes for the ExtendedDataframe object
        
        Parameters
        ---------- 
        *args :
            Parameter used to pass, optionally, a variable number of positional arguments.
        
        **kwargs :
            Parameter used to optionally pass a variable number of named arguments.

        """        
        super().__init__(*args,**kwargs)
        
    
    @property
    def _constructor(self):
        return ExtendedDataframe

    
    def printmd(self,string_to_print):
        """
        Method that allows us to print the data in markdown format.
        
        Using this method, we can set the headers, subtitles... that we want to
        show to the end user in Markdown format.

        Parameters
        ---------- 
        string_to_print :
            Data we want to display in markdown format

        Returns
        -------
            Prints the data of the string_to_print parameter in a Markdown format.
            
        """
        from IPython.display import display, HTML, Markdown
        display(HTML("<style>.container { width:100% !important; }</style>"))

        display(Markdown(string_to_print))
        
    def summary_report(self):
        """
        Displays a complete summary report using the pandas-profiling Library.
        
        This method allows us, using the pandas-profiling library, to generate a 
        report with a detailed exploratory data analysis. In order to make it work,
        first we have to install "pip install markupsafe==2.0.1".
        
        Returns
        -------
            Shows the user the complete report generated.        
        
        """
        #prints the summary report using the ProfileReport Library. Add a note that pre-requisite is to install pip install markupsafe==2.0.1
        from pandas_profiling import ProfileReport
        report = ProfileReport(self, minimal=False)
        return report   
   
    def check_numeric_and_categorical(self):
        """
        Provides a count of the number of numerical and categorical variables as
        well as a list of variables in each type.
        
        Returns
        -------
        check_numeric (list) :
            List containing the names of the numeric variables.
        check_categorical (list) :
            List containing the names of the categorical variables.
            
        """
        counter_numeric=0
        counter_categorical=0
        check_numeric = []
        check_categorical=[]
        for column_name in self:
            if (self[column_name].dtypes =='int64') | (self[column_name].dtypes == 'float64') :
                counter_numeric+=1
                check_numeric.append(column_name)
            if self[column_name].dtypes =='object':
                counter_categorical+=1
                check_categorical.append(column_name)
        return check_numeric, check_categorical
        
    def convert_categorical_to_numeric(self):
        """
        Converts the categorical variables into numeric ones.
        
        This method identifies variables whose data are numerical and those whose
        data are categorical. It then uses the LabelEncoder method, included in
        the Preprocessing package of the SciKit Learn library. In this way, we
        encode the categorical variables with value between 0 and n_classes-1,
        being able to apply machine learning models to the data.
        
        Returns
        -------
            Will show us each column name, telling the user if it is continuous/numeric
            or categorical. Will encode the categorical ones.
            
        """
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        for column_name in self:
            if (self[column_name].dtypes =='int64') | (self[column_name].dtypes == 'float64') :
                print (column_name, "is continuous/numeric")
            if self[column_name].dtypes =='object':
                print (column_name, "is categorical")
                self[column_name] = le.fit_transform(self[column_name])
                
    def eda_visualize_for_df(self):
        """
        Allows us to visualize the frequency distribution for each variable at once.
        
        It shows, for each of the variables in the dataset, whether categorical
        or numerical, a histogram that allows the user to observe the distribution
        of values and frequencies of each of the variables.
        
        Prints/Displays
        ---------------
            As many histograms as variables has our dataset, showing the frequency
            distribution for each of the variables.
        
        """
        
        numeric, categorical = self.check_numeric_and_categorical()
        self.printmd("# Your Data has the following Distribution (Histogram)")
        self[numeric].hist(bins=15, figsize=(20, 20), layout=(5, 5));        
        
        if len(categorical) !=0:
            self.printmd("# Your Categories has the following Distribution (Bar Plots)")
            fig, ax = plt.subplots(len(categorical), len(categorical), figsize=(15, 15))
            
            for variable, subplot in zip(categorical, ax.flatten()):
                sns.countplot(self[variable], ax=subplot)
                for label in subplot.get_xticklabels():
                    label.set_rotation(90)
            double_cat= len(categorical)*len(categorical)
            for j in reversed(range(0,double_cat)):
                if j>=len(categorical):
                    ax.flat[j].set_visible(False)            
                                
    
    def check_for_binary(self):
        """
        Identifies columns with binary fields (1,0).
        
        In order to help the user to properly identify the potential sensitive
        variables already prepared to perform the bias analysis, it indicates the
        list of binary variables, which are those that we can later analyse using
        the Fairdect class.
        
        Returns
        -------
        sensitive_list (list):
            List containing the columns with binary fields inside the dataframe.
            
        """
        #check the columns that contain binary fields (0,1) so that it suggests to the user the sensitive variables
        sensitive_list = self.columns[self.isin([0,1]).all()]
        return list(sensitive_list)
               
        
    
    def eda_descriptive_for_df(self):
        """
        Performs all the statistical EDA for the dataset at once.
        
        This method allows the user to perform an initial EDA by displaying the
        statistical data of the dataset. It shows the shape of the dataset, the
        columns, the types of data in each column, which variables are categorical
        and which are numerical, the missing values for each column, and the 
        average of the values in each column. To carry out this analysis, the 
        method uses methods previously defined within the ExtendedDataframe class,
        specifically "printmd" and "check_numeric_and_categorical".
        
        Prints/Displays
        ---------------
            The methods prints an statistical analysis of the dataframe, containing 
            the shape of the dataset, the columns, the types of data in each column,
            which variables are categorical and which are numerical, the missing values
            for each column, and the average of the values in each column.
            
        """
        self.printmd("# KNOW YOUR DATA (KND)- Exploratory Data Analysis (EDA)\n")
        
        self.printmd("## Know Your Dataset Shape  (rows,columns)")
        display(self.shape)
        
        self.printmd("## Know Your Dataset Columns")
        display(self.columns)
        
        
        self.printmd("## Know Your Dataset Data Types")
        display(self.dtypes)
        
        self.printmd("## Know Your Dataset Data Numeric and Categorical Columns")
        numeric, categorical = self.check_numeric_and_categorical()
        self.printmd("### Numeric Columns")
        print(numeric)
        self.printmd("### Categorical Columns")
        print(categorical)
       
        self.printmd("## Know your Missing Values")
        df_null= self.isnull().sum().to_frame()
        display(df_null)

        self.printmd("## Know some Statistics (mean)")
        display(self.mean().to_frame())
        
        self.printmd("## Know your counts")
        for self.column_name in self.columns:
            if self[self.column_name].dtypes =='object':
                display(self[self.column_name].value_counts().to_frame())
    