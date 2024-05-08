from explore_data import explore_data, explore_values, statistics, pair_plot, hist_plot, analyse_usage, print_crosstab
from predictions import ml_analysis

if __name__ == '__main__':
    file_path = 'Data/customer_churn.csv'
    explore_values(file_path)
    statistics(file_path)
    # Findings:
    # label: 'Churn' is integer
    # - 12 int variables, 1 float variable 'Customer Value'
    # No NaNs.

    # Pair plot (scatter matrix) to get a quick overview:
    pair_plot(file_path,
              'Results/pairplot.png')
    # There seem to be some correlations visible between customers churning and
    # - low usage (Seconds of Use, Freq. of Use, Freq. of SMS, Distinct Called Numbers)
    # - having complained
    # - being in Tariff Plan 1
    # - having Status 2 (nearly 50%!)
    # - low customer value.

    # A lot less customers churned than stayed, therefore relative numbers will be helpful.
    hist_plot(file_path, 'Results/histplot.png')
    # Histograms relative for both 'Churn' categories confirm that leaving
    # correlates with increased number of complaints, a low charged amount, low usage,
    # being in tariff 1 and status 2, as well as low customer value.
    # Totally irrelevant was the number of Call Failures,
    # and mostly irrelevant were Subscription Length and Age, although people <20 and >50 mostly didn't leave.

    # One question if the usage also correlates with status and tariff plan, which the pairplot indicated,
    # and if the usage correlation of the churn could be indirect.
    analyse_usage(file_path, 'Results/usage_tariff_plans.png', 'Tariff Plan')
    analyse_usage(file_path, 'Results/usage_status.png', 'Status')
    # The usage curves for status 2 look very similar to those of leaving customers seen before!
    # Maybe the status alone is a good indicator of churn?
    # The numbers of combined variables are too small for histograms though.

    # Interesting questions are:
    # - Which percentage left of those in status 2? And of the people in tariff 1, and of those who complained?
    print_crosstab(file_path)
    # We find that 47% of all Status 2 customers left, but only 5% of the Status 1 customers! 74% percent of leaving
    # customers had status 2, but only 16% of those who stayed.

    # Of the customers who complained, even 83% left, while only 10% of the customers who didn't complain left.
    # Nevertheless, customers who complain only make up 40% of the churn.

    # Also, 17% of the Tariff Plan 1 customers left, and only 2% of Plan 2 customers.
    # Leaving customers where almost all (99%) in Tariff 1.

    # Train models to predict if a customer will leave:
    ml_analysis()
    # Resulting accuracies:
    # Decision tree:    90.3%
    # Random forest:    91.4%
    # Neural network:   92.1%
