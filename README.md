# Kean-University-Data-Analytics-Capstone-Automobile-Pricing

Introduction
The automobile market is a complex ecosystem where car prices are influenced by a multitude of factors, including vehicle age, style, engine specifications, and fuel efficiency. Understanding these variables and their impact on pricing is essential for both consumers and manufacturers. With the increase of available vehicle data, it is now feasible to leverage data-driven methods to forecast automobile pricing with greater accuracy.
This research focuses on a comprehensive dataset comprising features such as vehicle age, style, engine horsepower, and fuel efficiency, aiming to predict the Manufacturer's Suggested Retail Price (MSRP) using these attributes. By analyzing these characteristics, this project seeks to develop a predictive model that can reliably estimate MSRP and uncover the most significant factors affecting pricing in the automobile industry.

Background
Estimating a vehicle’s MSRP is challenging due to the interplay of numerous variables. Characteristics such as the car’s age, engine specifications, and style all contribute to its pricing. Analyzing the importance of these factors enables the development of a predictive model, offering valuable insights for consumers, manufacturers, and dealerships alike.
The primary objective of this project is to develop a predictive model using a Random Forest Regressor to estimate MSRP based on diverse vehicle characteristics. Exploratory Data Analysis (EDA) is utilized to uncover critical relationships between features and their influence on MSRP. This research aims to deepen our understanding of automotive pricing dynamics while providing a reliable tool for estimating car costs based on specific features.

Explore
How various vehicle factors affect MSRP.
Variations in MSRP across vehicle styles and drive modes.
Best predictive model for MSRP estimation using machine learning techniques.

Executive Summary
The research investigates factors influencing automobile pricing and attempts to accurately predict MSRP. To achieve this, the following methodologies were employed:
Data Collection: Aggregating vehicle data through APIs and web scraping.
Data Wrangling: Transforming categorical variables and scaling numerical features.
Exploratory Data Analysis (EDA): Identifying correlations between features like engine horsepower, car age, and MSRP.
Predictive Modeling: Building machine learning models such as Linear Regression and Random Forest Regressor.
Key Findings:
Engine horsepower, vehicle style, and vehicle age are critical determinants of MSRP.
A logarithmic transformation of MSRP improved prediction accuracy.
The Random Forest Regressor outperformed Linear Regression, achieving an R-squared value of 0.9914.

Methodology
Data Collection
API Requests: Vehicle data retrieved via API and converted into structured formats for analysis.
Web Scraping: Used BeautifulSoup to extract additional features like vehicle styles and specifications from online sources.
Data Preprocessing
Missing numerical values imputed with the mean.
Categorical variables encoded using one-hot encoding.
Log transformation applied to MSRP to address skewness.
Exploratory Data Analysis (EDA)
Scatterplots revealed correlations between engine horsepower and MSRP.
Bar charts highlighted pricing differences across vehicle styles.
Histograms of MSRP supported the need for transformation.
Model Training
Dataset split into training (80%) and test (20%) sets.
Hyperparameter tuning using Bayesian Optimization for XGBoost.
Performance metrics evaluated: Mean Squared Error (MSE), R-squared, and Adjusted R-squared.

Results
Exploratory Data Analysis
Engine horsepower and car age emerged as the most influential predictors of MSRP.
Vehicle styles and drive modes showed distinct pricing trends.
Model Performance
Linear Regression: R-squared = 0.8533, struggled with nonlinear relationships.
Random Forest Regressor: R-squared = 0.9914, effectively captured feature interactions.
Key Insights
Higher engine horsepower strongly correlates with increased MSRP.
Luxury vehicle styles command significantly higher pricing.
Log transformation of MSRP enhanced model performance.

Visualization / Analytics
Scatterplot: Engine horsepower vs. MSRP.
Bar Chart: Pricing differences by vehicle style.
Histogram: MSRP distribution before and after log transformation.

Predictive Analytics
Random Forest Regressor demonstrated superior accuracy, capturing complex relationships between features.
Log transformation proved critical in improving predictions by minimizing the influence of extreme values.

Conclusion
Model Performance
Random Forest Regressor performed best with an R-squared of 0.9914, showcasing its ability to capture nonlinear interactions.
Feature Significance
Engine horsepower, car age, and vehicle style emerged as the most critical predictors of MSRP.
Additional Observations
Pricing strategies can be refined by leveraging insights from EDA.
Logarithmic transformation addressed skewness effectively.
Future Work
Employ additional models like XGBoost for comparison.
Expand the dataset for greater generalizability.
Explore advanced feature selection methods, such as PCA, to improve predictive accuracy further.

