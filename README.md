# 🚗 Kean University Data Analytics Capstone: Automobile Pricing  

![image](https://github.com/user-attachments/assets/2191e3bf-66c4-4a1f-8cb4-3e1d87d4e095)

## 📌 Introduction  
The automobile market is a complex ecosystem where car prices are influenced by a multitude of factors, including vehicle age, style, engine specifications, and fuel efficiency. Understanding these variables and their impact on pricing is essential for both consumers and manufacturers.  

This research focuses on a comprehensive dataset comprising features such as vehicle age, style, engine horsepower, and fuel efficiency. The objective is to predict the Manufacturer's Suggested Retail Price (MSRP) using these attributes. By analyzing these characteristics, this project develops a predictive model to reliably estimate MSRP and uncover the most significant factors affecting automobile pricing.  

---

## 🔍 Background  
Estimating a vehicle’s MSRP is challenging due to the interplay of numerous variables:  

- **Vehicle Age**  
- **Engine Specifications**  
- **Style and Drive Modes**  

This research employs a **Random Forest Regressor** model to estimate MSRP, supported by Exploratory Data Analysis (EDA) to uncover critical relationships. The insights are valuable for consumers, manufacturers, and dealerships seeking to understand automotive pricing dynamics.  

---

## 🎯 Objectives  
- Analyze how various vehicle factors affect MSRP.  
- Identify variations in MSRP across vehicle styles and drive modes.  
- Develop the **best predictive model** for MSRP estimation using machine learning techniques.  

---

## 📝 Executive Summary  

This project investigates automobile pricing using a robust machine learning pipeline:  

### Methodologies  
1. **Data Collection**: APIs and web scraping.  
2. **Data Wrangling**: Transformation of categorical and numerical data.  
3. **Exploratory Data Analysis (EDA)**: Feature analysis and correlation discovery.  
4. **Predictive Modeling**:  
   - Linear Regression  
   - Random Forest Regressor  

### 🔑 Key Findings  
- **Engine horsepower, vehicle style, and vehicle age** are the most critical determinants of MSRP.  
- A **logarithmic transformation** of MSRP improved prediction accuracy.  
- The **Random Forest Regressor** achieved an **R-squared value of 0.9914**, outperforming Linear Regression.  

---

## ⚙️ Methodology  

### 1. Data Collection  
- **API Requests**: Vehicle data retrieved and structured for analysis.  
- **Web Scraping**: Extracted additional attributes (e.g., styles and specifications) using `BeautifulSoup`.  

### 2. Data Preprocessing  
- Missing numerical values were **imputed** with the mean.  
- Categorical variables were **one-hot encoded**.  
- **Log transformation** addressed skewness in MSRP distribution.  

### 3. Exploratory Data Analysis (EDA)  
- **Scatterplots**: Engine horsepower vs. MSRP.  
- **Bar Charts**: Pricing differences across vehicle styles.  
- **Histograms**: Before and after MSRP log transformation.  

### 4. Model Training  
- Split dataset: **80% training** | **20% test**.  
- **Hyperparameter tuning**: Bayesian Optimization (XGBoost).  
- Metrics evaluated:  
   - Mean Squared Error (MSE)  
   - R-squared  
   - Adjusted R-squared  

---

## 📊 Results  

### 🔎 Exploratory Data Analysis  
- **Engine Horsepower** and **Car Age** are the most influential predictors.  
- Vehicle styles and drive modes showed distinct pricing trends.  

### 🏆 Model Performance  
| Model                  | R-squared | Observations                                   |  
|------------------------|-----------|-----------------------------------------------|  
| **Linear Regression**  | 0.8533    | Struggled with nonlinear relationships.       |  
| **Random Forest**      | 0.9914    | Captured complex feature interactions.        |  

---

## 📈 Visualization & Analytics  

1. **Scatterplot**: Engine horsepower vs. MSRP  
2. **Bar Chart**: Pricing differences by vehicle style  
3. **Histogram**: MSRP distribution (before and after log transformation)  

---

## 🤖 Predictive Analytics  
- **Random Forest Regressor** demonstrated superior accuracy.  
- **Log transformation** reduced skewness and improved predictions.  

---

## 🏁 Conclusion  

### 🚀 Model Performance  
- **Random Forest Regressor** achieved an **R-squared of 0.9914**, effectively modeling nonlinear relationships.  

### 🔑 Feature Importance  
- Engine horsepower  
- Vehicle style  
- Car age  

### 📌 Additional Insights  
- Logarithmic transformation improved model accuracy.  
- Pricing strategies can be refined using insights from EDA.  

---

## 🔮 Future Work  
- Implement additional models like **XGBoost** for comparison.  
- Expand the dataset for greater generalizability.  
- Explore advanced feature selection techniques such as **PCA** (Principal Component Analysis).  

---

## 🛠️ Tools & Libraries  
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost  

---

## 📂 Project Structure  


---

### 👨‍💻 Authors 

Jared Sheridan 
LinkedIn: https://www.linkedin.com/in/jared-sheridan-a20364226/


Evan Kavanagh
LinkedIn: https://www.linkedin.com/in/evan-kavanagh-b81b8324a/


---
