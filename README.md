# 💻 Laptop Pricing Analysis & Machine Learning Project

A comprehensive data analysis and machine learning project focused on predicting laptop prices using various features and advanced regression techniques.

## 📊 Project Overview

This project analyzes laptop pricing data to build predictive models that can estimate laptop prices based on hardware specifications and other features. The analysis includes data preprocessing, feature engineering, model development, and performance evaluation using various machine learning techniques.

## 🗂️ Project Structure

```
CompValue/
├── LaptopPricing-Dataset-3.csv     # Dataset with laptop specifications and prices
├── LaptopPricing-ML.ipynb          # Main Jupyter notebook with analysis and ML models
├── README.md                       # Project documentation
└── src/
    ├── R2_1.png                    # R² score visualization 1
    └── R2_2.png                    # R² score visualization 2
```

## 📋 Dataset Description

The dataset contains information about various laptops with the following features:

### Features:
- **Manufacturer**: Brand of the laptop (Acer, Dell, HP, etc.)
- **Category**: Type/category of laptop
- **GPU**: Graphics card specifications
- **OS**: Operating system
- **CPU_core**: Number of CPU cores
- **Screen_Size_inch**: Display size in inches
- **CPU_frequency**: Processor frequency (normalized)
- **RAM_GB**: RAM capacity in gigabytes
- **Storage_GB_SSD**: SSD storage capacity
- **Weight_pounds**: Weight of the laptop
- **Screen-Full_HD**: Full HD display indicator
- **Screen-IPS_panel**: IPS panel indicator

### Target Variable:
- **Price**: Laptop price (target for prediction)

## 🛠️ Tools and Libraries Used

### Core Data Science Libraries:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

### Machine Learning:
- **scikit-learn** - Machine learning library
  - `LinearRegression` - Linear regression modeling
  - `Ridge` - Ridge regression for regularization
  - `PolynomialFeatures` - Polynomial feature generation
  - `train_test_split` - Data splitting
  - `cross_val_score` - Cross-validation
  - `GridSearchCV` - Hyperparameter tuning

### Additional Tools:
- **tqdm** - Progress bars for loops
- **Jupyter Notebook** - Interactive development environment

## 🔬 Analysis Tasks

### Task 1: Cross Validation Model Improvement
- Split dataset into training (90%) and testing (10%) subsets
- Created single variable linear regression using CPU frequency
- Implemented 4-fold cross-validation
- Evaluated model performance using R² scores

### Task 2: Overfitting Analysis
- Re-split data with 50% for testing
- Created polynomial features (degrees 1-5) using CPU frequency
- Analyzed overfitting by plotting R² scores vs polynomial order
- Identified optimal polynomial degree before performance degradation

### Task 3: Ridge Regression Implementation
- Used multiple features: CPU_frequency, RAM_GB, Storage_GB_SSD, CPU_core, OS, GPU, Category
- Generated polynomial features with degree=2
- Implemented Ridge regression with alpha values from 0.001 to 1
- Compared training vs testing R² scores across different alpha values

### Task 4: Grid Search Optimization
- Applied GridSearchCV for hyperparameter tuning
- Tested alpha values: [0.0001, 0.001, 0.01, 0.1, 1, 10]
- Used 4-fold cross-validation
- Identified optimal alpha value for best model performance

## 📈 Key Findings

1. **Single Feature Performance**: CPU frequency alone provides limited predictive power
2. **Cross-Validation**: 4-fold CV helps provide more robust performance estimates
3. **Overfitting Detection**: Higher polynomial degrees lead to overfitting
4. **Ridge Regression**: Regularization improves model generalization
5. **Hyperparameter Tuning**: Grid search identifies optimal regularization strength

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tqdm jupyter
```

### Running the Analysis
1. Clone or download the project files
2. Ensure the dataset `LaptopPricing-Dataset-3.csv` is in the project directory
3. Open `LaptopPricing-ML.ipynb` in Jupyter Notebook
4. Run all cells sequentially to reproduce the analysis

## 📊 Model Performance

The project evaluates multiple models:
- **Linear Regression**: Baseline model using single feature
- **Polynomial Regression**: Enhanced feature space with polynomial terms
- **Ridge Regression**: Regularized model preventing overfitting

Performance metrics tracked:
- R² Score (Coefficient of Determination)
- Training vs Testing performance comparison
- Cross-validation scores with standard deviation

## 🔍 Visualizations

The project includes several visualizations:
- R² scores vs polynomial order (overfitting analysis)
- Training vs validation performance across alpha values
- Model performance comparisons

## 📝 Future Improvements

- Feature selection and engineering
- Additional regression algorithms (Random Forest, XGBoost)
- Hyperparameter optimization for other models
- Feature importance analysis
- Price prediction confidence intervals

## 👥 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

---

*This project demonstrates practical machine learning techniques for regression problems, including cross-validation, regularization, and hyperparameter tuning.*