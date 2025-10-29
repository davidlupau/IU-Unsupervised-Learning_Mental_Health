# Mental Health in Tech: Unsupervised Learning Analysis

This project analyzes mental health survey data from the technology sector using unsupervised learning techniques to identify patterns and help HR departments implement targeted mental health support programs. The analysis includes data preprocessing, feature engineering, and k-means clustering.

## Dataset

The analysis uses the OSMI Mental Health in Tech Survey 2016, available from:
https://www.kaggle.com/osmi/mental-health-in-tech-2016

## How to Use the Code

Run the following Python files in order:

1. `exploration_dataset.py` - Initial data exploration and basic statistics
2. `dataset_prep_ML_analysis.py` - Data cleaning and standardization
3. `ML_analysis.py` - Machine learning analysis including PCA and k-means clustering

## Tools and Dependencies

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yellowbrick

### Development Environment
- Python 3.x
- PyCharm IDE
- Git/GitHub for version control

## Project Structure
```
├── exploration_dataset.py        # Initial data exploration
├── dataset_prep_ML_analysis.py   # Data preprocessing
├── ML_analysis.py                # Machine learning analysis
├── functions.py                  # Helper functions
├── README.md
└── LICENSE.md
```

## Results

The analysis identifies three distinct clusters of employees with varying mental health experiences and needs:
- Self-Employed Remote Workers Group (17.6%)
- High Stigma Environment Group (40.9%)
- Supportive Environment Group (41.5%)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OSMI for providing the mental health in tech survey data
- IU International University for project supervision