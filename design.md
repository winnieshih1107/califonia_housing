prompt_v1:
  title: "L7-Multiple Linear Regression Workflow"
  version: "v1"
  role: >
    You are a professional data scientist, machine learning instructor,
    and multidisciplinary business analysis team.

  objective: >
    Write a complete Scikit-learn solution for the Boston Housing dataset.
    The solution must strictly follow the CRISP-DM process and include clean
    Python code, expert-level feature analysis, model comparison, cross-validation,
    and careful business interpretation.

  dataset:
    name: "Boston Housing"
    file_name: "boston_housing.csv"
    target_column: "medv"
    features:
      - "crim"
      - "zn"
      - "indus"
      - "chas"
      - "nox"
      - "rm"
      - "age"
      - "dis"
      - "rad"
      - "tax"
      - "ptratio"
      - "b"
      - "lstat"

  problem_type:
    learning_type: "Supervised Learning"
    task_type: "Regression"
    target: "Predict median neighborhood housing values (medv)"

  expert_panel:
    rm_expert:
      focus:
        - "Structural capacity"
        - "Average rooms"
        - "Space requirements"
      conclusion: >
        Average number of rooms per dwelling (rm) is a primary driver of home values,
        representing the physical size and functionality of the building.

    lstat_expert:
      focus:
        - "Socio-economic profile"
        - "Lower status percentage"
        - "Neighborhood wealth"
      conclusion: >
        The proportion of lower status population (lstat) captures neighborhood wealth
        and purchasing power. It is heavily negatively correlated with housing prices.

    ptratio_expert:
      focus:
        - "Education quality"
        - "School crowding"
        - "Pupil-teacher ratios"
      conclusion: >
        Local school quality (ptratio) is highly valued by families. Crowded schools
        (high ptratio) depress average property values.

    environmental_policy_expert:
      focus:
        - "Nitric oxides concentration"
        - "Charles River proximity"
        - "Environmental attributes"
      conclusion: >
        Proximity to the Charles River (chas) offers scenic premiums, while air pollution
        (nox) represents health costs that lower demand and prices.

    machine_learning_expert:
      focus:
        - "Multicollinearity"
        - "Standardization"
        - "Feature selection"
        - "Model generalization"
      conclusion: >
        With 506 samples, we can train all features. However, features have different scales
        (e.g., crime rate vs property tax rate). Standardization is critical to ensure stable
        coefficient comparison and prevent scaling bias.

  crisp_dm_steps:
    step_1_business_understanding:
      requirements:
        - "Explain the housing valuation problem."
        - "Explain why predicting medv is useful for buyers and urban planning."
        - "Explain how the model helps make informed real estate decisions."
        - "State that this is a supervised regression problem."

    step_2_data_understanding:
      requirements:
        - "Load the dataset and show shape, head, info, duplicates, and stats."
        - "Check Charles River dummy frequency."
        - "Show target correlations and groupby statistics for chas."
        - "Discuss feature relevance with expert opinions."

    step_3_data_preparation:
      requirements:
        - "Separate features and target."
        - "Scale numerical columns using ColumnTransformer and StandardScaler."
        - "Perform train-test split 80/20 with random_state=42."

    step_4_modeling:
      algorithm: "LinearRegression"
      model_experiments:
        model_1: "rm Only"
        model_2: "rm + lstat"
        model_3: "rm + lstat + ptratio"
        model_4: "All 13 Features"

    step_5_evaluation:
      requirements:
        - "Compare metrics across splits and 5-Fold cross-validation."
        - "Provide model selection justification."
        - "Explain coefficients and baseline values."

    step_6_deployment:
      requirements:
        - "Simulate new property valuation."
        - "Save pipeline as boston_housing_model.pkl."