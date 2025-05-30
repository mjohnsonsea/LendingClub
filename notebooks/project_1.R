# OPAN 6604 - Project 3 ####

# SAXA 3 #

# Emma Cranmer | Mike Johnson | Dylan Lowndes | Izzy Mendoza | Lola Oshodi #

## Set up ####

# Libraries
library(tidyverse)
library(caret)
library(fastDummies)
library(cluster) # standardizing variables
library(factoextra) # For PCA helper functions
library(corrplot) #For some correlation and PCA plots
library(ranger)
library(Metrics)


# Set random seed for reproducibility
set.seed(206)

# Set viz theme
theme_set(theme_classic())

# Load Data
train_raw = read.csv('data/raw/LC_train.csv')
test = read.csv('data/raw/LC_test.csv')

## Step 1: Create a train/test split ####
# Train/test split already performed. However, let's create a dev set for validation.
train_indices = createDataPartition(train_raw$int_rate, p = 0.9, list = FALSE)
train = train_raw[train_indices, ]
dev = train_raw[-train_indices, ]

## Step 2: Data Prep ####

# Initial view
str(train)


# Drop "current" columns. These features were not present during the loan origination
train = train %>% 
  select(-loan_status,
         -revol_bal,
         -revol_util,
         -tot_cur_bal,
         -total_acc)

dev = dev %>% 
  select(-loan_status,
         -revol_bal,
         -revol_util,
         -tot_cur_bal,
         -total_acc)

test = test %>% 
  select(-loan_status,
         -revol_bal,
         -revol_util,
         -tot_cur_bal,
         -total_acc)

# Drop purpose. Title is the same thing
train = train %>% 
  select(-purpose)

dev = dev %>% 
  select(-purpose)

test = test %>% 
  select(-purpose)

# Drop id from test set
test = test %>% 
  select(-ID)

### NA Handling ####

# Categorical columns are will get missed in NA count since they are "". Need to replace these so they get counted.
train = train %>% 
  mutate(across(where(is.character), ~na_if(., "")))

dev = dev %>% 
  mutate(across(where(is.character), ~na_if(., "")))

test= test %>% 
  mutate(across(where(is.character), ~na_if(., "")))

# Check for NA's
train %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>% 
  pivot_longer(
    cols = everything(),
    names_to = "column_name",
    values_to = "na_count"
  ) %>% 
  arrange(desc(na_count)) %>% 
  mutate("na_percent" = round(na_count / nrow(train),2) * 100) %>% 
  filter(na_count > 0)

#### mths_since_last_record ####

# 91% of records are blank. Drop this feature.
train = train %>% 
  select(-mths_since_last_record)

dev = dev %>% 
  select(-mths_since_last_record)

test = test %>% 
  select(-mths_since_last_record)

#### emp_title ####

# 15% are missing. Need to understand what we're working with here.
n_distinct(train['emp_title'])

# There are 32K distinct values here. A lot of noise... Let's drop it.
train = train %>% 
  select(-emp_title)

dev = dev %>% 
  select(-emp_title)

test = test %>% 
  select(-emp_title)

#### mths_since_recent_inq ####

# Months since most recent inquiry... Could be a case that there a no inquiries. Impute -1 for these cases.
train = train %>% 
  mutate(mths_since_recent_inq = ifelse(is.na(mths_since_recent_inq), 
                                 -1,
                                 mths_since_recent_inq))

dev = dev %>% 
  mutate(mths_since_recent_inq = ifelse(is.na(mths_since_recent_inq), 
                                        -1,
                                        mths_since_recent_inq))

test = test %>% 
  mutate(mths_since_recent_inq = ifelse(is.na(mths_since_recent_inq), 
                                        -1,
                                        mths_since_recent_inq))

#### emp_length ####

train %>% 
  group_by(emp_length) %>% 
  summarise(cnt = n())

# Likely a case that NA's represent the unemployed. Let's impute < 1 year
train = train %>% 
  mutate(emp_length = ifelse(is.na(emp_length), '< 1 year', emp_length))

dev = dev %>% 
  mutate(emp_length = ifelse(is.na(emp_length), '< 1 year', emp_length))

test = test %>% 
  mutate(emp_length = ifelse(is.na(emp_length), '< 1 year', emp_length))

#### mo_sin_old_il_acct ####

# Months since oldest bank installment account opened... Could be a case that there are no accounts. Impute -1 for these cases.
train = train %>% 
  mutate(mo_sin_old_il_acct = ifelse(is.na(mo_sin_old_il_acct), 
                                        -1,
                                        mo_sin_old_il_acct))

dev = dev %>% 
  mutate(mo_sin_old_il_acct = ifelse(is.na(mo_sin_old_il_acct), 
                                     -1,
                                     mo_sin_old_il_acct))

test = test %>% 
  mutate(mo_sin_old_il_acct = ifelse(is.na(mo_sin_old_il_acct), 
                                     -1,
                                     mo_sin_old_il_acct))

#### mths_since_rcnt_il ####

# Months since most recent installment accounts opened... Could be a case that there are no accounts. Impute -1 for these cases.
train = train %>% 
  mutate(mths_since_rcnt_il = ifelse(is.na(mths_since_rcnt_il), 
                                     -1,
                                     mths_since_rcnt_il))

dev = dev %>% 
  mutate(mths_since_rcnt_il = ifelse(is.na(mths_since_rcnt_il), 
                                     -1,
                                     mths_since_rcnt_il))

test = test %>% 
  mutate(mths_since_rcnt_il = ifelse(is.na(mths_since_rcnt_il), 
                                     -1,
                                     mths_since_rcnt_il))

#### mths_since_recent_bc ####

# Months since most recent bankcard account opened... Could be a case that there are no accounts. Impute -1 for these cases.
train = train %>% 
  mutate(mths_since_recent_bc = ifelse(is.na(mths_since_recent_bc), 
                                     -1,
                                     mths_since_recent_bc))

dev = dev %>% 
  mutate(mths_since_recent_bc = ifelse(is.na(mths_since_recent_bc), 
                                       -1,
                                       mths_since_recent_bc))

test = test %>% 
  mutate(mths_since_recent_bc = ifelse(is.na(mths_since_recent_bc), 
                                       -1,
                                       mths_since_recent_bc))

#### dti ####

# These are individuals with no income. Impute 0.
train = train %>% 
  mutate(dti = ifelse(is.na(dti), 0, dti))

dev = dev %>% 
  mutate(dti = ifelse(is.na(dti), 0, dti))

test = test %>% 
  mutate(dti = ifelse(is.na(dti), 0, dti))

#### all_util ####

# These are individuals with no collection amounts owed. Impute 0.
train = train %>% 
  mutate(all_util = ifelse(is.na(all_util), 0, all_util))

dev = dev %>% 
  mutate(all_util = ifelse(is.na(all_util), 0, all_util))

test = test %>% 
  mutate(all_util = ifelse(is.na(all_util), 0, all_util))



### Other Stuff ####

#### emp_length ####

# Convert to numerical value
train = train %>% 
  mutate(emp_length = case_when(emp_length == '< 1 year' ~ 0,
                                emp_length == '10+ years' ~ 10,
                                .default = as.numeric(str_extract(emp_length, "^\\d+"))))

dev = dev %>% 
  mutate(emp_length = case_when(emp_length == '< 1 year' ~ 0,
                                emp_length == '10+ years' ~ 10,
                                .default = as.numeric(str_extract(emp_length, "^\\d+"))))

test = test %>% 
  mutate(emp_length = case_when(emp_length == '< 1 year' ~ 0,
                                emp_length == '10+ years' ~ 10,
                                .default = as.numeric(str_extract(emp_length, "^\\d+"))))

#### fico_range ####

# There is a low score and high score. Consider an average of the two. Let's explore the range. 
train %>% 
  mutate(fico_spread = fico_range_high - fico_range_low) %>% 
  ggplot(aes(x = fico_spread)) +
  geom_histogram(binwidth = 1)

# Range is 4 is all records. Opting for 1 column.
train = train %>% 
  select(-fico_range_high)

dev = dev %>% 
  select(-fico_range_high)

test = test %>% 
  select(-fico_range_high)

#### Categorical ####

# Convert to factors
train = train %>% mutate(across(where(is.character), as.factor))
dev = dev %>% mutate(across(where(is.character), as.factor))
test = test %>% mutate(across(where(is.character), as.factor))

## Step 3: Data Exploration ####

### Interest Rate Distribution ####

# Overall distribution
train %>% 
  ggplot(aes(x = int_rate)) + 
  geom_density()

# What does it look like with application_type? Joint Apps appear higher
train %>% 
  ggplot(aes(x = int_rate, color = application_type)) + 
  geom_density()

train %>% 
  group_by(application_type) %>% 
  summarise(avg_rate = mean(int_rate))

train %>% 
  ggplot(aes(x = int_rate, y = application_type)) + 
  geom_boxplot()

# What does it look like with home_ownership? Similar distribution.
train %>% 
  ggplot(aes(x = int_rate, color = home_ownership)) + 
  geom_density()

train %>% 
  group_by(home_ownership) %>% 
  summarise(avg_rate = mean(int_rate))

train %>% 
  ggplot(aes(x = int_rate, y = home_ownership)) + 
  geom_boxplot()

# What does it look like with term? 60 month tend to be higher.
train %>% 
  ggplot(aes(x = int_rate, color = term)) + 
  geom_density()

train %>% 
  group_by(term) %>% 
  summarise(avg_rate = mean(int_rate))

train %>% 
  ggplot(aes(x = int_rate, y = term)) + 
  geom_boxplot()

# What does it look like with title? Moving, Vacation, Medical tend to be higher.
train %>% 
  ggplot(aes(x = int_rate, color = title)) + 
  geom_density()

train %>% 
  group_by(title) %>% 
  summarise(avg_rate = mean(int_rate)) %>% 
  arrange(desc(avg_rate))

train %>% 
  ggplot(aes(x = int_rate, y = title)) + 
  geom_boxplot()

# What does it look like with Verification Status? Similar distribution.
train %>% 
  ggplot(aes(x = int_rate, color = verification_status)) + 
  geom_density()

train %>% 
  group_by(verification_status) %>% 
  summarise(avg_rate = mean(int_rate)) %>% 
  arrange(desc(avg_rate))

train %>% 
  ggplot(aes(x = int_rate, y = verification_status)) + 
  geom_boxplot()

### Correlation Analysis ####

correlation_matrix = train %>% 
  select(where(is.numeric)) %>% 
  cor( , use = "pairwise.complete.obs") %>% 
  round(2)

correlation = correlation_matrix %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "var_1") %>% 
  pivot_longer(-var_1, names_to = "var_2", values_to = "r") %>% 
  mutate(var1 = pmin(var_1, var_2), var2 = pmax(var_1, var_2)) %>% 
  select(-var_1, -var_2) %>% 
  distinct(var1, var2, .keep_all = TRUE) %>% 
  select(var1, var2, r) %>% 
  filter(var1 != var2) %>% 
  mutate(r = round(r, 2)) %>% 
  arrange(desc(r))

# Variables with moderate correlation
correlation %>% 
  filter(abs(r) >= 0.3)

# Interest rate is moderately correlated to all_util, fico_range_low

# Visualize relationship between all_util and int_rate

util_int = train %>% 
  group_by(all_util) %>% 
  summarise(avg_int = mean(int_rate),
            cnt = n())

# As utilization increases, interest rate increases. But average interest rate fluctuates > 100
util_int %>% 
  ggplot(aes(x = all_util, y = avg_int)) +
  geom_line() + 
  labs(title = "Balance to credit limit on all trades vs Avg Interest Rate") +
  xlab("Balance to credit limit on all trades") +
  ylab("Average Interest Rate")

# Long tail > 100. Consider excluding these points since they're creating fluctuations.
train %>% 
  ggplot(aes(x = all_util)) +
  geom_histogram(binwidth = 5)

# Visualize relationship between all_util and fico_range_high

fico_int = train %>% 
  group_by(fico_range_low) %>% 
  summarise(avg_int = mean(int_rate),
            cnt = n())

# As the fico score increases, interest rate decreases. Rebounds > 825?
fico_int %>% 
  ggplot(aes(x = fico_range_low, y = avg_int)) +
  geom_line() + 
  labs(title = "Fico Score vs Avg Interest Rate") +
  xlab("Fico Score") +
  ylab("Average Interest Rate")


# Not a lot of data after 800. Should consider excluding records greater than 800.
train %>% 
  ggplot(aes(x = fico_range_low)) + 
  geom_histogram(binwidth = 25)

train %>% 
  ggplot(aes(x = fico_range_low, y = int_rate)) + 
  geom_point()



## Step 3: Data Pre-Processing ####

### Location-Based Features ####
# Should we exclude these features? Ethical concerns. Possible trap that Zafari intentionally added.
# Ideally, interest rate should be determined by credit worthiness, rather than location. 
# Prediction based on location opens the door for discrimination.

train = train %>% select(-addr_state, -zip_code)
dev = dev %>% select(-addr_state, -zip_code)
test = test %>% select(-addr_state, -zip_code)

### Remover Outliers ####

# Remove outlier fico scores and utilization from training set
train = train %>% 
  filter(fico_range_low < 800) %>% 
  filter(all_util < 100)

### Standardization for PCA ####

standardize = preProcess(train %>% select(-int_rate), method = c("center", "scale"))

train_s = predict(standardize, train)
dev_s = predict(standardize, dev)
test_s = predict(standardize, test)

# Dummy Variables for PCA
train_s = train_s %>% dummy_cols(select_columns = c('application_type', 'home_ownership', 'term', 'title', 'verification_status'),
                                 remove_selected_columns = T,
                                 remove_first_dummy = T)

dev_s = dev_s %>% dummy_cols(select_columns = c('application_type', 'home_ownership', 'term', 'title', 'verification_status'),
                                 remove_selected_columns = T,
                                 remove_first_dummy = T)

test_s = test_s %>% dummy_cols(select_columns = c('application_type', 'home_ownership', 'term', 'title', 'verification_status'),
                             remove_selected_columns = T,
                             remove_first_dummy = T)


## Step 4: Feature Engineering ####

### PCA ####

# Run PCA without target variable
pca = prcomp(train_s %>% select(-int_rate), scale. = TRUE)
get_eig(pca) # Eigenvalues

#### Variable Analysis ####
get_eig(pca) %>% round(2) # Up to 28 captures 90% of variance

fviz_eig(pca, addlabels = TRUE)

#Extract all the results (coordinates, squared cosine, contributions) for variables from PCA outputs
pca_by_var = get_pca_var(pca)

# Visualize variables mapped over the first two components
fviz_pca_var(pca,
             repel = TRUE)

# Quality of representation
corrplot(pca_by_var$cos2,
         is.corr = FALSE)

# Contribution of variables to PC:
corrplot(pca_by_var$contrib,
         is.corr = FALSE)

# Bar plots for variable contributions
fviz_contrib(pca, choice = "var", axes = 1)
fviz_contrib(pca, choice = "var", axes = 2)
fviz_contrib(pca, choice = "var", axes = 3)
fviz_contrib(pca, choice = "var", axes = 4)
fviz_contrib(pca, choice = "var", axes = 5)
fviz_contrib(pca, choice = "var", axes = 6)
fviz_contrib(pca, choice = "var", axes = 7)
fviz_contrib(pca, choice = "var", axes = 8)
fviz_contrib(pca, choice = "var", axes = 9)
fviz_contrib(pca, choice = "var", axes = 10)
fviz_contrib(pca, choice = "var", axes = 11)
fviz_contrib(pca, choice = "var", axes = 12)
fviz_contrib(pca, choice = "var", axes = 13)
fviz_contrib(pca, choice = "var", axes = 14)
fviz_contrib(pca, choice = "var", axes = 15)
fviz_contrib(pca, choice = "var", axes = 16)
fviz_contrib(pca, choice = "var", axes = 17)
fviz_contrib(pca, choice = "var", axes = 18)
fviz_contrib(pca, choice = "var", axes = 19)
fviz_contrib(pca, choice = "var", axes = 20)
fviz_contrib(pca, choice = "var", axes = 21)
fviz_contrib(pca, choice = "var", axes = 22)
fviz_contrib(pca, choice = "var", axes = 23)
fviz_contrib(pca, choice = "var", axes = 24)
fviz_contrib(pca, choice = "var", axes = 25)
fviz_contrib(pca, choice = "var", axes = 26)
fviz_contrib(pca, choice = "var", axes = 27)
fviz_contrib(pca, choice = "var", axes = 28)

#A circular plot for contribution values
fviz_pca_var(pca, col.var = "contrib", repel = TRUE)

# Extract PCA results for individuals
ind = get_pca_ind(pca)

fviz_contrib(pca, choice = "ind", axes = 2, top = 10)

#### Apply PCAs to train, dev, and test sets ####

# Predictions
train_pca_preds = predict(pca, newdata = train_s %>% select(-int_rate))
dev_pca_preds = predict(pca, newdata = dev_s %>% select(-int_rate))
test_pca_preds = predict(pca, newdata = test_s)

# Include only first 28 dimensions and make a dataframe. Bind with int_rate
train_pca = cbind(as.data.frame(train_pca_preds[, 1:28]), train_s %>% select(int_rate))
dev_pca = cbind(as.data.frame(dev_pca_preds[, 1:28]), dev_s %>% select(int_rate))
test_pca = as.data.frame(test_pca_preds[, 1:28])

## Step 5: Model Selection ####

# Define training control for cross-validation
train_control = trainControl(method = "cv",
                             number = 5)

### Baseline: Linear Regression ####

# Train linear regression model
lm_model = train(int_rate ~ .,
                 data = train,
                 method = "lm",
                 trControl = train_control,
                 metric = "RMSE")

summary(lm_model$finalModel)
plot(lm_model$finalModel)

lm_model$results
lm_model$resample

# Test against dev set
dev$int_lm = predict(lm_model, newdata = dev)

# Calculate performance metrics on dev set
mae_lm = mae(dev$int_rate, dev$int_lm)
rmse_lm = rmse(dev$int_rate, dev$int_lm)
r2_lm = cor(dev$int_rate, dev$int_lm)^2

# Alt model

# This one drops some of the variables that are correlated with each other.
# Performance doesn't improve in this case.

lm_model2 = train(int_rate ~ .,
                 data = train %>% 
                   select(-all_util, # Correlated with fico. Drop.
                          -inq_last_12m, # Correlated with inc_fi. Drop.
                          -total_bal_ex_mort, # Correlated with several variables. Drop.
                          - mths_since_recent_inq, # Correlated with inq_last_12m. Drop
                          -pub_rec_bankruptcies, # Correlated with pub_rec. Drop
                          ),
                 method = "lm",
                 trControl = train_control,
                 metric = "RMSE")

summary(lm_model2$finalModel)
plot(lm_model2$finalModel)

lm_model2$results
lm_model2$resample

# Test against dev set
dev$int_lm2 = predict(lm_model2, newdata = dev)

# Calculate performance metrics on dev set
mae_lm2 = mae(dev$int_rate, dev$int_lm2)
rmse_lm2 = rmse(dev$int_rate, dev$int_lm2)
r2_lm2 = cor(dev$int_rate, dev$int_lm2)^2

# PCA LM Model

lm_pca = train(int_rate ~ .,
               data = train_pca,
               method = "lm",
               trControl = train_control,
               metric = "RMSE")

summary(lm_pca)
lm_pca$results

plot(lm_pca$finalModel)

# Test against dev set
dev$int_lmpca = predict(lm_pca, newdata = dev_pca)

# Calculate performance metrics on dev set
mae_lmpca = mae(dev$int_rate, dev$int_lmpca)
rmse_lmpca = rmse(dev$int_rate, dev$int_lmpca)
r2_lmpca = cor(dev$int_rate, dev$int_lmpca)^2

### Artificial Neural Network ####

#### Normalization ####
# Need to normalize data.
normalizer = preProcess(train_pca %>% select(-int_rate), method = "range")
normalizer_int = preProcess(train_pca %>% select(int_rate), method = "range")

train_ann = predict(normalizer, train_pca) # Apply to independent variables
train_ann = predict(normalizer_int, train_pca) # Apply to dependent variable
dev_ann = predict(normalizer, dev_pca) # Apply to independent variables
dev_ann = predict(normalizer_int, dev_pca) # Apply to dependent variable
test_ann = predict(normalizer, test_pca)

#### Train Model ####
nn_model = train(int_rate ~ .,
                 data = train_ann,
                 method = "nnet",
                 trControl = train_control,
                 linout = TRUE,
                 trace = FALSE,
                 tuneGrid = expand.grid(size = c(10),
                                         decay = c(0.0001)))

nn_model
plot(nn_model)
nn_model$bestTune
nn_model$finalModel
nn_model$results
varImp(nn_model)

# De-normalize RMSE
int_rate_min = min(train$int_rate)
int_rate_max = max(train$int_rate)
ann_rmse_denorm = nn_model$results$RMSE * (int_rate_max - int_rate_min) + int_rate_min

ann_rmse_denorm # ANN RMSE
lm_model$results$RMSE # Base Line RMSE

# Test against dev set
dev$int_ann = predict(nn_model, newdata = dev_ann)
dev$int_ann = dev$int_ann * (int_rate_max - int_rate_min) + int_rate_min

# Calculate performance metrics on dev set
mae_ann = mae(dev$int_rate, dev$int_ann)
rmse_ann = rmse(dev$int_rate, dev$int_ann)
r2_ann = cor(dev$int_rate, dev$int_ann)^2

rmse_ann

### Random Forest ####

#Train model using ranger
rf_model_fast = train(
  int_rate ~ .,
  data = train,
  method = "ranger",                 
  trControl = trainControl(method = "cv", number = 3),
  metric = "RMSE",
  tuneLength = 3,
  num.trees = 100,
  importance = "permutation"
)          

rf_model_fast$results
varImp(rf_model_fast)

# Evaluate on dev set
dev$int_rf = predict(rf_model_fast, newdata = dev)

#calculate performance metrics on dev set
mae_rf = mae(dev$int_rate, dev$int_rf)
rmse_rf = rmse(dev$int_rate, dev$int_rf)
r2_rf = cor(dev$int_rate, dev$int_rf)^2


#Print Performance metrics
mae_val
rmse_val
r2_val

## Step 6: Model Evaluation

# Random Forest is the best performing model. Let's dive deeper into it's performance.
varImp(rf_model_fast) # Fico was the most imporant variable.

# Let's measure how the difference between actual and predicted to see how did in that dimension

dev %>% 
  mutate(dif = int_rate - int_ann) %>% 
  ggplot(aes(x = fico_range_low, y = dif, color = fico_range_low)) +
  geom_point() + 
  geom_hline(yintercept = 0, color = "red")

train %>% 
  mutate(
    fico_group = case_when(
      fico_range_low < 700 ~ "<700",
      fico_range_low >= 700 & fico_range_low <= 725 ~ "700-725",
      fico_range_low >= 726 & fico_range_low <= 750 ~ "726-750",
      fico_range_low >= 751 & fico_range_low <= 775 ~ "751-775",
      fico_range_low >= 776 & fico_range_low <= 800 ~ "776-800",
      fico_range_low > 800 ~ ">800"
    )
  ) %>% 
  ggplot(aes(x = int_rate, y = fico_group)) +
  geom_boxplot() 

