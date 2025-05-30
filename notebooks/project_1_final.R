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

## Step 1: Pre-Process ####
# Train/test split already performed. However, let's create a dev set for validation.
train_indices = createDataPartition(train_raw$int_rate, p = 0.9, list = FALSE)
train = train_raw[train_indices, ]
dev = train_raw[-train_indices, ]

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

## Step 2: Data Exploration ####

### Interest Rate Distribution ####

# Overall distribution
train %>% 
  ggplot(aes(x = int_rate)) + 
  geom_density() + 
  labs(title = "Interest Rate Distribution") + 
  xlab("Interest Rate") +
  ylab("Density")

# What does it look like with application_type? Joint Apps appear higher
train %>% 
  ggplot(aes(x = int_rate, color = application_type)) + 
  geom_density() +
  labs(title = "Interest Rate Distribution by Application Type") + 
  xlab("Interest Rate") +
  ylab("Density")

train %>% 
  group_by(application_type) %>% 
  summarise(avg_rate = mean(int_rate))

train %>% 
  ggplot(aes(x = int_rate, y = application_type)) + 
  geom_boxplot() +
  labs(title = "Interest Rate Distribution by Application Type") + 
  xlab("Interest Rate") +
  ylab("Type")

# What does it look like with home_ownership? Similar distribution.
train %>% 
  ggplot(aes(x = int_rate, color = home_ownership)) + 
  geom_density() +
  labs(title = "Interest Rate Distribution by Home Ownership") + 
  xlab("Interest Rate") +
  ylab("Density")

train %>% 
  group_by(home_ownership) %>% 
  summarise(avg_rate = mean(int_rate))

train %>% 
  ggplot(aes(x = int_rate, y = home_ownership)) + 
  geom_boxplot() +
  labs(title = "Interest Rate Distribution by Home Ownership") + 
  xlab("Interest Rate") +
  ylab("Home Ownership")

# What does it look like with term? 60 month tend to be higher.
train %>% 
  ggplot(aes(x = int_rate, color = term)) + 
  geom_density() +
  labs(title = "Interest Rate Distribution by Term") + 
  xlab("Interest Rate") +
  ylab("Density")

train %>% 
  group_by(term) %>% 
  summarise(avg_rate = mean(int_rate))

train %>% 
  ggplot(aes(x = int_rate, y = term)) + 
  geom_boxplot() +
  labs(title = "Interest Rate Distribution by Term") + 
  xlab("Interest Rate") +
  ylab("Term")

# What does it look like with title? Moving, Vacation, Medical tend to be higher.
train %>% 
  ggplot(aes(x = int_rate, color = title)) + 
  geom_density() +
  labs(title = "Interest Rate Distribution by Purpose") + 
  xlab("Interest Rate") +
  ylab("Density")

train %>% 
  group_by(title) %>% 
  summarise(avg_rate = mean(int_rate)) %>% 
  arrange(desc(avg_rate))

train %>% 
  ggplot(aes(x = int_rate, y = title)) + 
  geom_boxplot() +
  labs(title = "Interest Rate Distribution by Purpose") + 
  xlab("Interest Rate") +
  ylab("Purpose")

# What does it look like with Verification Status? Similar distribution.
train %>% 
  ggplot(aes(x = int_rate, color = verification_status)) + 
  geom_density() +
  labs(title = "Interest Rate Distribution by Verification Status") + 
  xlab("Interest Rate") +
  ylab("Density")

train %>% 
  group_by(verification_status) %>% 
  summarise(avg_rate = mean(int_rate)) %>% 
  arrange(desc(avg_rate))

train %>% 
  ggplot(aes(x = int_rate, y = verification_status)) + 
  geom_boxplot() +
  labs(title = "Interest Rate Distribution by Verfication Status") + 
  xlab("Interest Rate") +
  ylab("Verification Status")

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
  geom_histogram(binwidth = 5) +
  labs(title = "Balance to credit limit on all trades Distribution") + 
  xlab("Balance to credit limit on all trades") +
  ylab("Count")

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
  geom_histogram(binwidth = 25) + 
  labs(title = "Fico Score Distribution") +
  xlab("Fico Score") +
  ylab("Count")

train %>% 
  ggplot(aes(x = fico_range_low, y = int_rate)) + 
  geom_point()

### Annual Income ####

# Annual income has a huge range 0 - $8,645,185
summary(train$annual_inc)

# Very long tail. Consider excluding outliers
train %>% 
  ggplot(aes(x = annual_inc)) +
  geom_histogram() + 
  labs(title = "Annual Income Distribution") +
  xlab("Annual Income") +
  ylab("Count")

mean(train$annual_inc) - (3 * sd(train$annual_inc))

### Additional Pre-Processing ####

#### Location-Based Features ####
# Should we exclude these features? Ethical concerns.
# Ideally, interest rate should be determined by credit worthiness, rather than location. 
# Prediction based on location opens the door for discrimination.

train = train %>% select(-addr_state, -zip_code)
dev = dev %>% select(-addr_state, -zip_code)
test = test %>% select(-addr_state, -zip_code)

#### Remove Outliers ####

# Remove outlier fico scores and utilization from training set
train = train %>% 
  filter(fico_range_low < 800) %>% 
  filter(all_util < 100)

# Remove annual income outliers. Assume we want to capture 95% data.
train = train %>% 
  filter(annual_inc <= quantile(annual_inc, probs = .95))


## Step 3: Train Models ####

# Define training control for cross-validation
train_control = trainControl(method = "cv",
                             number = 5)

### Linear Regression ####

# Train linear regression model
# Exclude variables with evidence of multi-collinearity
lm_model = train(int_rate ~ .,
                 data = train %>% select(-pub_rec_bankruptcies, -inq_last_12m, -total_bal_ex_mort, -all_util, -inq_last_12m),
                 method = "lmStepAIC", # stepwise selection
                 trControl = train_control,
                 metric = "RMSE",
                 direction = "backward")

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

### Random Forest ####

#Train model using ranger
rf_model_fast = train(
  int_rate ~ .,
  data = train,
  method = "ranger",                 
  trControl = train_control,
  metric = "RMSE",
  tuneLength = 3,
  num.trees = 100,
  importance = "permutation"
)          

rf_model_fast$results

# Variables of Importance
plot(varImp(rf_model_fast), main = "Random Forest - Variables of Importance")

# Evaluate on dev set
dev$int_rf = predict(rf_model_fast, newdata = dev)

#calculate performance metrics on dev set
mae_rf = mae(dev$int_rate, dev$int_rf)
rmse_rf = rmse(dev$int_rate, dev$int_rf)
r2_rf = cor(dev$int_rate, dev$int_rf)^2

### Model Evaluation ####

# Gather model metrics
lm_metrics = c(RMSE = rmse_lm, MAE = mae_lm, R2 = r2_lm)
rf_metrics = c(RMSE = rmse_rf, MAE = mae_rf, R2 = r2_rf)

# Combine into data frame
model_metrics = as.data.frame(rbind(lm_metrics, rf_metrics))
model_metrics$Model = c("Linear Model (LM)", "Random Forest (RF)")

# Rerrange Columns and Round
model_metrics = model_metrics %>% 
  select(Model, RMSE, MAE, R2) %>% 
  mutate(across(c(MAE, R2, RMSE), ~ round(., 2)))

## Step 4: Predict on Test Set ####

# Use Random Forest model on test set
preds_test = predict(rf_model_fast, newdata = test)
preds_df = data.frame(int_rate = preds_test)

# Add ID
preds_df = preds_df %>% 
  mutate(ID = row_number()) %>% 
  select(ID, int_rate)
