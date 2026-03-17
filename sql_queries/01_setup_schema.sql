-- Create the accounts table
CREATE TABLE accounts (
    account_id VARCHAR(50) PRIMARY KEY,
    company_name VARCHAR(100),
    industry VARCHAR(50),
    region VARCHAR(50),
    signup_date DATE
);

-- Create the subscriptions table
CREATE TABLE subscriptions (
    subscription_id VARCHAR(50) PRIMARY KEY,
    account_id VARCHAR(50),
    plan_tier VARCHAR(20),
    monthly_charges DECIMAL(10, 2),
    number_of_seats INT,
    contract_start_date DATE,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Create the feature_usage table
CREATE TABLE feature_usage (
    usage_id SERIAL PRIMARY KEY,
    account_id VARCHAR(50),
    usage_date DATE,
    feature_name VARCHAR(50),
    usage_minutes INT,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Create the support_tickets table
CREATE TABLE support_tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    account_id VARCHAR(50),
    created_at TIMESTAMP,
    resolution_time_hrs DECIMAL(5, 2),
    satisfaction_score INT,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Create the churn_events table
CREATE TABLE churn_events (
    event_id SERIAL PRIMARY KEY,
    account_id VARCHAR(50),
    churn_date DATE,
    reason_code VARCHAR(50),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);