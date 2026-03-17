-- Monthly Signup Growth (Window Function)
SELECT 
    DATE_TRUNC('month', signup_date) AS signup_month,
    COUNT(account_id) AS new_signups,
    LAG(COUNT(account_id)) OVER (ORDER BY DATE_TRUNC('month', signup_date)) AS prev_month_signups
FROM accounts
GROUP BY 1;

-- Support Health by Industry
SELECT 
    a.industry,
    AVG(s.satisfaction_score) AS avg_satisfaction,
    AVG(s.resolution_time_hrs) AS avg_resolution_hrs
FROM accounts a
JOIN support_tickets s ON a.account_id = s.account_id
GROUP BY a.industry
ORDER BY avg_satisfaction ASC;

-- Beta Feature Adoption by Plan Tier
SELECT 
    sub.plan_tier,
    COUNT(DISTINCT f.account_id) AS active_beta_users,
    SUM(CASE WHEN f.feature_name = 'Beta_Analytics_V2' THEN 1 ELSE 0 END) AS specific_beta_usage
FROM subscriptions sub
LEFT JOIN feature_usage f ON sub.account_id = f.account_id
GROUP BY sub.plan_tier;