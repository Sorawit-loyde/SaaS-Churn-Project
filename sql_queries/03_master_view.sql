CREATE VIEW master_customer_data AS
WITH usage_stats AS (
    SELECT 
        account_id,
        AVG(usage_minutes) AS avg_weekly_minutes,
        COUNT(DISTINCT feature_name) AS unique_features_used
    FROM feature_usage
    GROUP BY account_id
),
support_stats AS (
    SELECT 
        account_id,
        AVG(satisfaction_score) AS avg_satisfaction,
        AVG(resolution_time_hrs) AS avg_resolution_time,
        COUNT(ticket_id) AS total_tickets
    FROM support_tickets
    GROUP BY account_id
)
SELECT 
    a.account_id,
    a.industry,
    a.region,
    s.plan_tier,
    s.monthly_charges,
    s.number_of_seats,
    COALESCE(u.avg_weekly_minutes, 0) AS active_weekly_minutes,
    COALESCE(u.unique_features_used, 0) AS beta_feature_usage,
    COALESCE(sup.avg_satisfaction, 3.0) AS avg_satisfaction_score, -- Default to neutral
    COALESCE(sup.avg_resolution_time, 0) AS avg_resolution_time_hrs,
    COALESCE(sup.total_tickets, 0) AS total_support_tickets,
    CASE WHEN c.churn_date IS NOT NULL THEN 1 ELSE 0 END AS churn_label
FROM accounts a
JOIN subscriptions s ON a.account_id = s.account_id
LEFT JOIN usage_stats u ON a.account_id = u.account_id
LEFT JOIN support_stats sup ON a.account_id = sup.account_id
LEFT JOIN churn_events c ON a.account_id = c.account_id;