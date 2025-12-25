-- SQL Questions for Scientist II - Reservations (Uber)
-- This file contains SQL questions relevant for data analysis and business intelligence

-- ============================================================================
-- QUESTION 1: Calculate Conversion Funnel Metrics
-- Difficulty: Easy
-- Topic: Business Analytics
-- ============================================================================

-- Calculate conversion rates at each stage of the booking funnel
-- Stages: saw_reserve -> clicked_reserve -> started_booking -> completed_booking -> completed_trip

WITH funnel_stages AS (
    SELECT 
        user_id,
        -- Stage indicators
        CASE WHEN saw_reserve = 1 THEN 1 ELSE 0 END as stage_awareness,
        CASE WHEN clicked_reserve = 1 THEN 1 ELSE 0 END as stage_interest,
        CASE WHEN started_booking = 1 THEN 1 ELSE 0 END as stage_consideration,
        CASE WHEN completed_booking = 1 THEN 1 ELSE 0 END as stage_booking,
        CASE WHEN completed_trip = 1 THEN 1 ELSE 0 END as stage_completion
    FROM user_interactions
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
),
stage_counts AS (
    SELECT 
        COUNT(*) as total_users,
        SUM(stage_awareness) as awareness_count,
        SUM(stage_interest) as interest_count,
        SUM(stage_consideration) as consideration_count,
        SUM(stage_booking) as booking_count,
        SUM(stage_completion) as completion_count
    FROM funnel_stages
)
SELECT 
    total_users,
    awareness_count,
    ROUND(100.0 * awareness_count / total_users, 2) as awareness_rate,
    
    interest_count,
    ROUND(100.0 * interest_count / awareness_count, 2) as interest_conversion_rate,
    
    consideration_count,
    ROUND(100.0 * consideration_count / interest_count, 2) as consideration_conversion_rate,
    
    booking_count,
    ROUND(100.0 * booking_count / consideration_count, 2) as booking_conversion_rate,
    
    completion_count,
    ROUND(100.0 * completion_count / booking_count, 2) as completion_conversion_rate,
    
    -- Overall conversion rate
    ROUND(100.0 * completion_count / total_users, 2) as overall_conversion_rate
FROM stage_counts;

-- ============================================================================
-- QUESTION 2: Calculate Cohort Retention
-- Difficulty: Medium
-- Topic: Business Analytics
-- ============================================================================

-- Calculate monthly cohort retention rates

WITH user_cohorts AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', MIN(created_at)) as cohort_month
    FROM trips
    GROUP BY user_id
),
cohort_periods AS (
    SELECT 
        t.user_id,
        uc.cohort_month,
        DATE_TRUNC('month', t.trip_date) as trip_month,
        EXTRACT(MONTH FROM AGE(t.trip_month, uc.cohort_month)) as period_number
    FROM trips t
    INNER JOIN user_cohorts uc ON t.user_id = uc.user_id
    WHERE t.trip_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '12 months')
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT user_id) as cohort_size
    FROM user_cohorts
    WHERE cohort_month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '12 months')
    GROUP BY cohort_month
),
period_user_counts AS (
    SELECT 
        cohort_month,
        period_number,
        COUNT(DISTINCT user_id) as users_in_period
    FROM cohort_periods
    GROUP BY cohort_month, period_number
)
SELECT 
    cs.cohort_month,
    cs.cohort_size,
    COALESCE(puc.period_number, 0) as period,
    COALESCE(puc.users_in_period, 0) as users_active,
    ROUND(100.0 * COALESCE(puc.users_in_period, 0) / cs.cohort_size, 2) as retention_rate
FROM cohort_sizes cs
LEFT JOIN period_user_counts puc ON cs.cohort_month = puc.cohort_month
ORDER BY cs.cohort_month, puc.period_number;

-- ============================================================================
-- QUESTION 3: Calculate A/B Test Results
-- Difficulty: Medium
-- Topic: Statistical Analysis
-- ============================================================================

-- Compare conversion rates between treatment and control groups with statistical test

WITH experiment_results AS (
    SELECT 
        experiment_group,
        COUNT(*) as total_users,
        SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) as conversions,
        AVG(CASE WHEN converted = 1 THEN 1.0 ELSE 0.0 END) as conversion_rate
    FROM experiments
    WHERE experiment_name = 'new_pricing_algorithm'
        AND date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY experiment_group
),
stats AS (
    SELECT 
        SUM(CASE WHEN experiment_group = 'control' THEN total_users ELSE 0 END) as n_control,
        SUM(CASE WHEN experiment_group = 'treatment' THEN total_users ELSE 0 END) as n_treatment,
        SUM(CASE WHEN experiment_group = 'control' THEN conversions ELSE 0 END) as conversions_control,
        SUM(CASE WHEN experiment_group = 'treatment' THEN conversions ELSE 0 END) as conversions_treatment
    FROM experiment_results
)
SELECT 
    'Control' as group_name,
    n_control as sample_size,
    conversions_control as conversions,
    ROUND(100.0 * conversions_control / n_control, 2) as conversion_rate_pct
FROM stats
UNION ALL
SELECT 
    'Treatment' as group_name,
    n_treatment as sample_size,
    conversions_treatment as conversions,
    ROUND(100.0 * conversions_treatment / n_treatment, 2) as conversion_rate_pct
FROM stats
UNION ALL
SELECT 
    'Difference' as group_name,
    NULL as sample_size,
    conversions_treatment - conversions_control as conversions,
    ROUND(100.0 * (conversions_treatment::FLOAT / n_treatment - 
                   conversions_control::FLOAT / n_control), 2) as conversion_rate_pct
FROM stats;

-- ============================================================================
-- QUESTION 4: Driver Availability Analysis
-- Difficulty: Medium
-- Topic: Supply Analytics
-- ============================================================================

-- Calculate driver availability metrics by hour and day of week

SELECT 
    EXTRACT(DOW FROM timestamp) as day_of_week,
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    COUNT(DISTINCT driver_id) as unique_drivers,
    AVG(available_drivers) as avg_available_drivers,
    AVG(active_trips) as avg_active_trips,
    AVG(available_drivers::FLOAT / NULLIF(total_drivers, 0)) as availability_rate,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY available_drivers) as median_available_drivers
FROM driver_supply_data
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY EXTRACT(DOW FROM timestamp), EXTRACT(HOUR FROM timestamp)
ORDER BY day_of_week, hour_of_day;

-- ============================================================================
-- QUESTION 5: Rider Lifetime Value (LTV)
-- Difficulty: Medium
-- Topic: Business Analytics
-- ============================================================================

-- Calculate customer lifetime value segmented by acquisition channel

WITH rider_metrics AS (
    SELECT 
        r.user_id,
        r.acquisition_channel,
        MIN(t.trip_date) as first_trip_date,
        MAX(t.trip_date) as last_trip_date,
        COUNT(DISTINCT t.trip_id) as total_trips,
        SUM(t.revenue) as total_revenue,
        SUM(t.revenue) / COUNT(DISTINCT t.trip_id) as avg_trip_value,
        EXTRACT(DAYS FROM (MAX(t.trip_date) - MIN(t.trip_date))) as customer_lifetime_days
    FROM riders r
    LEFT JOIN trips t ON r.user_id = t.rider_id
    WHERE r.created_at >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY r.user_id, r.acquisition_channel
)
SELECT 
    acquisition_channel,
    COUNT(*) as total_riders,
    AVG(total_trips) as avg_trips_per_rider,
    AVG(total_revenue) as avg_ltv,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_revenue) as median_ltv,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_revenue) as p95_ltv,
    AVG(avg_trip_value) as avg_trip_value,
    AVG(customer_lifetime_days) as avg_lifetime_days
FROM rider_metrics
GROUP BY acquisition_channel
ORDER BY avg_ltv DESC;

-- ============================================================================
-- QUESTION 6: Time Series Analysis - Booking Trends
-- Difficulty: Easy
-- Topic: Time Series Analysis
-- ============================================================================

-- Calculate booking trends by day with moving averages

SELECT 
    DATE(booking_time) as booking_date,
    COUNT(*) as daily_bookings,
    SUM(revenue) as daily_revenue,
    AVG(wait_time_minutes) as avg_wait_time,
    -- 7-day moving average
    AVG(COUNT(*)) OVER (
        ORDER BY DATE(booking_time) 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as bookings_7day_ma,
    -- Week-over-week change
    LAG(COUNT(*), 7) OVER (ORDER BY DATE(booking_time)) as bookings_same_day_last_week,
    ROUND(100.0 * (COUNT(*) - LAG(COUNT(*), 7) OVER (ORDER BY DATE(booking_time))) / 
          NULLIF(LAG(COUNT(*), 7) OVER (ORDER BY DATE(booking_time)), 0), 2) as wow_change_pct
FROM bookings
WHERE booking_time >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(booking_time)
ORDER BY booking_date;

-- ============================================================================
-- QUESTION 7: Matching Efficiency Analysis
-- Difficulty: Medium
-- Topic: Operations Analytics
-- ============================================================================

-- Analyze driver-rider matching efficiency metrics

SELECT 
    DATE(match_time) as match_date,
    COUNT(*) as total_matches,
    AVG(pickup_distance_km) as avg_pickup_distance,
    AVG(time_to_pickup_minutes) as avg_time_to_pickup,
    SUM(CASE WHEN driver_accepted = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as acceptance_rate,
    SUM(CASE WHEN trip_completed = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as completion_rate,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pickup_distance_km) as median_pickup_distance,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY pickup_distance_km) as p95_pickup_distance
FROM matches
WHERE match_time >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(match_time)
ORDER BY match_date;

-- ============================================================================
-- QUESTION 8: Feature Engineering for ML - Aggregated Features
-- Difficulty: Medium-Hard
-- Topic: Feature Engineering
-- ============================================================================

-- Create aggregated features for a trip cancellation prediction model

WITH rider_features AS (
    SELECT 
        rider_id,
        COUNT(*) as total_trips,
        SUM(CASE WHEN cancelled = 1 THEN 1 ELSE 0 END) as total_cancellations,
        AVG(CASE WHEN cancelled = 1 THEN 1.0 ELSE 0.0 END) as cancellation_rate,
        AVG(trip_distance) as avg_trip_distance,
        AVG(revenue) as avg_revenue,
        MAX(trip_date) as last_trip_date
    FROM trips
    WHERE trip_date < CURRENT_DATE
    GROUP BY rider_id
),
market_features AS (
    SELECT 
        DATE(trip_date) as trip_date,
        city_id,
        AVG(driver_supply_index) as avg_supply_index,
        AVG(demand_index) as avg_demand_index,
        AVG(surge_multiplier) as avg_surge
    FROM trips t
    JOIN market_data m ON DATE(t.trip_date) = m.date AND t.city_id = m.city_id
    WHERE trip_date >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY DATE(trip_date), city_id
)
SELECT 
    t.trip_id,
    t.rider_id,
    t.city_id,
    t.trip_distance,
    t.request_time,
    t.estimated_wait_time,
    
    -- Rider features
    COALESCE(rf.total_trips, 0) as rider_total_trips,
    COALESCE(rf.cancellation_rate, 0) as rider_historical_cancel_rate,
    COALESCE(rf.avg_trip_distance, 0) as rider_avg_trip_distance,
    EXTRACT(DAYS FROM (t.request_time - COALESCE(rf.last_trip_date, t.request_time))) as days_since_last_trip,
    
    -- Market features
    COALESCE(mf.avg_supply_index, 0) as market_supply_index,
    COALESCE(mf.avg_demand_index, 0) as market_demand_index,
    COALESCE(mf.avg_surge, 1.0) as market_avg_surge,
    
    -- Time features
    EXTRACT(DOW FROM t.request_time) as day_of_week,
    EXTRACT(HOUR FROM t.request_time) as hour_of_day,
    CASE WHEN EXTRACT(DOW FROM t.request_time) >= 5 THEN 1 ELSE 0 END as is_weekend,
    
    -- Target variable
    t.cancelled as target
FROM trips t
LEFT JOIN rider_features rf ON t.rider_id = rf.rider_id
LEFT JOIN market_features mf ON DATE(t.trip_date) = mf.trip_date AND t.city_id = mf.city_id
WHERE t.trip_date >= CURRENT_DATE - INTERVAL '30 days';

-- ============================================================================
-- QUESTION 9: Calculate Population Stability Index (PSI) for Feature Drift
-- Difficulty: Hard
-- Topic: Data Quality / MLOps
-- ============================================================================

-- Calculate PSI for trip_distance feature to detect data drift
-- This is a simplified version - full PSI calculation typically done in Python/R

WITH training_data AS (
    SELECT 
        trip_distance,
        WIDTH_BUCKET(trip_distance, 0, 50, 10) as bucket
    FROM trips
    WHERE trip_date >= CURRENT_DATE - INTERVAL '90 days'
        AND trip_date < CURRENT_DATE - INTERVAL '30 days'
),
production_data AS (
    SELECT 
        trip_distance,
        WIDTH_BUCKET(trip_distance, 0, 50, 10) as bucket
    FROM trips
    WHERE trip_date >= CURRENT_DATE - INTERVAL '7 days'
),
training_buckets AS (
    SELECT 
        bucket,
        COUNT(*)::FLOAT / (SELECT COUNT(*) FROM training_data) as expected_pct
    FROM training_data
    GROUP BY bucket
),
production_buckets AS (
    SELECT 
        bucket,
        COUNT(*)::FLOAT / (SELECT COUNT(*) FROM production_data) as actual_pct
    FROM production_data
    GROUP BY bucket
)
SELECT 
    COALESCE(tb.bucket, pb.bucket) as bucket,
    COALESCE(tb.expected_pct, 0) as expected_pct,
    COALESCE(pb.actual_pct, 0) as actual_pct,
    -- PSI component for this bucket
    (COALESCE(pb.actual_pct, 0) - COALESCE(tb.expected_pct, 0)) * 
    LN(NULLIF(COALESCE(pb.actual_pct, 0.0001) / NULLIF(COALESCE(tb.expected_pct, 0.0001), 0), 0)) as psi_component
FROM training_buckets tb
FULL OUTER JOIN production_buckets pb ON tb.bucket = pb.bucket
ORDER BY bucket;

-- ============================================================================
-- QUESTION 10: Difference-in-Differences Analysis
-- Difficulty: Medium-Hard
-- Topic: Causal Inference
-- ============================================================================

-- Calculate DiD estimator for a driver incentive program

WITH treatment_control AS (
    SELECT 
        city_id,
        CASE WHEN city_id IN (SELECT city_id FROM experiment_cities WHERE experiment = 'driver_incentive') 
             THEN 'treatment' ELSE 'control' END as group_type,
        DATE_TRUNC('week', trip_date) as week,
        CASE WHEN DATE_TRUNC('week', trip_date) >= '2024-01-15' THEN 'post' ELSE 'pre' END as period,
        COUNT(DISTINCT driver_id) as unique_drivers,
        COUNT(*) as total_trips,
        AVG(driver_earnings) as avg_driver_earnings
    FROM trips
    WHERE trip_date >= '2024-01-01' AND trip_date < '2024-02-01'
    GROUP BY city_id, group_type, week, period
),
group_period_means AS (
    SELECT 
        group_type,
        period,
        AVG(unique_drivers) as avg_drivers,
        AVG(total_trips) as avg_trips,
        AVG(avg_driver_earnings) as avg_earnings
    FROM treatment_control
    GROUP BY group_type, period
)
SELECT 
    'Treatment Pre' as metric,
    avg_drivers as value
FROM group_period_means
WHERE group_type = 'treatment' AND period = 'pre'
UNION ALL
SELECT 
    'Treatment Post',
    avg_drivers
FROM group_period_means
WHERE group_type = 'treatment' AND period = 'post'
UNION ALL
SELECT 
    'Control Pre',
    avg_drivers
FROM group_period_means
WHERE group_type = 'control' AND period = 'pre'
UNION ALL
SELECT 
    'Control Post',
    avg_drivers
FROM group_period_means
WHERE group_type = 'control' AND period = 'post'
UNION ALL
SELECT 
    'DiD Estimate',
    (SELECT avg_drivers FROM group_period_means WHERE group_type = 'treatment' AND period = 'post') -
    (SELECT avg_drivers FROM group_period_means WHERE group_type = 'treatment' AND period = 'pre') -
    (SELECT avg_drivers FROM group_period_means WHERE group_type = 'control' AND period = 'post') +
    (SELECT avg_drivers FROM group_period_means WHERE group_type = 'control' AND period = 'pre');

