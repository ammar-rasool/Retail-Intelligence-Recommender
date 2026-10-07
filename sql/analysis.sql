-- Analytical queries for data understanding

-- 1. Monthly Revenue & Order Volume
SELECT 
    strftime('%Y-%m', invoice_date) AS invoice_month,
    COUNT(DISTINCT i.invoice_no) AS total_orders,
    COUNT(DISTINCT i.customer_id) AS active_customers,
    SUM(ii.quantity * ii.unit_price) AS total_revenue
FROM invoices i
JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
WHERE i.is_cancellation = 0
GROUP BY invoice_month
ORDER BY invoice_month;

-- 2. Cohort Retention Base (First purchase month per customer)
WITH customer_cohort AS (
    SELECT 
        customer_id,
        strftime('%Y-%m', MIN(invoice_date)) AS cohort_month
    FROM invoices
    WHERE is_cancellation = 0
    GROUP BY customer_id
),
monthly_activity AS (
    SELECT DISTINCT
        customer_id,
        strftime('%Y-%m', invoice_date) AS activity_month
    FROM invoices
    WHERE is_cancellation = 0
)
SELECT 
    c.cohort_month,
    m.activity_month,
    COUNT(DISTINCT c.customer_id) AS active_customers
FROM customer_cohort c
JOIN monthly_activity m ON c.customer_id = m.customer_id
GROUP BY c.cohort_month, m.activity_month
ORDER BY c.cohort_month, m.activity_month;

-- 3. Top Popular Products
SELECT 
    p.stock_code,
    p.description,
    SUM(ii.quantity) AS total_quantity_sold,
    COUNT(DISTINCT i.invoice_no) AS times_purchased
FROM products p
JOIN invoice_items ii ON p.stock_code = ii.stock_code
JOIN invoices i ON ii.invoice_no = i.invoice_no
WHERE i.is_cancellation = 0
GROUP BY p.stock_code
ORDER BY total_quantity_sold DESC
LIMIT 100;
