-- Active: 1768744757453@@127.0.0.1@3306
CREATE TABLE sales (
    order_id INTEGER PRIMARY KEY,
    order_date TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    product TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    region TEXT NOT NULL
);

-- ============================================
-- 2. inserting data
-- ============================================
INSERT INTO sales (order_id, order_date, customer_id, product, category, price, quantity, region) VALUES
(1001, '2024-01-05', 'C001', 'Laptop Pro', 'Electronics', 1200, 1, 'North'),
(1002, '2024-01-06', 'C002', 'Wireless Mouse', 'Electronics', 25, 2, 'South'),
(1003, '2024-01-07', 'C003', 'Office Chair', 'Furniture', 180, 1, 'East'),
(1004, '2024-01-08', 'C001', 'USB-C Hub', 'Electronics', 45, 1, 'North'),
(1005, '2024-01-10', 'C004', 'Standing Desk', 'Furniture', 350, 1, 'West'),
(1006, '2024-01-11', 'C005', 'Notebook', 'Stationery', 5, 10, 'South'),
(1007, '2024-01-12', 'C006', 'Pen Set', 'Stationery', 12, 3, 'East'),
(1008, '2024-01-14', 'C002', 'Keyboard', 'Electronics', 75, 1, 'South'),
(1009, '2024-01-15', 'C007', 'Monitor 27"', 'Electronics', 300, 2, 'North'),
(1010, '2024-01-17', 'C008', 'Desk Lamp', 'Furniture', 40, 2, 'West'),
(1011, '2024-02-01', 'C001', 'Laptop Pro', 'Electronics', 1200, 1, 'North'),
(1012, '2024-02-03', 'C009', 'Office Chair', 'Furniture', 180, 2, 'South'),
(1013, '2024-02-05', 'C010', 'Notebook', 'Stationery', 5, 20, 'East'),
(1014, '2024-02-06', 'C003', 'Keyboard', 'Electronics', 75, 1, 'East'),
(1015, '2024-02-08', 'C004', 'Monitor 27"', 'Electronics', 300, 1, 'West'),
(1016, '2024-02-10', 'C005', 'Desk Lamp', 'Furniture', 40, 1, 'South'),
(1017, '2024-02-12', 'C006', 'Wireless Mouse', 'Electronics', 25, 3, 'East'),
(1018, '2024-02-15', 'C007', 'Standing Desk', 'Furniture', 350, 1, 'North'),
(1019, '2024-02-18', 'C008', 'Pen Set', 'Stationery', 12, 5, 'West'),
(1020, '2024-02-20', 'C002', 'Laptop Pro', 'Electronics', 1200, 1, 'South'),
(1021, '2024-03-01', 'C009', 'USB-C Hub', 'Electronics', 45, 2, 'South'),
(1022, '2024-03-03', 'C010', 'Office Chair', 'Furniture', 180, 1, 'East'),
(1023, '2024-03-05', 'C001', 'Monitor 27"', 'Electronics', 300, 1, 'North'),
(1024, '2024-03-07', 'C003', 'Notebook', 'Stationery', 5, 15, 'East'),
(1025, '2024-03-09', 'C004', 'Keyboard', 'Electronics', 75, 2, 'West'),
(1026, '2024-03-12', 'C005', 'Standing Desk', 'Furniture', 350, 1, 'South'),
(1027, '2024-03-14', 'C006', 'Laptop Pro', 'Electronics', 1200, 1, 'East'),
(1028, '2024-03-18', 'C007', 'Desk Lamp', 'Furniture', 40, 2, 'North'),
(1029, '2024-03-20', 'C008', 'Wireless Mouse', 'Electronics', 25, 2, 'West'),
(1030, '2024-03-22', 'C009', 'Pen Set', 'Stationery', 12, 4, 'South');

-- --------------------------------------------
-- query 1: total revenue
-- --------------------------------------------
SELECT
    SUM(price * quantity) AS total_revenue
FROM sales;

-- --------------------------------------------
-- query 2: revenue by category
-- --------------------------------------------
SELECT
    category,
    SUM(price * quantity) AS category_revenue
FROM sales
GROUP BY category
ORDER BY category_revenue DESC;

-- --------------------------------------------
-- query 3: top 5 products by revenue
-- --------------------------------------------
SELECT
    product,
    SUM(price * quantity) AS product_revenue
FROM sales
GROUP BY product
ORDER BY product_revenue DESC
LIMIT 5;

-- --------------------------------------------
-- query 4: monthly revenue
-- --------------------------------------------
SELECT
    strftime('%Y-%m', order_date) AS year_month,
    SUM(price * quantity) AS monthly_revenue
FROM sales
GROUP BY year_month
ORDER BY year_month;

-- --------------------------------------------
-- Query 5: Revenue by Region
-- --------------------------------------------
SELECT
    region,
    SUM(price * quantity) AS region_revenue
FROM sales
GROUP BY region
ORDER BY region_revenue DESC;

-- --------------------------------------------
-- query 6: customers with spending above average
-- --------------------------------------------
SELECT
    customer_id,
    SUM(price * quantity) AS total_spending
FROM sales
GROUP BY customer_id
HAVING total_spending > (
    SELECT AVG(customer_total)
    FROM (
        SELECT SUM(price * quantity) AS customer_total
        FROM sales
        GROUP BY customer_id
    )
)
ORDER BY total_spending DESC;
