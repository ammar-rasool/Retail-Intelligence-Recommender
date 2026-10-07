-- Schema for Retail Intelligence & Recommender Engine

-- Customers Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    country TEXT
);

-- Products Table
CREATE TABLE IF NOT EXISTS products (
    stock_code TEXT PRIMARY KEY,
    description TEXT
);

-- Invoices Table
CREATE TABLE IF NOT EXISTS invoices (
    invoice_no TEXT PRIMARY KEY,
    customer_id INTEGER,
    invoice_date TIMESTAMP,
    is_cancellation BOOLEAN DEFAULT 0,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

-- Invoice Items (Transactions) Table
CREATE TABLE IF NOT EXISTS invoice_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no TEXT,
    stock_code TEXT,
    quantity INTEGER,
    unit_price REAL,
    FOREIGN KEY(invoice_no) REFERENCES invoices(invoice_no),
    FOREIGN KEY(stock_code) REFERENCES products(stock_code)
);

-- Indexes for fast temporal querying and joins
CREATE INDEX IF NOT EXISTS idx_invoices_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_invoice_items_stock ON invoice_items(stock_code);
CREATE INDEX IF NOT EXISTS idx_invoice_items_invoice ON invoice_items(invoice_no);
