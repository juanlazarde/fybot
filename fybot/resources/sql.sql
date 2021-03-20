
-- download postgrsql server: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
-- uses paid TimeScaleDB: https://timescale.com
-- hacking the market:
--   Part 1. https://www.youtube.com/watch?v=MFudksxlZjk&list=PLvzuUVysUFOsrxL7UxmMrVqS8X2X0b8jd&index=1
--   Part 4. https://www.youtube.com/watch?v=P-flYBbmCws&list=PLvzuUVysUFOsrxL7UxmMrVqS8X2X0b8jd&index=4
-- optimizing sql: use-the-index-luke.com
-- cheat sheet https://websitesetup.org/sql-cheat-sheet/
--
-- https://www.postgresql.org/docs/12/sql-select.html
-- [ WITH [ RECURSIVE ] with_query [, ...] ]
-- SELECT [ ALL | DISTINCT [ ON ( expression [, ...] ) ] ]
--     [ * | expression [ [ AS ] output_name ] [, ...] ]
--     [ FROM from_item [, ...] ]
--     [ WHERE condition ]
--     [ GROUP BY grouping_element [, ...] ]
--     [ HAVING condition ]
--     [ WINDOW window_name AS ( window_definition ) [, ...] ]
--     [ { UNION | INTERSECT | EXCEPT } [ ALL | DISTINCT ] select ]
--     [ ORDER BY expression [ ASC | DESC | USING operator ] [ NULLS { FIRST | LAST } ] [, ...] ]
--     [ LIMIT { count | ALL } ]
--     [ OFFSET start [ ROW | ROWS ] ]
--     [ FETCH { FIRST | NEXT } [ count ] { ROW | ROWS } ONLY ]
--     [ FOR { UPDATE | NO KEY UPDATE | SHARE | KEY SHARE } [ OF table_name [, ...] ] [ NOWAIT | SKIP LOCKED ] [...] ]
--
-- where from_item can be one of:
--
--     [ ONLY ] table_name [ * ] [ [ AS ] alias [ ( column_alias [, ...] ) ] ]
--                 [ TABLESAMPLE sampling_method ( argument [, ...] ) [ REPEATABLE ( seed ) ] ]
--     [ LATERAL ] ( select ) [ AS ] alias [ ( column_alias [, ...] ) ]
--     with_query_name [ [ AS ] alias [ ( column_alias [, ...] ) ] ]
--     [ LATERAL ] function_name ( [ argument [, ...] ] )
--                 [ WITH ORDINALITY ] [ [ AS ] alias [ ( column_alias [, ...] ) ] ]
--     [ LATERAL ] function_name ( [ argument [, ...] ] ) [ AS ] alias ( column_definition [, ...] )
--     [ LATERAL ] function_name ( [ argument [, ...] ] ) AS ( column_definition [, ...] )
--     [ LATERAL ] ROWS FROM( function_name ( [ argument [, ...] ] ) [ AS ( column_definition [, ...] ) ] [, ...] )
--                 [ WITH ORDINALITY ] [ [ AS ] alias [ ( column_alias [, ...] ) ] ]
--     from_item [ NATURAL ] join_type from_item [ ON join_condition | USING ( join_column [, ...] ) ]
--
-- and grouping_element can be one of:
--
--     ( )
--     expression
--     ( expression [, ...] )
--     ROLLUP ( { expression | ( expression [, ...] ) } [, ...] )
--     CUBE ( { expression | ( expression [, ...] ) } [, ...] )
--     GROUPING SETS ( grouping_element [, ...] )
--
-- and with_query is:
--
--     with_query_name [ ( column_name [, ...] ) ] AS [ [ NOT ] MATERIALIZED ] ( select | values | insert | update | delete )
--
-- TABLE [ ONLY ] table_name [ * ]

-- Create tables

DROP TABLE IF EXISTS signals;
DROP TABLE IF EXISTS mention;
DROP TABLE IF EXISTS fundamentals;
DROP TABLE IF EXISTS portfolio;
DROP TABLE IF EXISTS price_history;
DROP TABLE IF EXISTS last_update;
DROP TABLE IF EXISTS rejected_symbols;
DROP TABLE IF EXISTS symbols;

CREATE TABLE IF NOT EXISTS symbols (
    id VARCHAR(12) UNIQUE PRIMARY KEY,
    symbol VARCHAR(8) NOT NULL,
    security VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    id VARCHAR(12) UNIQUE PRIMARY KEY,
    yfinance TEXT,
    CONSTRAINT fk_fundamentals
        FOREIGN KEY (id)
        REFERENCES symbols(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio (
    id VARCHAR(12) NOT NULL,
    holding_id VARCHAR(12) NOT NULL,
    date DATE NOT NULL,
    shares NUMERIC,
    weight NUMERIC,
    PRIMARY KEY (id, holding_id),
    CONSTRAINT fk_portfolio
        FOREIGN KEY (id)
        REFERENCES symbols(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_holding
        FOREIGN KEY (holding_id)
        REFERENCES symbols(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS price_history (
    symbol_id VARCHAR(12) NOT NULL,
    date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open NUMERIC (10, 2) NOT NULL,
    high NUMERIC (10, 2) NOT NULL,
    low NUMERIC (10, 2) NOT NULL,
    close NUMERIC (10, 2) NOT NULL,
    adj_close NUMERIC (10, 2) NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (symbol_id, date),
    CONSTRAINT fk_symbol
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS mention (
    symbol_id VARCHAR(12) NOT NULL,
    date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    mentions TEXT NOT NULL,
    source VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL,
    PRIMARY KEY (symbol_id, date),
    CONSTRAINT fk_mention
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE INDEX ON price_history (symbol_id, date DESC);

CREATE TABLE IF NOT EXISTS last_update (
    tbl VARCHAR(32) NOT NULL UNIQUE PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    symbol_id VARCHAR(12) NOT NULL,
    study VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (symbol_id, study),
    CONSTRAINT fk_signal
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id)
);

CREATE TABLE IF NOT EXISTS rejected_symbols (
    symbol_id VARCHAR(12) UNIQUE NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol_id, date),
    CONSTRAINT fk_rejects
        FOREIGN KEY (symbol_id)
        REFERENCES symbols(id)
);


-- DROP FUNCTION IF EXISTS trigger_set_timestamp() CASCADE;
-- CREATE OR REPLACE FUNCTION trigger_set_timestamp()
-- RETURNS TRIGGER AS $$
-- BEGIN
--   NEW.date = NOW();
--   RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;
--
-- CREATE TRIGGER set_timestamp
-- BEFORE UPDATE ON symbols
-- FOR EACH ROW
-- EXECUTE PROCEDURE trigger_set_timestamp();

--
-- ALTER TABLE fundamentals ADD content TEXT;
--
-- ALTER TABLE symbols ALTER COLUMN symbol TYPE VARCHAR(10);
--
-- ALTER TABLE symbols ADD security VARCHAR(255);
-- ALTER TABLE fundamentals RENAME COLUMN company_name TO security;
-- ALTER TABLE fundamentals ALTER COLUMN security TYPE VARCHAR(255);
--
-- INSERT INTO fundamentals (id, content)
--     VALUES ((SELECT id FROM symbols WHERE symbol = 'MMM'), 'hello');
--
-- DROP TABLE IF EXISTS fundamentals;
-- DROP TABLE IF EXISTS portfolio;
-- DROP TABLE IF EXISTS price_history;
-- DROP TABLE IF EXISTS symbols;
--
-- DROP TABLE IF EXISTS fundamentals, portfolio, price_history;
-- DROP TABLE IF EXISTS symbols;
--

-- Queries

SELECT (SELECT date
        FROM last_update
        WHERE tbl = 'symbols')
AS last_update;

SELECT date
FROM last_update
WHERE tbl = 'symbols';

SELECT symbols.symbol,
       symbols.security,
       fundamentals.is_etf
FROM symbols
    INNER JOIN fundamentals
    ON symbols.id = fundamentals.id
ORDER BY symbols.symbol ASC;

SELECT * FROM fundamentals WHERE is_etf = true AND symbol = 'ARKQ';

SELECT holding_id
FROM portfolio
WHERE date = '2021-02-18'
AND holding_id NOT IN (
    SELECT DISTINCT(holding_id)
    FROM portfolio
    WHERE date = '2021-02-17'
    );

SELECT COUNT(*) FROM mention;

SELECT COUNT(*) AS num_mentions, symbol_id, symbol
FROM mention JOIN symbols ON symbols.id = mention.symbol_id
GROUP BY symbol_id, symbol
HAVING COUNT(*) > 1
ORDER BY num_mentions DESC;

SELECT id FROM symbols
LEFT JOIN rejected_symbols
ON symbols.id = rejected_symbols.symbol_id;

-- CREATE DATABASE source;

INSERT INTO symbols (symbol, security) VALUES ('abcd', 'def');

-- Delete content


TRUNCATE symbols, last_update CASCADE;

REINDEX TABLE symbols;
REINDEX TABLE fundamentals;
REINDEX TABLE price_history;
REINDEX TABLE portfolio;


-- 'upsert' Update or Insert content
-- INSERT INTO spider_count (spider, tally) VALUES ('Googlebot', 1);
-- UPDATE spider_count SET tally=tally+1 WHERE date='today' AND spider='Googlebot';
-- results in:
-- WITH upsert AS (UPDATE spider_count SET tally=tally+1 WHERE date='today' AND spider='Googlebot' RETURNING *)
--     INSERT INTO spider_count (spider, tally) SELECT 'Googlebot', 1 WHERE NOT EXISTS (SELECT * FROM upsert)

UPDATE last_update
        SET date=NOW()
        WHERE "table"='hello4'
        RETURNING *;

INSERT INTO last_update (tbl)
VALUES ('hello3')
ON CONFLICT (tbl)
DO UPDATE SET date = NOW();

-- INSERT INTO "PDPC".collection (q1, q2, q3, q4, dg_fkey)
--       VALUES (:q1, :q2, :q3, :q4, :dg_no)
--       ON CONFLICT(dg_fkey) DO UPDATE
--       SET q1=:q1, q2=:q2, q3=:q3, q4=:q4

INSERT INTO symbols (symbol, security)
VALUES ('B', 'GOOD BYE')
ON CONFLICT (symbol) DO UPDATE
SET security=excluded.security;

INSERT INTO fundamentals (id, is_etf, content)
VALUES ((
    SELECT id
    FROM symbols
    WHERE symbol = 'A'),
        TRUE,
        'HELLO FUNDAMENTAL 2')
ON CONFLICT (id)
DO UPDATE
SET is_etf=excluded.is_etf, content=excluded.content;

INSERT INTO fundamentals (id, content)
VALUES ((
    SELECT id
    FROM symbols
    WHERE symbol = 'A'),
        'HELLO FUNDAMENTAL 2')
ON CONFLICT (id)
DO UPDATE
SET content=excluded.content;

INSERT INTO rejected_symbols (symbol_id)
VALUES ((SELECT id FROM symbols WHERE symbol = 'A'))
ON CONFLICT (symbol_id, date)
DO UPDATE SET date = NOW();

SELECT id, symbol, security
FROM symbols
FULL OUTER JOIN rejected_symbols
ON symbols.id = rejected_symbols.symbol_id
ORDER BY symbol;

SELECT symbol, security
FROM symbols
WHERE symbols.id
NOT IN (SELECT symbol_id
        FROM rejected_symbols);

SELECT symbols.symbol, fundamentals.yfinance
FROM fundamentals
INNER JOIN symbols
ON fundamentals.id = symbols.id
WHERE symbols.id
NOT IN (SELECT symbol_id FROM rejected_symbols)
ORDER BY symbols.symbol;

SELECT price_history.date,
    symbols.symbol,
    price_history.open,
    price_history.high,
    price_history.low,
    price_history.close,
    price_history.adj_close,
    price_history.volume
FROM price_history
INNER JOIN symbols
ON price_history.symbol_id = symbols.id
WHERE symbols.id
NOT IN (SELECT symbol_id FROM rejected_symbols)
ORDER BY symbols.symbol, price_history.date;

INSERT INTO signals (symbol_id, study, value)
VALUES ((SELECT id FROM symbols WHERE symbol='A'), 'ema', 'True')
ON CONFLICT (symbol_id, study) DO UPDATE
SET symbol_id = excluded.symbol_id,
 study = excluded.study,
 value = excluded.value;
