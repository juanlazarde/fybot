-- download PostgreSQL server: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

-- Run this in the terminal
-- python app.py create_table

-- DATABASE: source

-- DROP DATABASE IF EXISTS source;

CREATE DATABASE source
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'English_United States.1252'
    LC_CTYPE = 'English_United States.1252'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- SCHEMA: public

-- DROP SCHEMA public ;

CREATE SCHEMA public
    AUTHORIZATION postgres;

COMMENT ON SCHEMA public
    IS 'standard public schema';

GRANT ALL ON SCHEMA public TO PUBLIC;

GRANT ALL ON SCHEMA public TO postgres;

-- Delete the tables

DROP TABLE IF EXISTS signals;
DROP TABLE IF EXISTS mention;
DROP TABLE IF EXISTS fundamentals;
DROP TABLE IF EXISTS portfolio;
DROP TABLE IF EXISTS price_history;
DROP TABLE IF EXISTS last_update;
DROP TABLE IF EXISTS rejected_symbols;
DROP TABLE IF EXISTS symbols;

-- Create the tables

CREATE TABLE IF NOT EXISTS symbols
(
    id       VARCHAR(12) UNIQUE PRIMARY KEY,
    symbol   VARCHAR(8) NOT NULL,
    security VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS fundamentals
(
    symbol_id VARCHAR(12)              NOT NULL,
    source    VARCHAR(255)             NOT NULL,
    var       VARCHAR(255)             NOT NULL,
    val       TEXT,
    date      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol_id, source, var),
    CONSTRAINT fk_fundamentals
        FOREIGN KEY (symbol_id)
            REFERENCES symbols (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio
(
    id         VARCHAR(12) NOT NULL,
    holding_id VARCHAR(12) NOT NULL,
    date       DATE        NOT NULL,
    shares     NUMERIC,
    weight     NUMERIC,
    PRIMARY KEY (id, holding_id),
    CONSTRAINT fk_portfolio
        FOREIGN KEY (id)
            REFERENCES symbols (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
    CONSTRAINT fk_holding
        FOREIGN KEY (holding_id)
            REFERENCES symbols (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS price_history
(
    symbol_id VARCHAR(12)                 NOT NULL,
    date      TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open      NUMERIC(10, 2)              NOT NULL,
    high      NUMERIC(10, 2)              NOT NULL,
    low       NUMERIC(10, 2)              NOT NULL,
    close     NUMERIC(10, 2)              NOT NULL,
    volume    INTEGER                     NOT NULL,
    PRIMARY KEY (symbol_id, date),
    CONSTRAINT fk_symbol
        FOREIGN KEY (symbol_id)
            REFERENCES symbols (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
);

CREATE INDEX ON price_history (symbol_id, date DESC);

CREATE TABLE IF NOT EXISTS mention
(
    symbol_id VARCHAR(12)                 NOT NULL,
    date      TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    mentions  TEXT                        NOT NULL,
    source    VARCHAR(255)                NOT NULL,
    url       VARCHAR(255)                NOT NULL,
    PRIMARY KEY (symbol_id, date),
    CONSTRAINT fk_mention
        FOREIGN KEY (symbol_id)
            REFERENCES symbols (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS last_update
(
    tbl  VARCHAR(32)              NOT NULL UNIQUE PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals
(
    symbol_id VARCHAR(12)  NOT NULL,
    study     VARCHAR(255) NOT NULL,
    value     TEXT         NOT NULL,
    PRIMARY KEY (symbol_id, study),
    CONSTRAINT fk_signal
        FOREIGN KEY (symbol_id)
            REFERENCES symbols (id)
);

CREATE TABLE IF NOT EXISTS rejected_symbols
(
    symbol_id VARCHAR(12) UNIQUE       NOT NULL,
    date      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol_id, date),
    CONSTRAINT fk_rejects
        FOREIGN KEY (symbol_id)
            REFERENCES symbols (id)
);
