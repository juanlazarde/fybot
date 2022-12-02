
-- docker database initialization script
-- initialization will be skipped if a database already exists
-- the name of the database is set in docker-compose.yml with POSTGRES_DB: ${DB_NAME}

-- DOCKER-ONLY: Version is only for Docker installations. All other installations use fybot/config/create_tables.sql.
-- WHENEVER ONE IS UPDATED, CONSIDER UPDATING THE OTHER. create_tables_docker.sql <--> create_tables.sql

-- Create the schema

CREATE SCHEMA IF NOT EXISTS public
    AUTHORIZATION postgres;

COMMENT ON SCHEMA public
    IS 'standard public schema';

GRANT ALL ON SCHEMA public TO PUBLIC;

GRANT ALL ON SCHEMA public TO postgres;


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
