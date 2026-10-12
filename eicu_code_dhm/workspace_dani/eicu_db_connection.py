import os
from configobj import ConfigObj

from sqlalchemy import create_engine, event
from loguru import logger
import psycopg2

# Todo: aus dem Ding eine Klasse machen für die Connection

# Create a database connection using settings from config file


class DatabaseConnection:
    def __init__(self, config_path):
        # config = '../db/config_container.ini'
        config = config_path

        # connection info
        if os.path.isfile(config):
            config = ConfigObj(config)
            sql_username = config["username"]
            sql_password = config["password"]
            sql_host = config["host"]
            sql_port = config["port"]
            db_name = config["dbname"]
            schema_name = config["schema_name"]
        else:
            logger.debug("Setting up connection with default values")
            sql_username = "postgres"
            sql_password = "eicu"
            # sql_host = '172.25.93.63'
            sql_host = "localhost"
            sql_port = 6432
            db_name = "eicu"
            schema_name = "public,eicu_crd"

        # Connect to the eICU database
        print(f"Database: {db_name}")
        print(f"Database: {sql_username}")

        self.query_schema = "set search_path to " + schema_name + ";"
        logger.debug(f"Using following query Schema: {self.query_schema}")

        # db_credentials_url = "".join(
        #    [config['username'], ":", config['password'], "@", config['host'], ":", config['port'], "/", config['dbname']])

        db_credentials_url = "".join(
            [sql_username, ":", sql_password, "@", sql_host, ":", sql_port, "/", db_name]
        )

        self.database_url = "".join(["postgresql+psycopg2://", db_credentials_url])

        logger.info(f"Connecting to the database with database_url={self.database_url}")

        self.engine = create_engine(self.database_url)

        self.con_psy = psycopg2.connect(
            dbname=db_name, host=sql_host, port=sql_port, user=sql_username, password=sql_password
        )

        """
        @event.listens_for(self.engine, "connect")
        def set_search_path(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("SET search_path TO public,eicu_crd;")
            cursor.close()
        """
