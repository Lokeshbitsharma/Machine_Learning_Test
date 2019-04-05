import mysql.connector

class DatabaseContextManager:
    
    def __init__(self, dbConfig : dict) -> None:
        self.configuration = dbConfig

    def __enter__(self) -> 'cursor':
        self.conn = mysql.connector.connect(**self.configuration)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type,exc_value,exc_trace) -> None:
        self.conn.commit()
        self.cursor.close()
        self.conn.close()