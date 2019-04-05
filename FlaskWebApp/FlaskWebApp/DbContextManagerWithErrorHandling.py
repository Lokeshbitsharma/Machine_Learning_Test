import mysql.connector as mySqlConnector
import AppErrorHandling as appError

class DatabaseContextManagerWithErrorHandling:
    
    def __init__(self, dbConfig : dict) -> None:
        self.configuration = dbConfig



    def __enter__(self) -> 'cursor':
        try:
            self.conn = mySqlConnector.connect(**self.configuration)
            self.cursor = self.conn.cursor()
            return self.cursor
        except mySqlConnector.errors.InterfaceError as err:
            raise appError.ConnectionError(err)
        except mySqlConnector.errors.ProgrammingError as err:
            raise appError.CredentialsError(err)
        except Excetion as error:
            raise error


    def __exit__(self, exc_type,exc_value,exc_trace) -> None:
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        if exc_type is mySqlConnector.errors.ProgrammingError:
            raise appError.SqlError(exc_value)
        elif exc_type:
            raise exc_type(exc_value)