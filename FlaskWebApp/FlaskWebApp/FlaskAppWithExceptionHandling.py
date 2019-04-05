from flask import Flask, render_template, request, session
from flask import copy_current_request_context
from DbContextManagerWithErrorHandling import DatabaseContextManagerWithErrorHandling
from vsearch import search4letters
from Checker import checker_logged_in
import AppErrorHandling as appError
from threading import Thread

app = Flask(__name__)

app.secret_key = 'ThisIsMySecretKey'

app.config['dbconfig'] = {'host': 'localhost',
            'user': 'Lokesh',
            'passwd': '*****',
            'database': 'testdb'}

# Problems that can arise with existing code
# 1. log request can fail because of no database connection
# 2. SQL query can fail 
# 3. DbContextManager connection can fail
# 4. Db may take more time to respons to the request which could be frustrating for the user





@app.route('/login')
def do_login():
    session['logged_in'] =  True
    return "You are logged in."
    
@app.route('/logout')
def do_logout():
    session.pop('logged_in')
    return 'You are logged out.'

@app.route('/search4',methods = ['POST'])
def do_search() -> str:
    title = 'Here are your results:'
    phrase = request.form['phrase']
    letters = request.form['letters']
    results = str(search4letters(phrase,letters))

    @copy_current_request_context
    def log_request(req : 'flask request', res : str) -> None:
        try:
            with DatabaseContextManagerWithErrorHandling(app.config['dbconfig']) as cursor:
                _SQL = """insert into log
                (phrase,letters,ip,browser_setting,results)
                values (%s,%s,%s,%s,%s)"""
                cursor.execute(_SQL,(req.form['phrase'],
                            req.form['letters'],
                            req.remote_addr,
                            req.user_agent.browser,res,))

        except appError.ConnectionError as err:
            print('Is ur database switched on:' , str(err))
        except appError.CredentialError as err:
            print('UserId/Password issue:' , str(err))
        except appError.SqlError as err:
            print('Is ur query correct:' , str(err))
        except Error as err:
            print('Something went wrong:', str(err))

        
        
    try:
        t = Thread(target=log_request,args=(request,results))
        t.start()
    except Exception as err:
        print('**********Logging failed:*************', str(err))
    return render_template('results.html',
                           the_title = title,
                           the_phrase = phrase,
                           the_letter = letters,
                          the_results = results)
@app.route('/')
@app.route('/entry')
def entry_page() -> 'html':
    return render_template('entry.html', the_title = "Welcome to search4letter in web:")

#@app.route('/view_log')
#def view_the_log() -> str:
#    with open('vsearchlog') as log:
#       contents = log.read()
#    return escape(contents)


@app.route('/view_log')
@checker_logged_in
def view_the_log() -> 'html':    
    try:
        with DatabaseContextManagerWithErrorHandling(app.config['dbconfig']) as cursor:
            _SQL = """SELECT phrase,letters,ip,browser_setting,results from log"""
            cursor.execute(_SQL)
            contents = cursor.fetchall()
            titles = ['Phrase','Letters','IP','Browser','Results']
            return render_template('viewlog.html',
                               the_title = 'View Log',
                               row_titles = titles,
                               log_rows = contents)
    except appError.ConnectionError as err:
        print('Is ur database switched on:' , str(err))
    except appError.CredentialError as err:
        print('UserId/Password issue:' , str(err))
    except appError.SqlError as err:
        print('Is ur query correct:' , str(err))
    except Exception as err:
        print('Something went wrong:', str(err))
    

    
if __name__ == "__main__":
    app.run()


