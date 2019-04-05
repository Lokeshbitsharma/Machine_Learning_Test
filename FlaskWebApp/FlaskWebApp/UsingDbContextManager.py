from flask import Flask, render_template, request, escape, session
from vsearch import search4letters
import mysql.connector
from DbContextManager import DatabaseContextManager
from Checker import checker_logged_in

app = Flask(__name__)

app.secret_key = 'ThisIsMySecretKey'

app.config['dbconfig'] = {'host': 'localhost',
            'user': 'Lokesh',
            'passwd': '*****',
            'database': 'testdb'}

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
    log_request(request,results)
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
    with DatabaseContextManager(app.config['dbconfig']) as cursor:
        _SQL = """SELECT phrase,letters,ip,browser_setting,results from log"""
        cursor.execute(_SQL)
        contents = cursor.fetchall()
        titles = ['Phrase','Letters','IP','Browser','Results']
        return render_template('viewlog.html',
                           the_title = 'View Log',
                           row_titles = titles,
                           log_rows = contents)




def log_request(req : 'flask request', res : str) -> None:     
    with DatabaseContextManager(app.config['dbconfig']) as cursor:
        _SQL = """insert into log
        (phrase,letters,ip,browser_setting,results)
        values (%s,%s,%s,%s,%s)""" 

        cursor.execute(_SQL,
                   (req.form['phrase'],
                    req.form['letters'],
                    req.remote_addr,
                    req.user_agent.browser,res,))
    
if __name__ == "__main__":
    app.run(debug = True)

