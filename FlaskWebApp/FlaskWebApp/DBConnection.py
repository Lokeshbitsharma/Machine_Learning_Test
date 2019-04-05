from flask import Flask, render_template, request, escape
from vsearch import search4letters
import mysql.connector

app = Flask(__name__)

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
def view_the_log() -> 'html':
    contents = []
    with open('vsearchlog') as log:
        for line in log :
            contents.append([])
            for item in line.split('|'):
                contents[-1].append(escape(item))

    titles = {'Form Data','Remote_addr', 'User_agent','Results'}

    return render_template('viewlog.html',
                           the_title = 'View Log',
                           row_titles = titles,
                           log_rows = contents)




def log_request(req : 'flask request', res : str) -> None:
    dbconfig = {'host': 'localhost',
            'user': 'Lokesh',
            'passwd': '*****',
            'database': 'testdb'}
    conn =  mysql.connector.connect(**dbconfig)
    cursor = conn.cursor(buffered = True)

    _SQL = """insert into log 
    (phrase,letters,ip,browser_setting,results)
    values (%s,%s,%s,%s,%s)""" 

    cursor.execute(_SQL,
                   (req.form['phrase'],
                    req.form['letters'],
                    req.remote_addr,
                    req.user_agent.browser,res,))


    conn.commit()
    cursor.close()
    conn.close()
      
    
if __name__ == "__main__":
    app.run(debug = True)
