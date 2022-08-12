# Deploy_front/app.py

# Imports
from flask import Flask
from run_backend import *

app = Flask(__name__)


def start_backend(curr, df):
    """Extracts data from a database and returns a list with data to display in front-end"""

    curr.execute(""" SELECT * 
                    FROM animes_app 
                    ORDER BY score DESC 
                    LIMIT 20
                    """)

    video = pd.DataFrame(curr.fetchall())
    video.columns = ['anime_id', 'title', 'genre', 'num_episodes', 'picture', 'score']
    
    predictions = []
    for i, row in video.iterrows():
        predictions.append((row['picture'], row['title'], float(row['score'])))
    
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    predictions_formatted = []
    for e in predictions:
        predictions_formatted.append("<tr><th><a href=\"{picture}\">{title}</a></th><th>{score}</th></tr>".format(picture=e[0], title=e[1], score=e[2]))
        
    return ''.join(predictions_formatted)
    

# Database Credentials
host_name = '****'
username = '****'
password = '****'
dbname = '****'
port = 3306

conn = connect_to_db(host_name, username, password, dbname, port)
curr = conn.cursor()


@app.route("/")
def main_page():
    anime_list = start_backend(curr, dataframe)

    return """
    <head><h1> Anime Recommender </h1></head>
    <body>
    <table>
        {}
    </table>
    <body>""".format(anime_list)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')